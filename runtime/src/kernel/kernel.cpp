// Kernel core: lock, handle table, threads, and the waiting primitives used by SVCs.
#include "kernel.h"

#include <windows.h>

#include <algorithm>
#include <cstring>

#include "sync.h"

Ctx* hw_ctx_new(u64 stack_top);
void hw_ctx_bind(Ctx* c);

namespace kern {

KernelLock g_kernel_lock;
HandleTable g_handles;
ProcessInfo g_process;

// ------------------------------------------------------------------ lock
void KernelLock::lock() {
    u32 self = GetCurrentThreadId();
    if (owner_.load(std::memory_order_relaxed) == self) {
        depth_++;
        return;
    }
    m_.lock();
    owner_.store(self, std::memory_order_relaxed);
    depth_ = 1;
}
void KernelLock::unlock() {
    if (--depth_ == 0) {
        owner_.store(0, std::memory_order_relaxed);
        m_.unlock();
    }
}
int KernelLock::release_all() {
    int d = depth_;
    depth_ = 0;
    owner_.store(0, std::memory_order_relaxed);
    m_.unlock();
    return d;
}
void KernelLock::reacquire(int depth) {
    m_.lock();
    owner_.store(GetCurrentThreadId(), std::memory_order_relaxed);
    depth_ = depth;
}

// ------------------------------------------------------------------ handles
Handle HandleTable::add(ObjectPtr obj) {
    KGuard g(g_kernel_lock);
    Handle h = (next_++ << 15) | 0x10;  // HOS-like handle values (non-zero, low bits set)
    table_[h] = std::move(obj);
    return h;
}
ObjectPtr HandleTable::get(Handle h) const {
    KGuard g(g_kernel_lock);
    if (h == kCurrentThread) {
        Thread* t = current_thread();
        // The current thread's object is kept alive by its own handle entry.
        for (auto& [k, v] : table_)
            if (v.get() == t) return v;
        return nullptr;
    }
    auto it = table_.find(h);
    return it == table_.end() ? nullptr : it->second;
}
bool HandleTable::close(Handle h) {
    KGuard g(g_kernel_lock);
    return table_.erase(h) != 0;
}

// ------------------------------------------------------------------ objects
void Object::notify_waiters() {
    for (Thread* t : waiters) wake_thread(t);
}
void ReadableEvent::signal() {
    KGuard g(g_kernel_lock);
    signaled = true;
    notify_waiters();
}

// ------------------------------------------------------------------ threads
static thread_local Thread* t_current;
static std::atomic<u64> g_next_thread_id{0x50};


static std::mutex g_threads_lock;
static std::vector<std::shared_ptr<Thread>> g_threads;

Thread* current_thread() { return t_current; }
std::vector<std::shared_ptr<Thread>> all_threads() {
    std::lock_guard<std::mutex> l(g_threads_lock);
    return g_threads;
}

static DWORD WINAPI thread_main(void* p) {
    Thread* t = (Thread*)p;
    t_current = t;
    Ctx* c = t->ctx;
    hw_ctx_bind(c);
    c->x[0] = t->arg;
    c->x[30] = 0;
    hw_call(c, t->entry);
    // Guest threads end with svcExitThread; returning from the entry is equivalent.
    exit_current_thread();
}

std::shared_ptr<Thread> create_thread(u64 entry, u64 arg, u64 stack_top, s32 priority, s32 core) {
    auto t = std::make_shared<Thread>();
    t->entry = entry;
    t->arg = arg;
    t->stack_top = stack_top;
    t->priority = priority;
    t->ideal_core = core < 0 ? 0 : core;
    t->core_mask = 1ull << t->ideal_core;
    t->id = g_next_thread_id++;
    t->tls = mem::alloc_tls_slot();
    t->ctx = hw_ctx_new(stack_top);
    t->ctx->tpidrro_el0 = t->tls;
    t->wake_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    // The host stack only holds host frames (recompiled functions keep guest state in Ctx and
    // use the guest stack for guest data), but deep guest call chains still nest host frames.
    t->host_handle = CreateThread(nullptr, 8 << 20, thread_main, t.get(), CREATE_SUSPENDED, nullptr);
    {
        std::lock_guard<std::mutex> l(g_threads_lock);
        g_threads.push_back(t);
    }
    return t;
}

void start_thread(Thread* t) {
    t->started = true;
    ResumeThread(t->host_handle);
}

void exit_current_thread() {
    Thread* t = current_thread();
    {
        KGuard g(g_kernel_lock);
        t->exited = true;
        t->notify_waiters();
    }
    // Make sure no kernel lock depth leaks out of this thread.
    ExitThread(0);
}

bool park_current_thread(s64 timeout_ns) {
    Thread* t = current_thread();
    t->waiting = true;
    int depth = g_kernel_lock.release_all();
    DWORD ms = INFINITE;
    if (timeout_ns >= 0) ms = (DWORD)std::min<s64>((timeout_ns + 999999) / 1000000, 0x7FFFFFFF);
    DWORD r = WaitForSingleObject(t->wake_event, ms);
    g_kernel_lock.reacquire(depth);
    t->waiting = false;
    return r == WAIT_OBJECT_0;
}

void wake_thread(Thread* t) { SetEvent(t->wake_event); }

// ------------------------------------------------------------------ guest mutexes (ArbitrateLock)
// The guest mutex word holds the owner's thread handle, with bit 30 set while there are waiters.
constexpr u32 kHasWaiters = 0x40000000;

struct MutexWaiter {
    Thread* thread;
    Handle handle;  // the waiter's own handle (becomes the new owner value)
    bool acquired = false;
};
static std::multimap<u64, MutexWaiter*> g_mutex_waiters;

Result arbitrate_lock(Handle owner, u64 addr, Handle self) {
    KGuard g(g_kernel_lock);
    u32 v = *(volatile u32*)addr;
    if (v != (owner | kHasWaiters)) return ResultSuccess;  // state changed; userland retries
    MutexWaiter w{current_thread(), self};
    auto it = g_mutex_waiters.insert({addr, &w});
    while (!w.acquired) park_current_thread(-1);
    (void)it;
    return ResultSuccess;
}

static void release_mutex(u64 addr) {
    auto range = g_mutex_waiters.equal_range(addr);
    if (range.first == range.second) {
        __atomic_store_n((u32*)addr, 0u, __ATOMIC_SEQ_CST);
        return;
    }
    // Hand ownership to the first waiter (highest priority = lowest number).
    auto best = range.first;
    for (auto it = range.first; it != range.second; ++it)
        if (it->second->thread->priority < best->second->thread->priority) best = it;
    MutexWaiter* w = best->second;
    g_mutex_waiters.erase(best);
    bool more = g_mutex_waiters.count(addr) != 0;
    __atomic_store_n((u32*)addr, w->handle | (more ? kHasWaiters : 0u), __ATOMIC_SEQ_CST);
    w->acquired = true;
    wake_thread(w->thread);
}

Result arbitrate_unlock(u64 addr) {
    KGuard g(g_kernel_lock);
    release_mutex(addr);
    return ResultSuccess;
}

// ------------------------------------------------------------------ condition variables
struct CvWaiter {
    Thread* thread;
    u64 mutex_addr;
    Handle handle;
    bool signaled = false;
    MutexWaiter mw{};
};
static std::multimap<u64, CvWaiter*> g_cv_waiters;

Result wait_process_wide_key(u64 mutex_addr, u64 key, Handle self, s64 timeout_ns) {
    KGuard g(g_kernel_lock);
    release_mutex(mutex_addr);
    CvWaiter w{current_thread(), mutex_addr, self};
    auto it = g_cv_waiters.insert({key, &w});
    __atomic_store_n((u32*)key, 1u, __ATOMIC_SEQ_CST);
    if (timeout_ns == 0) {
        g_cv_waiters.erase(it);
        return ResultTimedOut;
    }
    LARGE_INTEGER f, start;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&start);
    for (;;) {
        if (w.mw.acquired) return ResultSuccess;
        s64 left = -1;
        if (timeout_ns > 0) {
            LARGE_INTEGER now;
            QueryPerformanceCounter(&now);
            s64 elapsed = (s64)((now.QuadPart - start.QuadPart) * 1000000000.0 / f.QuadPart);
            left = timeout_ns - elapsed;
            if (left <= 0) break;
        }
        park_current_thread(left);
    }
    // Timed out: if still waiting on the condvar, leave it; if already queued on the mutex, leave that.
    if (!w.signaled) {
        for (auto i = g_cv_waiters.begin(); i != g_cv_waiters.end(); ++i)
            if (i->second == &w) { g_cv_waiters.erase(i); break; }
    } else if (!w.mw.acquired) {
        for (auto i = g_mutex_waiters.begin(); i != g_mutex_waiters.end(); ++i)
            if (i->second == &w.mw) { g_mutex_waiters.erase(i); break; }
    }
    return w.mw.acquired ? ResultSuccess : ResultTimedOut;
}

void signal_process_wide_key(u64 key, s32 count) {
    KGuard g(g_kernel_lock);
    s32 n = 0;
    while (count <= 0 || n < count) {
        auto range = g_cv_waiters.equal_range(key);
        if (range.first == range.second) break;
        auto best = range.first;
        for (auto it = range.first; it != range.second; ++it)
            if (it->second->thread->priority < best->second->thread->priority) best = it;
        CvWaiter* w = best->second;
        g_cv_waiters.erase(best);
        w->signaled = true;
        // Try to acquire the mutex on the waiter's behalf.
        u32 v = __atomic_load_n((u32*)w->mutex_addr, __ATOMIC_SEQ_CST);
        if (v == 0) {
            __atomic_store_n((u32*)w->mutex_addr, w->handle, __ATOMIC_SEQ_CST);
            w->mw.acquired = true;
            wake_thread(w->thread);
        } else {
            __atomic_fetch_or((u32*)w->mutex_addr, kHasWaiters, __ATOMIC_SEQ_CST);
            w->mw.thread = w->thread;
            w->mw.handle = w->handle;
            g_mutex_waiters.insert({w->mutex_addr, &w->mw});
        }
        n++;
    }
    if (g_cv_waiters.count(key) == 0) __atomic_store_n((u32*)key, 0u, __ATOMIC_SEQ_CST);
}

// ------------------------------------------------------------------ address arbiter
struct AddrWaiter {
    Thread* thread;
    bool signaled = false;
};
static std::multimap<u64, AddrWaiter*> g_addr_waiters;

Result wait_for_address(u64 addr, u32 type, s32 value, s64 timeout_ns) {
    KGuard g(g_kernel_lock);
    s32 cur = __atomic_load_n((s32*)addr, __ATOMIC_SEQ_CST);
    switch (type) {
    case 0:  // WaitIfLessThan
        if (cur >= value) return MakeResult(1, 125);
        break;
    case 1:  // DecrementAndWaitIfLessThan
        if (cur >= value) return MakeResult(1, 125);
        __atomic_store_n((s32*)addr, cur - 1, __ATOMIC_SEQ_CST);
        break;
    case 2:  // WaitIfEqual
        if (cur != value) return MakeResult(1, 125);
        break;
    default:
        return ResultInvalidEnumValue;
    }
    if (timeout_ns == 0) return ResultTimedOut;
    AddrWaiter w{current_thread()};
    auto it = g_addr_waiters.insert({addr, &w});
    while (!w.signaled) {
        if (!park_current_thread(timeout_ns) && timeout_ns >= 0 && !w.signaled) {
            g_addr_waiters.erase(it);
            return ResultTimedOut;
        }
    }
    return ResultSuccess;
}

Result signal_to_address(u64 addr, u32 type, s32 value, s32 count) {
    KGuard g(g_kernel_lock);
    auto wake_n = [&](s32 n) {
        s32 woken = 0;
        while ((n <= 0 || woken < n)) {
            auto it = g_addr_waiters.find(addr);
            if (it == g_addr_waiters.end()) break;
            it->second->signaled = true;
            wake_thread(it->second->thread);
            g_addr_waiters.erase(it);
            woken++;
        }
    };
    s32 cur = __atomic_load_n((s32*)addr, __ATOMIC_SEQ_CST);
    switch (type) {
    case 0:  // Signal
        break;
    case 1:  // SignalAndIncrementIfEqual
        if (cur != value) return MakeResult(1, 125);
        __atomic_store_n((s32*)addr, cur + 1, __ATOMIC_SEQ_CST);
        break;
    case 2: {  // SignalAndModifyByWaitingCountIfEqual
        if (cur != value) return MakeResult(1, 125);
        s32 waiting = (s32)g_addr_waiters.count(addr);
        s32 nv;
        if (count <= 0) nv = waiting > 0 ? value - 2 : value + 1;
        else if (waiting > 0) nv = waiting <= count ? value + 1 : value - 1;
        else nv = value + 1;
        __atomic_store_n((s32*)addr, nv, __ATOMIC_SEQ_CST);
        break;
    }
    default:
        return ResultInvalidEnumValue;
    }
    wake_n(count);
    return ResultSuccess;
}

// ------------------------------------------------------------------ shared memory
std::shared_ptr<SharedMemory> create_shared_memory(u64 size) {
    auto s = std::make_shared<SharedMemory>();
    s->size = (size + 0xFFF) & ~0xFFFull;
    s->backing = (u8*)VirtualAlloc(nullptr, s->size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    return s;
}
Result map_shared_memory(SharedMemory* s, u64 addr, u64 size, u32 perm) {
    KGuard g(g_kernel_lock);
    mem::map(addr, s->size, mem::Shared, perm);
    memcpy((void*)addr, s->backing, s->size);
    s->mapped_at = addr;
    return ResultSuccess;
}
void unmap_shared_memory(SharedMemory* s) {
    KGuard g(g_kernel_lock);
    if (!s->mapped_at) return;
    memcpy(s->backing, (void*)s->mapped_at, s->size);
    mem::unmap(s->mapped_at, s->size);
    s->mapped_at = 0;
}

// ------------------------------------------------------------------ WaitSynchronization
static Result wait_synchronization_impl(const Handle* handles, s32 count, s64 timeout_ns, s32* out_index) {
    KGuard g(g_kernel_lock);
    std::vector<ObjectPtr> objs;
    for (s32 i = 0; i < count; i++) {
        ObjectPtr o = g_handles.get(handles[i]);
        if (!o) return ResultInvalidHandle;
        objs.push_back(o);
    }
    Thread* self = current_thread();
    LARGE_INTEGER f, start;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&start);
    for (;;) {
        for (s32 i = 0; i < count; i++) {
            if (objs[i]->is_signaled()) {
                *out_index = i;
                return ResultSuccess;
            }
        }
        if (self->cancel_requested) {
            self->cancel_requested = false;
            return ResultCancelled;
        }
        s64 left = -1;
        if (timeout_ns >= 0) {
            LARGE_INTEGER now;
            QueryPerformanceCounter(&now);
            s64 elapsed = (s64)((now.QuadPart - start.QuadPart) * 1000000000.0 / f.QuadPart);
            left = timeout_ns - elapsed;
            if (left <= 0) return ResultTimedOut;
        }
        for (auto& o : objs) o->waiters.push_back(self);
        park_current_thread(left);
        for (auto& o : objs) {
            auto& w = o->waiters;
            w.erase(std::remove(w.begin(), w.end(), self), w.end());
        }
    }
}

// Wrapper logging waits that block a guest thread for a long time (stall diagnostics).
Result wait_synchronization(const Handle* handles, s32 count, s64 timeout_ns, s32* out_index) {
    LARGE_INTEGER f, a, b;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&a);
    Result r = wait_synchronization_impl(handles, count, timeout_ns, out_index);
    QueryPerformanceCounter(&b);
    double ms = (b.QuadPart - a.QuadPart) * 1000.0 / f.QuadPart;
    static const bool trace = getenv("HWDER_WAIT_TRACE") != nullptr;
    if (trace && ms > 150.0) {
        ObjectPtr o = g_handles.get(handles[0]);
        hw_log("kernel: wait on %d handle(s) [0x%x %s %p%s] took %.0f ms (timeout %lld ms, result 0x%x, index %d)", count,
               handles[0], o ? o->type_name() : "?", (void*)o.get(), count > 1 ? ", ..." : "", ms,
               (long long)(timeout_ns < 0 ? -1 : timeout_ns / 1000000), r, *out_index);
    }
    return r;
}

}  // namespace kern
