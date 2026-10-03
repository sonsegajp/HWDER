// HWDER kernel: HOS (Horizon) kernel objects implemented natively.
//
// Every guest thread is a host thread running recompiled code; an `svc` instruction is a direct
// call into svc_dispatch(). Kernel state is guarded by one recursive lock (g_kernel_lock); blocking
// waits release it and park on a per-thread Win32 event.
#pragma once
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "hwder/runtime.h"

namespace kern {

using Handle = u32;
using Result = u32;

constexpr Result ResultSuccess = 0;
// Kernel result codes (module 1)
constexpr Result MakeResult(u32 module, u32 desc) { return module | (desc << 9); }
constexpr Result ResultInvalidHandle = MakeResult(1, 114);
constexpr Result ResultTimedOut = MakeResult(1, 117);
constexpr Result ResultCancelled = MakeResult(1, 118);
constexpr Result ResultInvalidAddress = MakeResult(1, 102);
constexpr Result ResultInvalidSize = MakeResult(1, 101);
constexpr Result ResultInvalidState = MakeResult(1, 125);
constexpr Result ResultInvalidCurrentMemory = MakeResult(1, 106);
constexpr Result ResultOutOfMemory = MakeResult(1, 104);
constexpr Result ResultNotFound = MakeResult(1, 120);
constexpr Result ResultInvalidEnumValue = MakeResult(1, 120);
constexpr Result ResultInvalidCombination = MakeResult(1, 116);
constexpr Result ResultSessionClosed = MakeResult(1, 123);
constexpr Result ResultInvalidPriority = MakeResult(1, 112);
constexpr Result ResultInvalidCoreId = MakeResult(1, 113);

constexpr Handle kCurrentProcess = 0xFFFF8001;
constexpr Handle kCurrentThread = 0xFFFF8000;

// Recursive kernel lock that a blocking wait can fully release and later restore.
class KernelLock {
public:
    void lock();
    void unlock();
    int release_all();          // returns the recursion depth that was held
    void reacquire(int depth);

private:
    std::mutex m_;
    std::atomic<u32> owner_{0};
    int depth_ = 0;
};
extern KernelLock g_kernel_lock;
using KGuard = std::lock_guard<KernelLock>;

// ------------------------------------------------------------------ objects
struct Object {
    virtual ~Object() = default;
    virtual const char* type_name() const = 0;
    // Synchronization (WaitSynchronization)
    virtual bool is_signaled() const { return false; }
    std::vector<struct Thread*> waiters;  // threads blocked in WaitSynchronization on this object
    void notify_waiters();
};
using ObjectPtr = std::shared_ptr<Object>;

struct Thread : Object {
    const char* type_name() const override { return "Thread"; }
    bool is_signaled() const override { return exited; }
    Ctx* ctx = nullptr;
    u64 entry = 0, arg = 0, stack_top = 0;
    u64 tls = 0;  // guest address of this thread's 0x200-byte TLS block (TPIDRRO_EL0)
    s32 priority = 44, ideal_core = 0;
    u64 core_mask = 1;
    u64 id = 0;
    void* host_handle = nullptr;
    void* wake_event = nullptr;  // Win32 auto-reset event used to park the thread
    std::atomic<bool> started{false}, exited{false};
    // wait state
    Result wait_result = 0;
    s32 wait_index = -1;
    bool waiting = false;
    bool cancel_requested = false;
    Handle handle = 0;  // the thread's own handle (value guest mutexes store as owner)
    std::string name;
};

struct ReadableEvent : Object {
    const char* type_name() const override { return "ReadableEvent"; }
    bool is_signaled() const override { return signaled; }
    bool signaled = false;
    void signal();
    void clear() { signaled = false; }
};
struct WritableEvent : Object {
    const char* type_name() const override { return "WritableEvent"; }
    std::shared_ptr<ReadableEvent> readable;
};

// Shared memory written by services (HID input, applet storage...). With identity mapping a page
// cannot appear at two addresses, so the guest mapping *becomes* the storage: services always access
// it through ptr(), which follows the mapping once the guest maps it.
struct SharedMemory : Object {
    const char* type_name() const override { return "SharedMemory"; }
    u8* backing = nullptr;
    u64 size = 0;
    u64 mapped_at = 0;
    u8* ptr() const { return mapped_at ? (u8*)mapped_at : backing; }
};
std::shared_ptr<SharedMemory> create_shared_memory(u64 size);
Result map_shared_memory(SharedMemory* s, u64 addr, u64 size, u32 perm);
void unmap_shared_memory(SharedMemory* s);

struct TransferMemory : Object {
    const char* type_name() const override { return "TransferMemory"; }
    u64 addr = 0, size = 0;
    u32 perm = 0;
};

struct Process : Object {
    const char* type_name() const override { return "Process"; }
};


// ------------------------------------------------------------------ handle table
class HandleTable {
public:
    Handle add(ObjectPtr obj);
    ObjectPtr get(Handle h) const;
    template <typename T>
    std::shared_ptr<T> get_as(Handle h) const { return std::dynamic_pointer_cast<T>(get(h)); }
    bool close(Handle h);

private:
    std::map<Handle, ObjectPtr> table_;
    u32 next_ = 1;
};
extern HandleTable g_handles;

// ------------------------------------------------------------------ threads
Thread* current_thread();
std::shared_ptr<Thread> create_thread(u64 entry, u64 arg, u64 stack_top, s32 priority, s32 core);
void start_thread(Thread* t);
[[noreturn]] void exit_current_thread();
// Park the current thread (kernel lock held on entry and exit). Returns false on timeout.
bool park_current_thread(s64 timeout_ns);
void wake_thread(Thread* t);
// Every thread ever created (kept alive for diagnostics).
std::vector<std::shared_ptr<Thread>> all_threads();

// ------------------------------------------------------------------ memory
namespace mem {
enum State : u32 {
    Free = 0, Io = 1, Static = 2, Code = 3, CodeData = 4, Normal = 5, Shared = 6, Alias = 7, AliasCode = 8,
    AliasCodeData = 9, Ipc = 0xA, Stack = 0xB, ThreadLocal = 0xC, Transfered = 0xD, SharedTransfered = 0xE,
    SharedCode = 0xF, Inaccessible = 0x10,
};
enum Perm : u32 { None = 0, R = 1, W = 2, X = 4, RW = 3, RX = 5 };

// Address space layout (39-bit). Guest addresses are identity mapped to host addresses.
constexpr u64 kAddressSpaceBase = 0x8000000;
constexpr u64 kAddressSpaceEnd = 1ull << 39;
constexpr u64 kHeapBase = 0x2000000000ull, kHeapMax = 0x180000000ull;      // 6 GiB
constexpr u64 kAliasBase = 0x3000000000ull, kAliasSize = 0x1000000000ull;   // 64 GiB
constexpr u64 kStackBase = 0x4000000000ull, kStackSize = 0x80000000ull;     // 2 GiB
constexpr u64 kTlsBase = 0x4800000000ull, kTlsSize = 0x1000000ull;
constexpr u64 kSystemBase = 0x5000000000ull;  // kernel-owned objects (shared memory etc.)

void init();
void map(u64 addr, u64 size, State state, u32 perm, bool commit = true);
void unmap(u64 addr, u64 size);
void set_perm(u64 addr, u64 size, u32 perm);
void set_attr(u64 addr, u64 size, u32 mask, u32 attr);
struct Region { u64 addr, size; State state; u32 perm, attr; };
Region query(u64 addr);
u64 heap_size();
Result set_heap_size(u64 size, u64* out_addr);
u64 alloc_tls_slot();
u64 alloc_system(u64 size);  // kernel-owned guest-visible memory
}  // namespace mem

// ------------------------------------------------------------------ process info
struct ProcessInfo {
    u64 title_id = 0x0100AE00096EA000ull;
    u32 main_thread_priority = 44;
    u32 main_thread_core = 0;
    u32 main_thread_stack = 0x100000;
    u64 total_memory = 0xCD500000ull;  // ~3.2 GiB application pool
};
extern ProcessInfo g_process;

void svc_dispatch(Ctx* c, u32 imm);
void boot(const std::string& exefs_dir);

}  // namespace kern
