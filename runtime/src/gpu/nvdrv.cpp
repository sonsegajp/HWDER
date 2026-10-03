// nvdrv: the NVIDIA driver interface. Devices: nvmap, nvhost-as-gpu, nvhost-ctrl, nvhost-ctrl-gpu,
// nvhost-gpu (channels). Video devices register themselves through nvdrv_device.h.
#include <condition_variable>
#include <map>
#include <mutex>
#include <unordered_map>

#include "gpu.h"
#include "nvdrv_device.h"
#include "service/common.h"

extern "C" u64 hw_cntvct();

namespace nv {

// ------------------------------------------------------------------ device registry
static std::map<std::string, std::function<std::shared_ptr<Device>()>>& device_registry() {
    static std::map<std::string, std::function<std::shared_ptr<Device>()>> r;
    return r;
}
void register_device(const std::string& path, std::function<std::shared_ptr<Device>()> factory) {
    device_registry()[path] = std::move(factory);
}

// ------------------------------------------------------------------ syncpoints
constexpr u32 kMaxSyncpoints = 192;
struct HostAction {
    u32 id, threshold;
    std::function<void()> fn;
};
static struct {
    std::mutex m;
    std::condition_variable cv;
    std::atomic<u32> value[kMaxSyncpoints];
    std::atomic<u32> max[kMaxSyncpoints];
    bool allocated[kMaxSyncpoints];
    std::vector<HostAction> actions;
} g_sp;

static bool reached(u32 value, u32 threshold) { return (s32)(value - threshold) >= 0; }

u32 syncpoint_allocate() {
    std::lock_guard<std::mutex> l(g_sp.m);
    for (u32 i = 1; i < kMaxSyncpoints; i++)
        if (!g_sp.allocated[i]) {
            g_sp.allocated[i] = true;
            return i;
        }
    hw_log("nv: out of syncpoints");
    return 0;
}
u32 syncpoint_read(u32 id) { return id < kMaxSyncpoints ? g_sp.value[id].load() : 0; }
u32 syncpoint_reserve(u32 id, u32 count) { return id < kMaxSyncpoints ? g_sp.max[id] += count : 0; }
u32 syncpoint_increment(u32 id) {
    if (id >= kMaxSyncpoints) return 0;
    std::vector<std::function<void()>> run;
    u32 v;
    {
        std::lock_guard<std::mutex> l(g_sp.m);
        v = ++g_sp.value[id];
        if (reached(v, g_sp.max[id])) g_sp.max[id] = v;
        for (size_t i = 0; i < g_sp.actions.size();) {
            if (g_sp.actions[i].id == id && reached(v, g_sp.actions[i].threshold)) {
                run.push_back(std::move(g_sp.actions[i].fn));
                g_sp.actions[i] = std::move(g_sp.actions.back());
                g_sp.actions.pop_back();
            } else {
                i++;
            }
        }
    }
    g_sp.cv.notify_all();
    for (auto& f : run) f();
    return v;
}
bool syncpoint_wait(u32 id, u32 threshold, s64 timeout_ns) {
    if (id >= kMaxSyncpoints) return true;
    std::unique_lock<std::mutex> l(g_sp.m);
    auto ok = [&] { return reached(g_sp.value[id], threshold); };
    if (timeout_ns < 0) {
        g_sp.cv.wait(l, ok);
        return true;
    }
    return g_sp.cv.wait_for(l, std::chrono::nanoseconds(timeout_ns), ok);
}
// Run fn when syncpoint `id` reaches threshold (immediately if already reached).
static void syncpoint_on(u32 id, u32 threshold, std::function<void()> fn) {
    {
        std::lock_guard<std::mutex> l(g_sp.m);
        if (!reached(g_sp.value[id], threshold)) {
            g_sp.actions.push_back({id, threshold, std::move(fn)});
            return;
        }
    }
    fn();
}

// ------------------------------------------------------------------ nvmap
struct NvMapHandle {
    u32 size = 0, align = 0, kind = 0, heap = 0, flags = 0;
    u64 addr = 0;
    s32 refs = 1;
};
static std::mutex g_nvmap_lock;
static std::unordered_map<u32, NvMapHandle> g_nvmap;
static u32 g_nvmap_next = 1;

u64 nvmap_address(u32 handle) {
    std::lock_guard<std::mutex> l(g_nvmap_lock);
    auto it = g_nvmap.find(handle);
    return it == g_nvmap.end() ? 0 : it->second.addr;
}
u64 nvmap_size(u32 handle) {
    std::lock_guard<std::mutex> l(g_nvmap_lock);
    auto it = g_nvmap.find(handle);
    return it == g_nvmap.end() ? 0 : it->second.size;
}

template <typename T>
static T& io(IoctlBuffers& b) {
    return *(T*)b.out;
}

class NvMap : public Device {
public:
    NvResult ioctl(u32, u32 cmd, IoctlBuffers& b) override {
        std::lock_guard<std::mutex> l(g_nvmap_lock);
        u32* p = (u32*)b.out;
        switch (cmd & 0xFF) {
        case 0x01: {  // Create {size, handle}
            u32 h = g_nvmap_next++;
            g_nvmap[h].size = p[0];
            p[1] = h;
            return Success;
        }
        case 0x03:  // FromId {id, handle}
            if (!g_nvmap.count(p[0])) return BadValue;
            g_nvmap[p[0]].refs++;
            p[1] = p[0];
            return Success;
        case 0x04: {  // Alloc {handle, heap_mask, flags, align, u8 kind, pad[7], u64 addr}
            auto it = g_nvmap.find(p[0]);
            if (it == g_nvmap.end()) return BadValue;
            it->second.heap = p[1];
            it->second.flags = p[2];
            it->second.align = p[3];
            it->second.kind = ((u8*)b.out)[16];
            it->second.addr = *(u64*)(b.out + 24);
            return Success;
        }
        case 0x05: {  // Free {handle, pad, u64 address, u32 size, u32 flags}
            auto it = g_nvmap.find(p[0]);
            if (it == g_nvmap.end()) return BadValue;
            *(u64*)(b.out + 8) = it->second.addr;
            p[4] = it->second.size;
            p[5] = 0;  // memory not freed by us (the guest owns it)
            if (--it->second.refs <= 0) g_nvmap.erase(it);
            return Success;
        }
        case 0x09: {  // Param {handle, param, result}
            auto it = g_nvmap.find(p[0]);
            if (it == g_nvmap.end()) return BadValue;
            const NvMapHandle& h = it->second;
            switch (p[1]) {
            case 1: p[2] = h.size; break;
            case 2: p[2] = h.align; break;
            case 3: p[2] = (u32)h.addr; break;
            case 4: p[2] = h.heap ? h.heap : 0x40000000; break;
            case 5: p[2] = h.kind; break;
            case 6: p[2] = 0; break;
            default: return BadParameter;
            }
            return Success;
        }
        case 0x0E:  // GetId {id, handle}
            p[0] = p[1];
            return Success;
        }
        hw_log("nvmap: ioctl 0x%08x not implemented", cmd);
        return NotImplemented;
    }
};

// ------------------------------------------------------------------ GPU address space
class AddressSpace : public gpu::MemoryManager {
public:
    static constexpr u32 kPageBits = 12;
    static constexpr u64 kPageSize = 1ull << kPageBits;
    static constexpr u32 kLeafBits = 14;
    static constexpr u64 kVaBits = 40;

    u8* translate(u64 va) const override {
        u64 pte = lookup(va);
        return pte ? (u8*)(pte + (va & (kPageSize - 1))) : nullptr;
    }
    u64 contiguous_size(u64 va, u64 size) const override {
        u8* base = translate(va);
        if (!base) return 0;
        u64 done = kPageSize - (va & (kPageSize - 1));
        while (done < size) {
            if (translate(va + done) != base + done) return done;
            done += kPageSize;
        }
        return size;
    }
    void read(u64 va, void* dst, u64 size) const override {
        u8* d = (u8*)dst;
        while (size) {
            u64 n = std::min<u64>(size, kPageSize - (va & (kPageSize - 1)));
            u8* src = translate(va);
            if (src) memcpy(d, src, n);
            else memset(d, 0, n);
            va += n, d += n, size -= n;
        }
    }
    void write(u64 va, const void* src, u64 size) override {
        const u8* s = (const u8*)src;
        while (size) {
            u64 n = std::min<u64>(size, kPageSize - (va & (kPageSize - 1)));
            u8* dst = translate(va);
            if (dst) memcpy(dst, s, n);
            va += n, s += n, size -= n;
        }
    }

    void map(u64 va, u64 cpu, u64 size) {
        std::lock_guard<std::mutex> l(m_);
        for (u64 off = 0; off < size; off += kPageSize) set(va + off, cpu + off);
    }
    void unmap(u64 va, u64 size) {
        std::lock_guard<std::mutex> l(m_);
        for (u64 off = 0; off < size; off += kPageSize) set(va + off, 0);
    }
    // VA allocation (simple first-fit over a sorted free list per region).
    u64 allocate(u64 size, u64 align, bool big) {
        std::lock_guard<std::mutex> l(m_);
        auto& free = big ? big_free_ : small_free_;
        align = std::max<u64>(align, kPageSize);
        for (auto it = free.begin(); it != free.end(); ++it) {
            u64 start = (it->first + align - 1) & ~(align - 1);
            u64 end = it->first + it->second;
            if (start + size <= end) {
                u64 a = it->first, s = it->second;
                free.erase(it);
                if (start > a) free[a] = start - a;
                if (start + size < a + s) free[start + size] = a + s - (start + size);
                return start;
            }
        }
        hw_log("as-gpu: out of GPU VA (size 0x%llx)", (unsigned long long)size);
        return 0;
    }
    void reserve(u64 va, u64 size) {  // carve a fixed range out of the free lists
        std::lock_guard<std::mutex> l(m_);
        for (auto* free : {&small_free_, &big_free_}) {
            for (auto it = free->begin(); it != free->end();) {
                u64 a = it->first, e = a + it->second;
                if (e <= va || a >= va + size) {
                    ++it;
                    continue;
                }
                it = free->erase(it);
                if (a < va) (*free)[a] = va - a;
                if (e > va + size) (*free)[va + size] = e - (va + size);
            }
        }
    }
    void release(u64 va, u64 size) {
        std::lock_guard<std::mutex> l(m_);
        auto& free = va >= split_ ? big_free_ : small_free_;
        free[va] = size;
        // merge neighbours
        auto it = free.find(va);
        if (it != free.begin()) {
            auto prev = std::prev(it);
            if (prev->first + prev->second == it->first) {
                prev->second += it->second;
                free.erase(it);
                it = prev;
            }
        }
        auto next = std::next(it);
        if (next != free.end() && it->first + it->second == next->first) {
            it->second += next->second;
            free.erase(next);
        }
    }
    void init_regions(u64 start, u64 split, u64 end, u32 big_page) {
        std::lock_guard<std::mutex> l(m_);
        start_ = start, split_ = split, end_ = end, big_page_ = big_page;
        small_free_.clear();
        big_free_.clear();
        small_free_[start] = split - start;
        big_free_[split] = end - split;
    }
    u64 start_ = 0x8000000, split_ = 1ull << 34, end_ = 1ull << 40;
    u32 big_page_ = 0x20000;

    AddressSpace() { init_regions(start_, split_, end_, big_page_); }

    // mapping bookkeeping for unmap/remap
    struct Mapping {
        u64 size;
        bool sparse_reserved;
        u64 cpu;
    };
    std::map<u64, Mapping> mappings;
    std::map<u64, u64> reservations;  // AllocSpace ranges

private:
    u64 lookup(u64 va) const {
        if (va >> kVaBits) return 0;
        u64 page = va >> kPageBits;
        const u64* leaf = top_[page >> kLeafBits];
        return leaf ? leaf[page & ((1u << kLeafBits) - 1)] : 0;
    }
    void set(u64 va, u64 cpu) {
        if (va >> kVaBits) return;
        u64 page = va >> kPageBits;
        u64*& leaf = top_[page >> kLeafBits];
        if (!leaf) {
            if (!cpu) return;
            leaf = new u64[1u << kLeafBits]();
        }
        leaf[page & ((1u << kLeafBits) - 1)] = cpu;
    }
    std::mutex m_;
    u64* top_[1u << (kVaBits - kPageBits - kLeafBits)] = {};
    std::map<u64, u64> small_free_, big_free_;
};

static AddressSpace* g_as;
static AddressSpace* address_space() {
    static std::once_flag once;
    std::call_once(once, [] { g_as = new AddressSpace; });
    return g_as;
}

class AsGpu : public Device {
public:
    NvResult ioctl(u32, u32 cmd, IoctlBuffers& b) override {
        AddressSpace* as = address_space();
        u8* p = b.out;
        switch (cmd & 0xFF) {
        case 0x01:  // BindChannel {fd}
            return Success;
        case 0x02: {  // AllocSpace {pages, page_size, flags, pad, u64 offset/align}
            u32 pages = *(u32*)p, page_size = *(u32*)(p + 4), flags = *(u32*)(p + 8);
            u64 size = (u64)pages * page_size;
            u64 va;
            if (flags & 1) {  // FIXED
                va = *(u64*)(p + 16);
                as->reserve(va, size);
            } else {
                va = as->allocate(size, page_size, page_size >= as->big_page_);
                if (!va) return InsufficientMemory;
            }
            as->reservations[va] = size;
            if (getenv("HWDER_AS_TRACE"))
                hw_log("as-gpu: alloc space va %llx size %llx page %x flags %x", (unsigned long long)va, (unsigned long long)size, page_size, flags);
            *(u64*)(p + 16) = va;
            return Success;
        }
        case 0x03: {  // FreeSpace {u64 offset, pages, page_size}
            u64 va = *(u64*)p;
            u64 size = (u64)*(u32*)(p + 8) * *(u32*)(p + 12);
            as->unmap(va, size);
            as->release(va, size);
            as->reservations.erase(va);
            return Success;
        }
        case 0x05: {  // UnmapBuffer {s64 offset}
            u64 va = *(u64*)p;
            auto it = as->mappings.find(va);
            if (it != as->mappings.end()) {
                as->unmap(va, it->second.size);
                if (!it->second.sparse_reserved) as->release(va, it->second.size);
                as->mappings.erase(it);
            }
            return Success;
        }
        case 0x06: {  // MapBufferEx {flags, kind, nvmap_handle, page_size, s64 buffer_offset, u64 size, s64 offset}
            u32 flags = *(u32*)p, handle = *(u32*)(p + 8), page_size = *(u32*)(p + 12);
            u64 buf_off = *(u64*)(p + 16), size = *(u64*)(p + 24), va = *(u64*)(p + 32);
            if (flags & 0x100) {  // REMAP: a subregion of an existing mapping (same backing, new kind)
                auto it = as->mappings.find(va);
                if (it == as->mappings.end()) return BadValue;
                as->map(va + buf_off, it->second.cpu + buf_off, size);
                return Success;
            }
            if (flags & 0x100) {  // REMAP: a subregion of an existing mapping (same backing, new kind)
                auto it = as->mappings.find(va);
                if (it == as->mappings.end()) return BadValue;
                as->map(va + buf_off, it->second.cpu + buf_off, size);
                return Success;
            }
            u64 base = nvmap_address(handle);
            if (!base) {
                hw_log("as-gpu: MapBufferEx with unknown nvmap handle %u (flags 0x%x)", handle, flags);
                return BadValue;
            }
            if (!size) size = nvmap_size(handle) - buf_off;
            size = (size + 0xFFF) & ~0xFFFull;
            bool in_reservation = false;
            if (flags & 1) {  // FIXED
                for (auto& [rva, rsize] : as->reservations)
                    if (va >= rva && va + size <= rva + rsize) in_reservation = true;
                if (!in_reservation) as->reserve(va, size);
            } else {
                if (!page_size) page_size = 0x1000;
                bool big = page_size >= as->big_page_ || size >= as->big_page_;
                va = as->allocate(size, big ? as->big_page_ : page_size, big);
                if (!va) return InsufficientMemory;
            }
            as->map(va, base + buf_off, size);
            as->mappings[va] = {size, in_reservation, base + buf_off};
            if (getenv("HWDER_AS_TRACE"))
                hw_log("as-gpu: map va %llx size %llx handle %u off %llx flags %x page %x -> %llx", (unsigned long long)va,
                       (unsigned long long)size, handle, (unsigned long long)buf_off, flags, page_size, (unsigned long long)(base + buf_off));
            *(u64*)(p + 32) = va;
            return Success;
        }
        case 0x08: {  // GetVaRegions {u64 buf_addr, u32 buf_size, pad, VaRegion[2] {u64 offset, u32 page_size, pad, u64 pages}}
            *(u32*)(p + 8) = 2 * 0x18;
            u8* r = p + 16;
            *(u64*)r = as->start_;
            *(u32*)(r + 8) = 0x1000;
            *(u64*)(r + 16) = (as->split_ - as->start_) >> 12;
            r += 0x18;
            *(u64*)r = as->split_;
            *(u32*)(r + 8) = as->big_page_;
            *(u64*)(r + 16) = (as->end_ - as->split_) / as->big_page_;
            if (b.out2 && b.out2_size >= 0x30) memcpy(b.out2, p + 16, 0x30);
            return Success;
        }
        case 0x09: {  // InitializeEx {big_page_size, as_fd, flags, reserved, u64 start, u64 end, u64 split}
            u32 big = *(u32*)p;
            if (big) as->big_page_ = big;
            if (getenv("HWDER_AS_TRACE")) hw_log("as-gpu: InitializeEx big page %x", big);
            return Success;
        }
        case 0x14: {  // Remap: entries {u16 flags, u16 kind, u32 handle, u32 mem_offset_lo, u32 gpu_offset, u32 pages}
            u32 n = (u32)(((cmd >> 16) & 0x3FFF) / 20);
            for (u32 i = 0; i < n; i++) {
                const u8* e = p + i * 20;
                u32 handle = *(u32*)(e + 4);
                u64 mem_off = (u64)*(u32*)(e + 8) << 16;
                u64 va = (u64)*(u32*)(e + 12) << 16;
                u64 size = (u64)*(u32*)(e + 16) << 16;
                if (getenv("HWDER_AS_TRACE"))
                    hw_log("as-gpu: remap va %llx size %llx handle %u memoff %llx flags %x", (unsigned long long)va,
                           (unsigned long long)size, handle, (unsigned long long)mem_off, *(u16*)e);
                if (!handle) as->unmap(va, size);
                else as->map(va, nvmap_address(handle) + mem_off, size);
            }
            return Success;
        }
        }
        hw_log("as-gpu: ioctl 0x%08x not implemented", cmd);
        return NotImplemented;
    }
};

// ------------------------------------------------------------------ nvhost-ctrl
constexpr u32 kMaxEvents = 64;
struct CtrlEvent {
    bool registered = false;
    std::shared_ptr<kern::ReadableEvent> event;
    u32 syncpt = 0, threshold = 0;
    std::atomic<u32> generation{0};
};
static CtrlEvent g_events[kMaxEvents];
static std::mutex g_events_lock;

class Ctrl : public Device {
public:
    NvResult ioctl(u32, u32 cmd, IoctlBuffers& b) override {
        u32* p = (u32*)b.out;
        switch (cmd & 0xFF) {
        case 0x14:  // SyncptRead {id, value}
            p[1] = syncpoint_read(p[0]);
            return Success;
        case 0x15:  // SyncptIncr {id}
            syncpoint_increment(p[0]);
            return Success;
        case 0x16:  // SyncptWait {id, thresh, timeout}
        case 0x19: {  // SyncptWaitEx {id, thresh, timeout, value}
            s32 timeout = (s32)p[2];
            static const bool trace = getenv("HWDER_SYNC_TRACE") != nullptr;
            if (trace && !reached(syncpoint_read(p[0]), p[1]))
                hw_log("sync: syncpt wait %u >= %u (value %u) timeout %d", p[0], p[1], syncpoint_read(p[0]), timeout);
            bool ok = syncpoint_wait(p[0], p[1], timeout < 0 ? -1 : (s64)timeout * 1000000);
            if (trace && !ok) hw_log("sync: syncpt wait %u >= %u timed out", p[0], p[1]);
            if ((cmd & 0xFF) == 0x19) p[3] = syncpoint_read(p[0]);
            return ok ? Success : Timeout;
        }
        case 0x1A:  // SyncptReadMax {id, value}
            p[1] = g_sp.max[p[0] % kMaxSyncpoints];
            return Success;
        case 0x1B:  // GetConfig: not configured -> caller uses defaults
            return NotImplemented;
        case 0x1C: {  // EventSignal / ClearEventWait {event_id}
            u32 slot = (p[0] & 0x10000000) ? (p[0] & 0xF) : (p[0] & 0xFFFF);
            if (slot < kMaxEvents) {
                std::lock_guard<std::mutex> l(g_events_lock);
                CtrlEvent& ev = g_events[slot];
                // Clearing a wait whose threshold was never reached means the guest's wait timed out.
                if (ev.threshold && !reached(syncpoint_read(ev.syncpt), ev.threshold))
                    hw_log("nvhost-ctrl: wait for syncpt %u >= %u abandoned (value %u, max %u, host thread %lu)", ev.syncpt,
                           ev.threshold, syncpoint_read(ev.syncpt), g_sp.max[ev.syncpt % kMaxSyncpoints].load(),
                           GetCurrentThreadId());
                ev.threshold = 0;
                ev.generation++;
                if (ev.event) ev.event->clear();
            }
            return Success;
        }
        case 0x1D:   // EventWait {fence id, fence value, timeout, value}
        case 0x1E:   // EventWaitAsync
            return event_wait(p, (cmd & 0xFF) == 0x1D);
        case 0x1F: {  // EventRegister {user_event_id}
            u32 slot = p[0];
            if (slot >= kMaxEvents) return BadParameter;
            std::lock_guard<std::mutex> l(g_events_lock);
            g_events[slot].registered = true;
            if (!g_events[slot].event) g_events[slot].event = ipc::make_event();
            return Success;
        }
        case 0x20: {  // EventUnregister {user_event_id}
            u32 slot = p[0];
            if (slot < kMaxEvents) {
                std::lock_guard<std::mutex> l(g_events_lock);
                g_events[slot].registered = false;
            }
            return Success;
        }
        case 0x21:  // EventUnregisterBatch / EventKill {u64 mask}
            return Success;
        }
        hw_log("nvhost-ctrl: ioctl 0x%08x not implemented", cmd);
        return NotImplemented;
    }

    NvResult event_wait(u32* p, bool allocate) {
        u32 id = p[0], thresh = p[1];
        s32 timeout = (s32)p[2];
        static const bool trace = getenv("HWDER_SYNC_TRACE") != nullptr;
        if (trace && !(thresh == 0 || reached(syncpoint_read(id), thresh)))
            hw_log("sync: event wait syncpt %u >= %u (value %u, max %u) timeout %d", id, thresh, syncpoint_read(id),
                   g_sp.max[id % kMaxSyncpoints].load(), timeout);
        if (id >= kMaxSyncpoints) return BadParameter;
        if (thresh == 0 || reached(syncpoint_read(id), thresh)) {
            p[3] = syncpoint_read(id);
            return Success;
        }
        std::lock_guard<std::mutex> l(g_events_lock);
        u32 slot;
        if (allocate) {
            slot = kMaxEvents;
            for (u32 i = 0; i < kMaxEvents; i++)
                if (g_events[i].registered && g_events[i].syncpt == id) { slot = i; break; }
            if (slot == kMaxEvents)
                for (u32 i = 0; i < kMaxEvents; i++)
                    if (!g_events[i].registered) {
                        slot = i;
                        g_events[i].registered = true;
                        if (!g_events[i].event) g_events[i].event = ipc::make_event();
                        break;
                    }
        } else {
            slot = p[3] & 0xFFFF;
        }
        if (slot >= kMaxEvents) return BadParameter;
        if (timeout == 0) return Timeout;
        CtrlEvent& ev = g_events[slot];
        if (!ev.event) ev.event = ipc::make_event();
        ev.registered = true;
        ev.syncpt = id;
        ev.threshold = thresh;
        ev.event->clear();
        u32 gen = ++ev.generation;
        auto event = ev.event;
        syncpoint_on(id, thresh, [slot, gen, event] {
            if (g_events[slot].generation == gen) event->signal();
        });
        // Value encodes the slot and syncpoint so QueryEvent can find the event.
        p[3] = allocate ? (slot | ((id & 0xFFF) << 16) | 0x10000000) : (slot | (id << 4));
        return Timeout;
    }

    std::shared_ptr<kern::ReadableEvent> query_event(u32, u32 event_id) override {
        u32 slot = (event_id & 0x10000000) ? (event_id & 0xFFFF) : (event_id & 0xF);
        if (slot >= kMaxEvents) return nullptr;
        std::lock_guard<std::mutex> l(g_events_lock);
        if (!g_events[slot].event) g_events[slot].event = ipc::make_event();
        return g_events[slot].event;
    }
};

// ------------------------------------------------------------------ nvhost-ctrl-gpu
class CtrlGpu : public Device {
public:
    NvResult ioctl(u32, u32 cmd, IoctlBuffers& b) override {
        u8* p = b.out;
        switch (cmd & 0xFF) {
        case 0x01:  // ZcullGetCtxSize
            *(u32*)p = 1;
            return Success;
        case 0x02: {  // ZcullGetInfo
            u32 v[10] = {0x20, 0x20, 0x400, 0x800, 0x20, 0x20, 0xC0, 0x20, 0x40, 0x10};
            memcpy(p, v, sizeof(v));
            return Success;
        }
        case 0x03: case 0x04:  // ZbcSetTable / ZbcQueryTable
            return Success;
        case 0x05: {  // GetCharacteristics {u64 buf_size, u64 buf_addr, GpuCharacteristics (0xA0)}
            u8 gc[0xA0] = {};
            auto w32 = [&](u32 off, u32 v) { memcpy(gc + off, &v, 4); };
            auto w64 = [&](u32 off, u64 v) { memcpy(gc + off, &v, 8); };
            w32(0x00, 0x120); w32(0x04, 0xB); w32(0x08, 0xA1); w32(0x0C, 1);
            w64(0x10, 0x40000); w64(0x18, 0);
            w32(0x20, 2); w32(0x24, 0x20); w32(0x28, 0x20000); w32(0x2C, 0x20000);
            w32(0x30, 0x1B); w32(0x34, 0x30000); w32(0x38, 1); w32(0x3C, 0x503);
            w32(0x40, 0x503); w32(0x44, 0x80); w32(0x48, 0x28); w32(0x4C, 0);
            w64(0x50, 0x55);
            w32(0x58, 0x902D); w32(0x5C, 0xB197); w32(0x60, 0xB1C0); w32(0x64, 0xB06F);
            w32(0x68, 0xA140); w32(0x6C, 0xB0B5); w32(0x70, 1); w32(0x74, 0);
            w32(0x78, 2); w32(0x7C, 1); w32(0x80, 0); w32(0x84, 1);
            w32(0x88, 0x21D70); w32(0x8C, 0);
            w64(0x90, 0x6230326D67ull); w64(0x98, 0);
            *(u64*)p = 0xA0;
            if (!*(u64*)(p + 8)) *(u64*)(p + 8) = 0xdeadbeef;
            memcpy(p + 16, gc, sizeof(gc));
            if (b.out2 && b.out2_size) memcpy(b.out2, gc, std::min<u64>(sizeof(gc), b.out2_size));
            return Success;
        }
        case 0x06: {  // GetTPCMasks {u32 mask_buf_size, pad, u64 mask_buf_addr, u64 tpc_mask}
            *(u64*)(p + 16) = 3;
            if (b.out2 && b.out2_size >= 4) *(u32*)b.out2 = 3;
            return Success;
        }
        case 0x07:  // FlushL2
            return Success;
        case 0x13:  // NumVsms
            *(u32*)p = 2;
            return Success;
        case 0x14:  // GetActiveSlotMask
            *(u32*)p = 7;
            *(u32*)(p + 4) = 1;
            return Success;
        case 0x15:  // PmuGetGpuLoad
            *(u32*)p = 0;
            return Success;
        case 0x1C:  // GetGpuTime (ns)
            *(u64*)p = hw_cntvct() * 625 / 12;
            return Success;
        }
        hw_log("nvhost-ctrl-gpu: ioctl 0x%08x not implemented", cmd);
        return NotImplemented;
    }
};

// ------------------------------------------------------------------ nvhost-gpu (channel)
class GpuChannel : public Device {
public:
    GpuChannel() { syncpt_ = syncpoint_allocate(); }

    NvResult ioctl(u32, u32 cmd, IoctlBuffers& b) override {
        u8* p = b.out;
        u32 group = (cmd >> 8) & 0xFF, nr = cmd & 0xFF;
        if (group == 0x00 && nr == 0x03) {  // GetWaitbase
            *(u32*)(p + 4) = 0;
            return Success;
        }
        if (group == 'G') {
            if (nr == 0x14) { client_data_ = *(u64*)p; return Success; }  // SetClientData
            if (nr == 0x15) { *(u64*)p = client_data_; return Success; }  // GetClientData
        }
        if (group != 'H') return unknown(cmd);
        switch (nr) {
        case 0x01:  // SetNvmapFd
        case 0x03:  // SetTimeout
        case 0x0B:  // ZCullBind
        case 0x0C:  // SetErrorNotifier
        case 0x0D:  // SetPriority
        case 0x1D:  // SetTimeslice
            return Success;
        case 0x09: {  // AllocObjCtx {class, flags, u64 obj_id}
            *(u64*)(p + 8) = *(u32*)p;
            return Success;
        }
        case 0x18: case 0x1A: {  // AllocGPFIFOEx(2) {num_entries, num_jobs, flags, fence out, reserved[3]}
            if (!channel_) channel_ = gpu::create_channel(address_space());
            *(u32*)(p + 12) = syncpt_;
            *(u32*)(p + 16) = syncpoint_read(syncpt_);
            return Success;
        }
        case 0x08: case 0x1B: {  // SubmitGPFIFO {u64 address, u32 num_entries, u32 flags, fence}
            u32 n = *(u32*)(p + 8), flags = *(u32*)(p + 12);
            const u64* entries;
            if (nr == 0x08) entries = (const u64*)(b.in + 0x18);         // inline after the header
            else if (b.in2) entries = (const u64*)b.in2;                 // Ioctl2: separate input buffer
            else entries = (const u64*)(uintptr_t)*(u64*)p;              // KickoffPB: guest pointer
            return submit(entries, n, flags, (u32*)(p + 16));
        }
        }
        return unknown(cmd);
    }

    std::shared_ptr<kern::ReadableEvent> query_event(u32, u32) override { return ipc::make_event(); }

private:
    NvResult submit(const u64* entries, u32 n, u32 flags, u32* fence) {
        if (!channel_) channel_ = gpu::create_channel(address_space());
        gpu::Submission s;
        s.entries.resize(n);
        memcpy(s.entries.data(), entries, (size_t)n * 8);
        s.syncpoint = syncpt_;
        if (flags & 1) {  // FENCE_WAIT
            s.wait = true;
            s.wait_fence = {fence[0], fence[1]};
        }
        if (flags & 2) {  // FENCE_INCREMENT: the work increments our syncpoint (WFI + increment = 2)
            s.increment = true;
            s.increment_count = 2;
            // INCREMENT_VALUE (0x100): fence.value is how many extra increments the pushbuffer itself
            // performs; they count towards the returned fence.
            u32 extra = (flags & 0x100) ? fence[1] : 0;
            u32 max = syncpoint_reserve(syncpt_, 2 + extra);
            fence[0] = syncpt_;
            fence[1] = max;
        } else {
            fence[0] = syncpt_;
            fence[1] = g_sp.max[syncpt_];
        }
        static const bool trace = getenv("HWDER_SYNC_TRACE") != nullptr;
        if (trace)
            hw_log("sync: submit ch(syncpt %u) flags 0x%x entries %u wait %u:%u -> fence %u:%u (value %u)", syncpt_, flags, n,
                   s.wait ? s.wait_fence.id : 0, s.wait ? s.wait_fence.value : 0, fence[0], fence[1], syncpoint_read(syncpt_));
        gpu::channel_submit(channel_, std::move(s));
        return Success;
    }
    NvResult unknown(u32 cmd) {
        hw_log("nvhost-gpu: ioctl 0x%08x not implemented", cmd);
        return NotImplemented;
    }
    u32 syncpt_;
    u64 client_data_ = 0;
    gpu::Channel* channel_ = nullptr;
};

class NullDevice : public Device {
public:
    explicit NullDevice(std::string n) : name_(std::move(n)) {}
    NvResult ioctl(u32, u32 cmd, IoctlBuffers&) override {
        hw_log("nv: %s ioctl 0x%08x (stub)", name_.c_str(), cmd);
        return Success;
    }
    std::string name_;
};

}  // namespace nv

namespace gpu {
MemoryManager* create_address_space() { return nv::address_space(); }
}  // namespace gpu

// ------------------------------------------------------------------ nvdrv service
namespace ipc {
namespace {

using namespace nv;

class NvDrv : public SimpleService {
public:
    explicit NvDrv(const char* name) : SimpleService(name) {
        reg(0, [this](Request& rq, Response& rs) {  // Open(path) -> fd, error
            std::string path = rq.in_string();
            auto& reg = device_registry();
            auto it = reg.find(path);
            std::shared_ptr<Device> dev = it != reg.end() ? it->second() : std::make_shared<NullDevice>(path);
            if (it == reg.end()) hw_log("nvdrv: unknown device %s (stub)", path.c_str());
            std::lock_guard<std::mutex> l(m_);
            u32 fd = next_fd_++;
            fds_[fd] = dev;
            dev->on_open(fd);
            rs.push<u32>(fd);
            rs.push<u32>(Success);
        });
        reg(1, [this](Request& rq, Response& rs) { rs.push<u32>(do_ioctl(rq, 1)); });  // Ioctl
        reg(2, [this](Request& rq, Response& rs) {                                     // Close
            u32 fd = rq.pop<u32>();
            std::lock_guard<std::mutex> l(m_);
            auto it = fds_.find(fd);
            if (it != fds_.end()) {
                it->second->on_close(fd);
                fds_.erase(it);
            }
            rs.push<u32>(Success);
        });
        reg(3, [](Request&, Response& rs) { rs.push<u32>(Success); });  // Initialize
        reg(4, [this](Request& rq, Response& rs) {                      // QueryEvent(fd, id)
            u32 fd = rq.pop<u32>(), id = rq.pop<u32>();
            auto dev = get(fd);
            auto ev = dev ? dev->query_event(fd, id) : nullptr;
            if (!ev) {
                hw_log("nvdrv: QueryEvent(fd %u, 0x%x) - no event", fd, id);
                ev = make_event();
            }
            push_handle(rs, ev);
            rs.push<u32>(Success);
        });
        reg(5, [](Request&, Response& rs) { rs.push<u32>(Success); });  // MapSharedMemory
        reg(6, [](Request&, Response& rs) {                             // GetStatus
            for (int i = 0; i < 8; i++) rs.push<u32>(0);
            rs.push<u32>(Success);
        });
        reg(7, [](Request&, Response& rs) { rs.push<u32>(Success); });
        reg(8, [](Request&, Response& rs) { rs.push<u32>(Success); });  // SetAruid
        reg(9, [](Request&, Response&) {});                             // DumpGraphicsMemoryInfo
        reg(10, [](Request&, Response& rs) { rs.push<u32>(Success); });
        reg(11, [this](Request& rq, Response& rs) { rs.push<u32>(do_ioctl(rq, 2)); });  // Ioctl2
        reg(12, [this](Request& rq, Response& rs) { rs.push<u32>(do_ioctl(rq, 3)); });  // Ioctl3
        reg(13, [](Request&, Response&) {});  // SetGraphicsFirmwareMemoryMarginEnabled
    }

private:
    std::shared_ptr<Device> get(u32 fd) {
        std::lock_guard<std::mutex> l(m_);
        auto it = fds_.find(fd);
        return it == fds_.end() ? nullptr : it->second;
    }

    u32 do_ioctl(Request& rq, int kind) {
        u32 fd = rq.pop<u32>(), cmd = rq.pop<u32>();
        auto dev = get(fd);
        if (!dev) return BadParameter;
        Buffer in = rq.in_buffer(0), out = rq.out_buffer(0);
        Buffer in2 = kind == 2 ? rq.in_buffer(1) : Buffer{};
        Buffer out2 = kind == 3 ? rq.out_buffer(1) : Buffer{};
        u64 isz = (cmd >> 16) & 0x3FFF;
        // Work buffer: input copied in, device writes results in place, copied to the output.
        std::vector<u8> work(std::max<u64>({isz, in.size, out.size, 8}) + 64, 0);
        if (in.addr) memcpy(work.data(), (void*)in.addr, in.size);
        std::vector<u8> out2_buf(out2.size);
        IoctlBuffers b;
        b.in = in.addr ? (const u8*)in.addr : work.data();
        b.in_size = in.size;
        b.out = work.data();
        b.out_size = out.size;
        b.in2 = in2.addr ? (const u8*)in2.addr : nullptr;
        b.in2_size = in2.size;
        b.out2 = out2.size ? out2_buf.data() : nullptr;
        b.out2_size = out2.size;
        NvResult r = dev->ioctl(fd, cmd, b);
        if (out.addr && (cmd >> 31 & 1)) memcpy((void*)out.addr, work.data(), out.size);
        else if (out.addr) memcpy((void*)out.addr, work.data(), out.size);
        if (out2.addr) memcpy((void*)out2.addr, out2_buf.data(), out2.size);
        static const bool trace = getenv("HWDER_NV_TRACE") != nullptr;
        if (trace || (r != Success && r != Timeout)) hw_log("nvdrv: fd %u ioctl 0x%08x -> %u", fd, cmd, r);
        return r;
    }

    std::mutex m_;
    std::map<u32, std::shared_ptr<Device>> fds_;
    u32 next_fd_ = 1;
};

}  // namespace

void register_nvdrv_services() {
    register_device("/dev/nvmap", [] { return std::make_shared<NvMap>(); });
    register_device("/dev/nvhost-as-gpu", [] { return std::make_shared<AsGpu>(); });
    register_device("/dev/nvhost-ctrl", [] { return std::make_shared<Ctrl>(); });
    register_device("/dev/nvhost-ctrl-gpu", [] { return std::make_shared<CtrlGpu>(); });
    register_device("/dev/nvhost-gpu", [] { return std::make_shared<GpuChannel>(); });
    for (const char* n : {"nvdrv", "nvdrv:a", "nvdrv:s", "nvdrv:t"})
        register_service(n, [n] { return std::make_shared<NvDrv>(n); });
}

}  // namespace ipc
