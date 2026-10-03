// Guest address space. Guest virtual addresses are identity mapped to host addresses: each
// kernel region (heap, alias, stack, TLS, system) is a fixed host reservation, committed on map.
// A region map tracks HOS memory state/permissions for svcQueryMemory.
#include <windows.h>

#include <functional>

#include "kernel.h"

namespace kern::mem {

static std::map<u64, Region> g_regions;  // start -> region (only non-Free regions are stored)
static u64 g_heap_size = 0;
static u64 g_tls_next = kTlsBase;
static u64 g_system_next = kSystemBase;

static void reserve(u64 addr, u64 size, const char* what) {
    void* p = VirtualAlloc((void*)addr, size, MEM_RESERVE, PAGE_NOACCESS);
    if (p != (void*)addr) {
        hw_log("mem: cannot reserve %s at 0x%llx (+0x%llx): error %lu", what, (unsigned long long)addr,
               (unsigned long long)size, GetLastError());
        hw_fatal("address space reservation");
    }
}

void init() {
    reserve(kHeapBase, kHeapMax, "heap region");
    reserve(kAliasBase, kAliasSize, "alias region");
    reserve(kStackBase, kStackSize, "stack region");
    reserve(kTlsBase, kTlsSize, "tls region");
    reserve(kSystemBase, 0x100000000ull, "system region");
}

static void commit(u64 addr, u64 size) {
    if (!VirtualAlloc((void*)addr, size, MEM_COMMIT, PAGE_READWRITE)) {
        // Outside our reservations (e.g. module images): reserve + commit in one go.
        if (!VirtualAlloc((void*)addr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)) {
            hw_log("mem: commit failed at 0x%llx (+0x%llx): error %lu", (unsigned long long)addr,
                   (unsigned long long)size, GetLastError());
            hw_fatal("commit");
        }
    }
}

// Remove [addr, addr+size) from the region map, splitting regions at the edges.
static void carve(u64 addr, u64 size) {
    u64 end = addr + size;
    auto it = g_regions.lower_bound(addr);
    if (it != g_regions.begin()) {
        auto prev = std::prev(it);
        if (prev->second.addr + prev->second.size > addr) it = prev;
    }
    while (it != g_regions.end() && it->second.addr < end) {
        Region r = it->second;
        it = g_regions.erase(it);
        if (r.addr < addr) g_regions[r.addr] = {r.addr, addr - r.addr, r.state, r.perm, r.attr};
        if (r.addr + r.size > end) g_regions[end] = {end, r.addr + r.size - end, r.state, r.perm, r.attr};
    }
}

void map(u64 addr, u64 size, State state, u32 perm, bool do_commit) {
    KGuard g(g_kernel_lock);
    if (do_commit) commit(addr, size);
    carve(addr, size);
    g_regions[addr] = {addr, size, state, perm, 0};
}

void unmap(u64 addr, u64 size) {
    KGuard g(g_kernel_lock);
    carve(addr, size);
    VirtualFree((void*)addr, size, MEM_DECOMMIT);
}

static void modify(u64 addr, u64 size, const std::function<void(Region&)>& fn) {
    std::vector<Region> parts;
    u64 end = addr + size;
    for (u64 a = addr; a < end;) {
        Region r = query(a);
        u64 rend = std::min(end, r.addr + r.size);
        parts.push_back({a, rend - a, r.state, r.perm, r.attr});
        a = rend;
    }
    carve(addr, size);
    for (Region r : parts) {
        fn(r);
        if (r.state != Free) g_regions[r.addr] = r;
    }
}

void set_perm(u64 addr, u64 size, u32 perm) {
    KGuard g(g_kernel_lock);
    modify(addr, size, [&](Region& r) { r.perm = perm; });
}

void set_attr(u64 addr, u64 size, u32 mask, u32 attr) {
    KGuard g(g_kernel_lock);
    modify(addr, size, [&](Region& r) { r.attr = (r.attr & ~mask) | (attr & mask); });
}

Region query(u64 addr) {
    KGuard g(g_kernel_lock);
    if (addr >= kAddressSpaceEnd) {
        return {kAddressSpaceEnd, ~0ull - kAddressSpaceEnd + 1, Inaccessible, None, 0};
    }
    auto it = g_regions.upper_bound(addr);
    u64 free_start = 0, free_end = kAddressSpaceEnd;
    if (it != g_regions.end()) free_end = it->second.addr;
    if (it != g_regions.begin()) {
        auto prev = std::prev(it);
        if (prev->second.addr + prev->second.size > addr) return prev->second;
        free_start = prev->second.addr + prev->second.size;
    }
    return {free_start, free_end - free_start, Free, None, 0};
}

u64 heap_size() { return g_heap_size; }

Result set_heap_size(u64 size, u64* out_addr) {
    KGuard g(g_kernel_lock);
    if (size > kHeapMax || (size & 0x1FFFFF)) return ResultInvalidSize;
    if (size > g_heap_size) {
        map(kHeapBase + g_heap_size, size - g_heap_size, Normal, RW);
    } else if (size < g_heap_size) {
        unmap(kHeapBase + size, g_heap_size - size);
    }
    g_heap_size = size;
    *out_addr = kHeapBase;
    return ResultSuccess;
}

u64 alloc_tls_slot() {
    KGuard g(g_kernel_lock);
    u64 slot = g_tls_next;
    if ((slot & 0xFFF) == 0) map(slot, 0x1000, ThreadLocal, RW);
    g_tls_next += 0x200;
    memset((void*)slot, 0, 0x200);
    return slot;
}

u64 alloc_system(u64 size) {
    KGuard g(g_kernel_lock);
    size = (size + 0xFFF) & ~0xFFFull;
    u64 a = g_system_next;
    g_system_next += size;
    commit(a, size);
    memset((void*)a, 0, size);
    return a;
}

}  // namespace kern::mem
