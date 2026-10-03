// Horizon supervisor calls. Called directly from recompiled code (`svc #imm` -> hw_svc).
// Register conventions follow the HOS SVC ABI (arguments in x0-x7, results in w0 + x1..).
#include <windows.h>

#include <cstring>
#include <random>

#include "../service/ipc.h"
#include "kernel.h"
#include "sync.h"

namespace kern {

static u64 now_ticks() { return hw_cntvct(); }

static void svc_query_memory(Ctx* c) {
    mem::Region r = mem::query(c->x[2]);
    u8* out = (u8*)c->x[0];
    memset(out, 0, 0x28);
    *(u64*)(out + 0x00) = r.addr;
    *(u64*)(out + 0x08) = r.size;
    *(u32*)(out + 0x10) = r.state;
    *(u32*)(out + 0x14) = r.attr;
    *(u32*)(out + 0x18) = r.perm;
    c->x[0] = 0;
    c->x[1] = 0;  // page info
}

static Result get_info(u32 id0, Handle h, u64 id1, u64* out) {
    switch (id0) {
    case 0: *out = 0x7; return 0;                              // CoreMask: applications get cores 0-2 (npdm)
    case 1: *out = 0xFFFFFFFFull << 28; return 0;              // PriorityMask: priorities 28-59
    case 2: *out = mem::kAliasBase; return 0;
    case 3: *out = mem::kAliasSize; return 0;
    case 4: *out = mem::kHeapBase; return 0;
    case 5: *out = mem::kHeapMax; return 0;
    case 6: *out = g_process.total_memory; return 0;           // TotalMemorySize
    case 7: *out = mem::heap_size() + 0x10000000; return 0;    // UsedMemorySize (approx.)
    case 8: *out = 0; return 0;                                // DebuggerAttached
    case 9: *out = 0; return 0;                                // ResourceLimit (no handle)
    case 10: *out = 0; return 0;                               // IdleTickCount
    case 11: {                                                 // RandomEntropy
        static std::mt19937_64 rng(0x48574445 ^ GetTickCount64());
        *out = rng();
        return 0;
    }
    case 12: *out = mem::kAddressSpaceBase; return 0;          // AslrRegionAddress
    case 13: *out = mem::kAddressSpaceEnd - mem::kAddressSpaceBase; return 0;
    case 14: *out = mem::kStackBase; return 0;
    case 15: *out = mem::kStackSize; return 0;
    case 16: *out = 0; return 0;                               // SystemResourceSizeTotal
    case 17: *out = 0; return 0;
    case 18: *out = g_process.title_id; return 0;              // ProgramId
    case 20: *out = 0; return 0;                               // UserExceptionContextAddress
    case 21: *out = g_process.total_memory; return 0;          // TotalNonSystemMemorySize
    case 22: *out = mem::heap_size() + 0x10000000; return 0;   // UsedNonSystemMemorySize
    case 23: *out = 1; return 0;                               // IsApplication
    case 0xF0000002: *out = now_ticks(); return 0;             // ThreadTickCount
    default:
        hw_log("svc GetInfo: unknown id %u/%llu", id0, (unsigned long long)id1);
        *out = 0;
        return ResultInvalidEnumValue;
    }
}

static void log_unimplemented(u32 imm, Ctx* c) {
    hw_log("svc 0x%02x unimplemented (x0=0x%llx x1=0x%llx x2=0x%llx)", imm, (unsigned long long)c->x[0],
           (unsigned long long)c->x[1], (unsigned long long)c->x[2]);
}

void svc_dispatch(Ctx* c, u32 imm) {
    Result r = ResultSuccess;
    switch (imm) {
    case 0x01: {  // SetHeapSize(size) -> addr
        u64 addr = 0;
        r = mem::set_heap_size(c->x[1], &addr);
        c->x[1] = addr;
        break;
    }
    case 0x02:  // SetMemoryPermission
        mem::set_perm(c->x[0], c->x[1], (u32)c->x[2]);
        break;
    case 0x03:  // SetMemoryAttribute(addr, size, mask, attr)
        mem::set_attr(c->x[0], c->x[1], (u32)c->x[2], (u32)c->x[3]);
        break;
    case 0x04: {  // MapMemory(dst, src, size): alias src at dst
        u64 dst = c->x[0], src = c->x[1], size = c->x[2];
        mem::map(dst, size, mem::Alias, mem::RW);
        memcpy((void*)dst, (void*)src, size);
        mem::set_perm(src, size, mem::None);
        // Identity mapping cannot alias two addresses; MapMemory users (stacks) only use dst.
        break;
    }
    case 0x05: {  // UnmapMemory(dst, src, size)
        u64 dst = c->x[0], src = c->x[1], size = c->x[2];
        memcpy((void*)src, (void*)dst, size);
        mem::unmap(dst, size);
        mem::set_perm(src, size, mem::RW);
        break;
    }
    case 0x06:
        svc_query_memory(c);
        return;
    case 0x07:  // ExitProcess
        hw_log("guest called svcExitProcess");
        ExitProcess(0);
    case 0x08: {  // CreateThread(entry, arg, stack_top, priority, core) -> handle
        s32 core = (s32)c->x[5];
        if (core == -2) core = g_process.main_thread_core;
        auto t = create_thread(c->x[1], c->x[2], c->x[3], (s32)c->x[4], core);
        t->handle = g_handles.add(t);
        c->x[1] = t->handle;
        break;
    }
    case 0x09: {  // StartThread
        auto t = g_handles.get_as<Thread>((Handle)c->x[0]);
        if (!t) { r = ResultInvalidHandle; break; }
        start_thread(t.get());
        break;
    }
    case 0x0A:  // ExitThread
        exit_current_thread();
    case 0x0B: {  // SleepThread(ns)
        s64 ns = (s64)c->x[0];
        if (ns <= 0) SwitchToThread();
        else hw_sleep_ns(ns);
        return;
    }
    case 0x0C: {  // GetThreadPriority(handle) -> prio
        auto t = g_handles.get_as<Thread>((Handle)c->x[1]);
        if (!t) { r = ResultInvalidHandle; break; }
        c->x[1] = (u64)t->priority;
        break;
    }
    case 0x0D: {
        auto t = g_handles.get_as<Thread>((Handle)c->x[0]);
        if (!t) { r = ResultInvalidHandle; break; }
        t->priority = (s32)c->x[1];
        break;
    }
    case 0x0E: {  // GetThreadCoreMask(handle) -> core, mask
        auto t = g_handles.get_as<Thread>((Handle)c->x[2]);
        if (!t) { r = ResultInvalidHandle; break; }
        c->x[1] = (u64)(u32)t->ideal_core;
        c->x[2] = t->core_mask;
        break;
    }
    case 0x0F: {  // SetThreadCoreMask(handle, core, mask); core -2 = process ideal core, -1 / -3 keep it
        auto t = g_handles.get_as<Thread>((Handle)c->x[0]);
        if (!t) { r = ResultInvalidHandle; break; }
        s32 core = (s32)c->x[1];
        if (core >= 0) t->ideal_core = core;
        else if (core == -2) t->ideal_core = (s32)g_process.main_thread_core;  // -1 / -3: keep
        if (c->x[2]) t->core_mask = c->x[2];
        break;
    }
    case 0x10: {  // GetCurrentProcessorNumber: Horizon's rule, not the host CPU. Games index per-core work
                  // buffers by this, so it must agree with the thread's core mask.
        Thread* t = current_thread();
        u64 mask = t ? t->core_mask & 0xF : 1;
        s32 core = t ? t->ideal_core : 0;
        if (!mask) mask = 1;
        if (!((mask >> core) & 1)) {
            core = 0;
            while (!((mask >> core) & 1)) core++;
        }
        c->x[0] = (u64)core;
        return;
    }
    case 0x11: {  // SignalEvent(writable)
        auto e = g_handles.get_as<WritableEvent>((Handle)c->x[0]);
        if (!e) { r = ResultInvalidHandle; break; }
        e->readable->signal();
        break;
    }
    case 0x12: {  // ClearEvent(either end)
        ObjectPtr o = g_handles.get((Handle)c->x[0]);
        if (auto w = std::dynamic_pointer_cast<WritableEvent>(o)) w->readable->clear();
        else if (auto rd = std::dynamic_pointer_cast<ReadableEvent>(o)) rd->clear();
        else r = ResultInvalidHandle;
        break;
    }
    case 0x13: {  // MapSharedMemory(handle, addr, size, perm)
        auto s = g_handles.get_as<SharedMemory>((Handle)c->x[0]);
        if (!s) { r = ResultInvalidHandle; break; }
        r = map_shared_memory(s.get(), c->x[1], c->x[2], (u32)c->x[3]);
        break;
    }
    case 0x14: {  // UnmapSharedMemory
        auto s = g_handles.get_as<SharedMemory>((Handle)c->x[0]);
        if (!s) { r = ResultInvalidHandle; break; }
        unmap_shared_memory(s.get());
        break;
    }
    case 0x15: {  // CreateTransferMemory(addr, size, perm) -> handle
        auto t = std::make_shared<TransferMemory>();
        t->addr = c->x[1];
        t->size = c->x[2];
        t->perm = (u32)c->x[3];
        c->x[1] = g_handles.add(t);
        break;
    }
    case 0x16:  // CloseHandle
        g_handles.close((Handle)c->x[0]);
        break;
    case 0x17: {  // ResetSignal
        ObjectPtr o = g_handles.get((Handle)c->x[0]);
        auto e = std::dynamic_pointer_cast<ReadableEvent>(o);
        if (auto w = std::dynamic_pointer_cast<WritableEvent>(o)) e = w->readable;
        if (!e) { r = ResultInvalidHandle; break; }
        KGuard g(g_kernel_lock);
        if (!e->signaled) r = ResultInvalidState;
        e->clear();
        break;
    }
    case 0x18: {  // WaitSynchronization(handles, count, timeout) -> index
        s32 idx = -1;
        r = wait_synchronization((const Handle*)c->x[1], (s32)c->x[2], (s64)c->x[3], &idx);
        c->x[1] = (u64)(s64)idx;
        break;
    }
    case 0x19: {  // CancelSynchronization(thread)
        auto t = g_handles.get_as<Thread>((Handle)c->x[0]);
        if (!t) { r = ResultInvalidHandle; break; }
        KGuard g(g_kernel_lock);
        t->cancel_requested = true;
        wake_thread(t.get());
        break;
    }
    case 0x1A:
        r = arbitrate_lock((Handle)c->x[0], c->x[1], (Handle)c->x[2]);
        break;
    case 0x1B:
        r = arbitrate_unlock(c->x[0]);
        break;
    case 0x1C:
        r = wait_process_wide_key(c->x[0], c->x[1], (Handle)c->x[2], (s64)c->x[3]);
        break;
    case 0x1D:
        signal_process_wide_key(c->x[0], (s32)c->x[1]);
        break;
    case 0x1E:  // GetSystemTick
        c->x[0] = now_ticks();
        return;
    case 0x1F: {  // ConnectToNamedPort(name) -> handle
        Handle h = 0;
        r = ipc::connect_to_named_port((const char*)c->x[1], &h);
        c->x[1] = h;
        break;
    }
    case 0x21:  // SendSyncRequest(handle) - message in TLS
        r = ipc::send_sync_request((Handle)c->x[0], (u8*)c->tpidrro_el0);
        break;
    case 0x22:  // SendSyncRequestWithUserBuffer(buf, size, handle)
        r = ipc::send_sync_request((Handle)c->x[2], (u8*)c->x[0]);
        break;
    case 0x24:  // GetProcessId
        c->x[1] = 0x51;
        break;
    case 0x25: {  // GetThreadId(handle)
        auto t = g_handles.get_as<Thread>((Handle)c->x[1]);
        if (!t) { r = ResultInvalidHandle; break; }
        c->x[1] = t->id;
        break;
    }
    case 0x26: {  // Break(reason, arg, size)
        hw_log("guest svcBreak(reason=0x%llx, arg=0x%llx, size=0x%llx)", (unsigned long long)c->x[0],
               (unsigned long long)c->x[1], (unsigned long long)c->x[2]);
        if (c->x[0] & 0x80000000) break;  // notification only
        {  // guest backtrace via the frame-pointer chain
            hw_log("  lr=0x%llx", (unsigned long long)c->x[30]);
            u64 fp = c->x[29];
            for (int i = 0; i < 16 && fp && !(fp & 7) && fp >= 0x8000000 && fp < (1ull << 39); i++) {
                hw_log("  frame %d: 0x%llx", i, (unsigned long long)((u64*)fp)[1]);
                fp = ((u64*)fp)[0];
            }
        }
        hw_fatal("guest break");
    }
    case 0x27: {  // OutputDebugString(str, size)
        std::string s((const char*)c->x[0], (size_t)c->x[1]);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        hw_log("[guest] %s", s.c_str());
        break;
    }
    case 0x28:  // ReturnFromException
        hw_fatal("svcReturnFromException");
    case 0x29: {  // GetInfo(id0, handle, id1) -> value
        u64 v = 0;
        r = get_info((u32)c->x[1], (Handle)c->x[2], c->x[3], &v);
        c->x[1] = v;
        break;
    }
    case 0x2A: case 0x2B:  // FlushEntireDataCache / FlushDataCache
        break;
    case 0x2C:  // MapPhysicalMemory(addr, size)
        mem::map(c->x[0], c->x[1], mem::Normal, mem::RW);
        break;
    case 0x2D:  // UnmapPhysicalMemory
        mem::unmap(c->x[0], c->x[1]);
        break;
    case 0x32:  // SetThreadActivity
        break;
    case 0x34:
        r = wait_for_address(c->x[0], (u32)c->x[1], (s32)c->x[2], (s64)c->x[3]);
        break;
    case 0x35:
        r = signal_to_address(c->x[0], (u32)c->x[1], (s32)c->x[2], (s32)c->x[3]);
        break;
    case 0x36:  // SynchronizePreemptionState
        break;
    case 0x45: {  // CreateEvent() -> writable, readable
        auto rd = std::make_shared<ReadableEvent>();
        auto wr = std::make_shared<WritableEvent>();
        wr->readable = rd;
        c->x[1] = g_handles.add(wr);
        c->x[2] = g_handles.add(rd);
        break;
    }
    default:
        log_unimplemented(imm, c);
        r = MakeResult(1, 1);
        break;
    }
    c->x[0] = r;
}

}  // namespace kern

extern "C" void hw_svc(Ctx* c, u32 imm) { kern::svc_dispatch(c, imm); }
