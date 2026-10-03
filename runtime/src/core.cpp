// HWDER runtime core: code dispatch, logging/crash reporting, and process boot.
#include <exception>
#include <csignal>
#include <windows.h>
#include <psapi.h>
#include <windows.h>
#include <dbghelp.h>
#include <timeapi.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "hwder/dispatch.h"
#include "hwder/module.h"
#include "hwder/runtime.h"
#include "kernel/kernel.h"
#include "service/ipc.h"
#include "gpu/display.h"
#include "fs/nsp.h"

// ------------------------------------------------------------------ dispatch
// One dense table per module text segment: (addr - text_lo) / 4 -> host function.
struct TextTable {
    u64 lo, hi;
    std::vector<HwFn> fns;
};
static std::vector<TextTable> g_tables;
extern "C" const unsigned long long hw_resumable[];
extern "C" const unsigned hw_resumable_count;
static std::vector<u64> g_starts;  // sorted function starts

static void build_dispatch() {
    for (unsigned m = 0; m < hw_module_count; m++) {
        TextTable t{hw_modules[m].text_lo, hw_modules[m].text_hi, {}};
        t.fns.assign((t.hi - t.lo) >> 2, nullptr);
        g_tables.push_back(std::move(t));
    }
    for (unsigned i = 0; i < hw_func_count; i++) {
        const HwFuncEntry& e = hw_func_table[i];
        for (auto& t : g_tables)
            if (e.addr >= t.lo && e.addr < t.hi) t.fns[(e.addr - t.lo) >> 2] = e.fn;
    }
    g_starts.reserve(hw_func_count);
    for (unsigned i = 0; i < hw_func_count; i++) g_starts.push_back(hw_func_table[i].addr);
    std::sort(g_starts.begin(), g_starts.end());
    hw_log("dispatch: %u functions in %u modules (%u resumable)", hw_func_count, hw_module_count, hw_resumable_count);
}

extern "C" HwFn hw_lookup(u64 addr) {
    for (auto& t : g_tables)
        if (addr >= t.lo && addr < t.hi) return t.fns[(addr - t.lo) >> 2];
    return nullptr;
}

// Paths are UTF-8 internally (the ANSI code page cannot spell every folder name a user may have).
static std::wstring to_w(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}
static std::string to_u8(const std::wstring& w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}
static std::string env_u8(const char* name) {
    const wchar_t* e = _wgetenv(to_w(name).c_str());
    return e ? to_u8(e) : std::string();
}
static FILE* fopen_u8(const std::string& path, const char* mode) { return _wfopen(to_w(path).c_str(), to_w(mode).c_str()); }

static std::string exe_dir() {
    std::wstring p(32768, L'\0');
    DWORD n = GetModuleFileNameW(nullptr, p.data(), (DWORD)p.size());
    p.resize(n);
    std::string s = to_u8(p);
    return s.substr(0, s.find_last_of("\\/"));
}


// Functions with an unresolved indirect branch accept entry at any instruction (see gen.py):
// the dispatcher sets c->pc = target | 1 and the function's prologue jumps to that label.
static bool try_resume(Ctx* c) {
    u64 pc = c->pc;
    auto it = std::upper_bound(g_starts.begin(), g_starts.end(), pc);
    if (it == g_starts.begin()) return false;
    u64 start = *(it - 1);
    if (!std::binary_search(hw_resumable, hw_resumable + hw_resumable_count, start)) return false;
    HwFn f = hw_lookup(start);
    if (!f) return false;
    static std::atomic<int> logged{0};
    if (logged++ < 20) hw_log("dispatch: resuming 0x%llx inside f_%llx", (unsigned long long)pc, (unsigned long long)start);
    c->pc = pc | 1;
    f(c);
    return true;
}

static void unknown_target(Ctx* c) {
    if (try_resume(c)) return;
    hw_log("FATAL: jump to unrecompiled guest address 0x%llx (lr=0x%llx)", (unsigned long long)c->pc,
           (unsigned long long)c->x[30]);
    for (auto& t : g_tables) {
        if (c->pc >= t.lo && c->pc < t.hi) {
            // Record it so the next generator run adds it as an entry point.
            if (FILE* f = fopen_u8(exe_dir() + "\\extra_entries.txt", "a")) {
                fprintf(f, "0x%llx\n", (unsigned long long)c->pc);
                fclose(f);
            }
            hw_log("recorded 0x%llx in extra_entries.txt - regenerate to include it", (unsigned long long)c->pc);
        }
    }
    hw_fatal("unrecompiled target");
}

extern "C" void hw_call(Ctx* c, u64 addr) {
    HwFn f = hw_lookup(addr);
    if (!f) {
        c->pc = addr;
        unknown_target(c);
        return;
    }
    f(c);
}

extern "C" void hw_dispatch(Ctx* c) {
    HwFn f = hw_lookup(c->pc);
    if (!f) {
        unknown_target(c);
        return;
    }
    HW_MUSTTAIL return f(c);
}

// ------------------------------------------------------------------ misc runtime hooks
extern "C" u64 hw_cntvct(void) {
    static LARGE_INTEGER freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (u64)((double)t.QuadPart * (19200000.0 / (double)freq.QuadPart));
}
extern "C" void hw_yield(void) { SwitchToThread(); }
extern "C" void hw_brk(Ctx* c, u64 pc, u32 imm) {
    hw_log("guest brk #0x%x at 0x%llx (x0=0x%llx)", imm, (unsigned long long)pc, (unsigned long long)c->x[0]);
    hw_fatal("guest abort");
}
extern "C" void hw_unimpl(Ctx* c, u64 pc, u32 word) {
    hw_log("unimplemented instruction %08x at 0x%llx", word, (unsigned long long)pc);
    hw_fatal("unimplemented instruction");
}
extern "C" void hw_sleep_ns(s64 ns) {
    if (ns <= 0) {
        SwitchToThread();
        return;
    }
    static thread_local HANDLE timer =
        CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    LARGE_INTEGER due;
    due.QuadPart = -(ns / 100);
    if (timer && SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) WaitForSingleObject(timer, INFINITE);
    else Sleep((DWORD)((ns + 999999) / 1000000));
}

// ------------------------------------------------------------------ logging / crashes
static FILE* g_log;
static SRWLOCK g_log_lock = SRWLOCK_INIT;

extern "C" void hw_log(const char* fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    static const ULONGLONG t0 = GetTickCount64();
    unsigned ms = (unsigned)(GetTickCount64() - t0);
    AcquireSRWLockExclusive(&g_log_lock);
    fprintf(stderr, "[hwder] %s\n", buf);
    if (g_log) {
        fprintf(g_log, "[%3u.%03u] %s\n", ms / 1000, ms % 1000, buf);
        fflush(g_log);
    }
    ReleaseSRWLockExclusive(&g_log_lock);
}

extern "C" void hw_fatal(const char* why) {
    hw_log("fatal: %s", why);
    if (IsDebuggerPresent()) __debugbreak();
    if (getenv("HWDER_HOLD")) {  // keep the window up showing the reason
        display::set_status(std::string("halted: ") + why);
        for (;;) Sleep(1000);
    }
    ExitProcess(1);
}

static LONG WINAPI crash_handler(EXCEPTION_POINTERS* ep) {
    auto* er = ep->ExceptionRecord;
    if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION && er->ExceptionCode != EXCEPTION_ILLEGAL_INSTRUCTION &&
        er->ExceptionCode != EXCEPTION_INT_DIVIDE_BY_ZERO && er->ExceptionCode != EXCEPTION_STACK_OVERFLOW)
        return EXCEPTION_CONTINUE_SEARCH;
    HANDLE proc = GetCurrentProcess();
    char symbuf[sizeof(SYMBOL_INFO) + 256] = {};
    auto* sym = (SYMBOL_INFO*)symbuf;
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 disp = 0;
    const char* name = SymFromAddr(proc, (DWORD64)er->ExceptionAddress, &disp, sym) ? sym->Name : "?";
    auto modname = [](const void* pc, char* out, size_t n) {
        HMODULE mod = nullptr;
        strcpy_s(out, n, "?");
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)pc, &mod))
            GetModuleBaseNameA(GetCurrentProcess(), mod, out, (DWORD)n);
    };
    char mn[MAX_PATH];
    modname(er->ExceptionAddress, mn, sizeof(mn));
    PWSTR tdesc = nullptr;
    GetThreadDescription(GetCurrentThread(), &tdesc);
    hw_log("CRASH code=0x%08lx at %p (%s!%s+0x%llx) access=%s 0x%llx (thread %lu %ls)", er->ExceptionCode,
           er->ExceptionAddress, mn, name, (unsigned long long)disp,
           er->NumberParameters > 0 && er->ExceptionInformation[0] ? "write" : "read",
           er->NumberParameters > 1 ? (unsigned long long)er->ExceptionInformation[1] : 0ull, GetCurrentThreadId(), tdesc ? tdesc : L"");
    CONTEXT ctx = *ep->ContextRecord;
    STACKFRAME64 sf = {};
    sf.AddrPC.Offset = ctx.Rip;
    sf.AddrPC.Mode = AddrModeFlat;
    sf.AddrFrame.Offset = ctx.Rbp;
    sf.AddrFrame.Mode = AddrModeFlat;
    sf.AddrStack.Offset = ctx.Rsp;
    sf.AddrStack.Mode = AddrModeFlat;
    for (int i = 0; i < 40; i++) {
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, GetCurrentThread(), &sf, &ctx, nullptr,
                         SymFunctionTableAccess64, SymGetModuleBase64, nullptr) || !sf.AddrPC.Offset)
            break;
        disp = 0;
        const char* n = SymFromAddr(proc, sf.AddrPC.Offset, &disp, sym) ? sym->Name : "?";
        char fm[MAX_PATH];
        modname((const void*)sf.AddrPC.Offset, fm, sizeof(fm));
        hw_log("  #%02d %s!%s+0x%llx", i, fm, n, (unsigned long long)disp);
    }
    ExitProcess(2);
}

// Log the calling thread's host stack (guest functions appear as f_<addr>).
extern "C" void hw_log_backtrace(const char* tag) {
    void* pcs[32];
    USHORT n = RtlCaptureStackBackTrace(1, 32, pcs, nullptr);
    HANDLE proc = GetCurrentProcess();
    char symbuf[sizeof(SYMBOL_INFO) + 256] = {};
    auto* sym = (SYMBOL_INFO*)symbuf;
    std::string line;
    for (USHORT i = 0; i < n; i++) {
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 255;
        DWORD64 disp = 0;
        const char* name = SymFromAddr(proc, (DWORD64)pcs[i], &disp, sym) ? sym->Name : "?";
        if (strncmp(name, "f_", 2) != 0) continue;  // guest frames only
        line += " ";
        line += name;
    }
    hw_log("%s backtrace:%s", tag, line.c_str());
}

// ------------------------------------------------------------------ thread dump (HWDER_THREAD_DUMP=<seconds>)
// Periodically logs every guest thread's host call stack (guest functions appear as f_<addr>).
static void dump_threads() {
    HANDLE proc = GetCurrentProcess();
    for (auto& t : kern::all_threads()) {
        if (!t->started || t->exited || !t->host_handle) continue;
        u64 pcs[24];
        int n = 0;
        if (SuspendThread(t->host_handle) == (DWORD)-1) continue;
        CONTEXT ctx = {};
        ctx.ContextFlags = CONTEXT_FULL;
        if (GetThreadContext(t->host_handle, &ctx)) {
            // RtlVirtualUnwind does not allocate, so it is safe while the target is suspended.
            for (; n < 24 && ctx.Rip; n++) {
                pcs[n] = ctx.Rip;
                DWORD64 base = 0;
                PRUNTIME_FUNCTION rf = RtlLookupFunctionEntry(ctx.Rip, &base, nullptr);
                if (!rf) {
                    ctx.Rip = *(DWORD64*)ctx.Rsp;
                    ctx.Rsp += 8;
                    continue;
                }
                void* hd = nullptr;
                DWORD64 ef = 0;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, ctx.Rip, rf, &ctx, &hd, &ef, nullptr);
            }
        }
        ResumeThread(t->host_handle);
        std::string line;
        char symbuf[sizeof(SYMBOL_INFO) + 256] = {};
        auto* sym = (SYMBOL_INFO*)symbuf;
        for (int i = 0; i < n; i++) {
            sym->SizeOfStruct = sizeof(SYMBOL_INFO);
            sym->MaxNameLen = 255;
            DWORD64 disp = 0;
            char b[300];
            snprintf(b, sizeof(b), " %s+0x%llx", SymFromAddr(proc, pcs[i], &disp, sym) ? sym->Name : "?",
                     (unsigned long long)disp);
            line += b;
            if (i < n - 1) line += " <";
        }
        hw_log("thread %llu '%s' prio %d%s:%s", (unsigned long long)t->id, t->name.c_str(), t->priority,
               t->waiting ? " [waiting]" : "", line.c_str());
    }
}
static void start_thread_dumper() {
    const char* e = getenv("HWDER_THREAD_DUMP");
    if (!e || !atoi(e)) return;
    static int period = atoi(e);
    CreateThread(nullptr, 0, [](void*) -> DWORD {
        for (;;) {
            Sleep(period * 1000);
            hw_log("---- thread dump ----");
            dump_threads();
        }
    }, nullptr, 0, nullptr);
}

// ------------------------------------------------------------------ NSO loading
static bool lz4_decompress(const u8* src, size_t src_size, u8* dst, size_t dst_size) {
    const u8* ip = src;
    const u8* iend = src + src_size;
    u8* op = dst;
    u8* oend = dst + dst_size;
    while (ip < iend) {
        u8 token = *ip++;
        size_t lit = token >> 4;
        if (lit == 15) {
            u8 b;
            do { b = *ip++; lit += b; } while (b == 255);
        }
        if (op + lit > oend || ip + lit > iend) return false;
        memcpy(op, ip, lit);
        op += lit;
        ip += lit;
        if (ip >= iend) break;
        size_t off = ip[0] | (ip[1] << 8);
        ip += 2;
        size_t len = token & 15;
        if (len == 15) {
            u8 b;
            do { b = *ip++; len += b; } while (b == 255);
        }
        len += 4;
        u8* match = op - off;
        if (match < dst || op + len > oend) return false;
        for (size_t i = 0; i < len; i++) op[i] = match[i];  // overlapping copy
        op += len;
    }
    return op == oend;
}

static std::vector<u8> read_file(const std::string& path) {
    std::vector<u8> d;
    FILE* f = fopen_u8(path, "rb");
    if (!f) return d;
    fseek(f, 0, SEEK_END);
    d.resize((size_t)ftell(f));
    fseek(f, 0, SEEK_SET);
    fread(d.data(), 1, d.size(), f);
    fclose(f);
    return d;
}

// ExeFS file: from the NSP when one is loaded, else from the extracted exefs directory.
static std::vector<u8> read_exefs(const std::string& exefs, const char* name) {
    std::vector<u8> d;
    if (hwfs::nsp::active()) {
        hwfs::nsp::read_exefs(name, d);
        return d;
    }
    return read_file(exefs + "\\" + name);
}

static void load_module(const HwModuleInfo& mi, const std::string& exefs) {
    std::vector<u8> nso = read_exefs(exefs, mi.name);
    if (nso.size() < 0x100 || memcmp(nso.data(), "NSO0", 4) != 0) {
        hw_log("loader: cannot read %s from %s", mi.name, hwfs::nsp::active() ? hwfs::nsp::path().c_str() : exefs.c_str());
        hw_fatal("missing ExeFS module");
    }
    u32 flags = *(u32*)&nso[0xC];
    struct Seg { u32 file_off, mem_off, size, extra; } seg[3];
    memcpy(seg, &nso[0x10], sizeof(seg));
    u32 csize[3];
    memcpy(csize, &nso[0x60], sizeof(csize));
    u64 bss = seg[2].extra;
    u64 image = (seg[2].mem_off + seg[2].size + bss + 0xFFF) & ~0xFFFull;
    kern::mem::map(mi.base, image, kern::mem::CodeData, kern::mem::RW);
    for (int i = 0; i < 3; i++) {
        u8* dst = (u8*)(mi.base + seg[i].mem_off);
        if (flags & (1u << i)) {
            if (!lz4_decompress(&nso[seg[i].file_off], csize[i], dst, seg[i].size)) hw_fatal("lz4");
        } else {
            memcpy(dst, &nso[seg[i].file_off], seg[i].size);
        }
    }
    // Experiment (HWDER_PATCH_DT=<hz>): the game's per-frame time step lives in two rodata floats
    // (1/60 s, plus 1/30 s for its half-rate mode). Rewriting them to 1/hz together with releasing
    // frames at hz tells us whether the simulation is dt-driven (then high refresh is feasible).
    if (!strcmp(mi.name, "main")) {
        if (const char* hz = getenv("HWDER_PATCH_DT")) {
            float f = (float)atof(hz);
            if (f >= 30.0f) {
                float dt = 1.0f / f, dt2 = 2.0f / f;
                memcpy((void*)(mi.base + 0x73c7d0), &dt, 4);
                memcpy((void*)(mi.base + 0x73f038), &dt, 4);
                memcpy((void*)(mi.base + 0x73c2ac), &dt2, 4);
                hw_log("loader: patched frame delta constants to 1/%.0f s", f);
            }
        }
    }
    u64 text_end = (seg[0].mem_off + seg[0].size + 0xFFF) & ~0xFFFull;
    u64 ro_end = (seg[1].mem_off + seg[1].size + 0xFFF) & ~0xFFFull;
    kern::mem::map(mi.base, text_end, kern::mem::Code, kern::mem::RX, false);
    kern::mem::map(mi.base + seg[1].mem_off, ro_end - seg[1].mem_off, kern::mem::Code, kern::mem::R, false);
    kern::mem::map(mi.base + seg[2].mem_off, image - seg[2].mem_off, kern::mem::CodeData, kern::mem::RW, false);
    hw_log("loader: %-8s at 0x%llx (0x%llx bytes)", mi.name, (unsigned long long)mi.base, (unsigned long long)image);
}

static void load_npdm(const std::string& exefs) {
    std::vector<u8> d = read_exefs(exefs, "main.npdm");
    if (d.size() < 0x80 || memcmp(d.data(), "META", 4) != 0) return;
    kern::g_process.main_thread_priority = d[0xE];
    kern::g_process.main_thread_core = d[0xF];
    kern::g_process.main_thread_stack = *(u32*)&d[0x1C];
    hw_log("npdm: main thread prio %u core %u stack 0x%x", kern::g_process.main_thread_priority,
           kern::g_process.main_thread_core, kern::g_process.main_thread_stack);
}

// ------------------------------------------------------------------ boot
void hw_config_load(const std::string& dir);

int main(int argc, char** argv) {
    std::string dir = exe_dir();
    SetCurrentDirectoryW(to_w(dir).c_str());  // caches, settings and saves live beside the exe
    std::string log_path = env_u8("HWDER_LOG");
    g_log = fopen_u8(log_path.empty() ? dir + "\\hwder.log" : log_path, "w");
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    AddVectoredExceptionHandler(1, crash_handler);
    // Deaths that bypass SEH: C++ terminate (uncaught exception, e.g. in a worker thread), abort(),
    // CRT invalid-parameter. Log them with the thread id instead of dying silently.
    std::set_terminate([] {
        hw_log("TERMINATE: std::terminate called (uncaught exception?) on thread %lu", GetCurrentThreadId());
        ExitProcess(3);
    });
    signal(SIGABRT, [](int) {
        PWSTR desc = nullptr;
        GetThreadDescription(GetCurrentThread(), &desc);
        hw_log("ABORT: abort() called on thread %lu (%ls)", GetCurrentThreadId(), desc ? desc : L"?");
        void* pcs[48];
        USHORT n = RtlCaptureStackBackTrace(0, 48, pcs, nullptr);
        char symbuf[sizeof(SYMBOL_INFO) + 256] = {};
        auto* sym = (SYMBOL_INFO*)symbuf;
        for (USHORT i = 0; i < n; i++) {
            sym->SizeOfStruct = sizeof(SYMBOL_INFO);
            sym->MaxNameLen = 255;
            DWORD64 disp = 0;
            const char* name = SymFromAddr(GetCurrentProcess(), (DWORD64)pcs[i], &disp, sym) ? sym->Name : "?";
            HMODULE mod = nullptr;
            char modname[MAX_PATH] = "?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)pcs[i], &mod))
                GetModuleBaseNameA(GetCurrentProcess(), mod, modname, sizeof(modname));
            hw_log("  #%02u %s!%s+0x%llx", i, modname, name, (unsigned long long)disp);
        }
        ExitProcess(3);
    });
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _set_invalid_parameter_handler([](const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t) {
        hw_log("CRT: invalid parameter on thread %lu", GetCurrentThreadId());
    });
    timeBeginPeriod(1);
    hw_log("HWDER - Hyrule Warriors: Definitive Edition static recompilation");
    hw_config_load(dir);

    std::string exefs = env_u8("HWDER_EXEFS");
    if (exefs.empty()) exefs = dir + "\\exefs";
    {
        // Game data: an NSP beside the exe (decrypted in-process with the user's prod.keys) or an
        // extracted exefs/ + romfs/ pair. Nothing is written out either way.
        std::string err;
        bool nsp = hwfs::nsp::init(dir, err);
        bool have_dir = GetFileAttributesW(to_w(exefs + "\\main").c_str()) != INVALID_FILE_ATTRIBUTES;
        if (!nsp && !have_dir) {
            std::string msg = err.empty() ? "No game data found.\n\nPut your own dump of Hyrule Warriors: Definitive Edition "
                                            "(the .nsp) and your prod.keys next to hwder.exe.\n\n(Looked for *.nsp in: " + dir + ")"
                                          : "Cannot load the game from the NSP:\n\n" + err;
            hw_log("boot: %s", msg.c_str());
            if (!getenv("HWDER_MUTE")) MessageBoxW(nullptr, to_w(msg).c_str(), L"HWDER", MB_ICONERROR | MB_OK);
            return 2;
        }
        hw_log("boot: game data from %s", nsp ? hwfs::nsp::path().c_str() : exefs.c_str());
    }
    kern::mem::init();
    build_dispatch();
    display::init();
    ipc::register_all_services();
    load_npdm(exefs);
    for (unsigned i = 0; i < hw_module_count; i++) load_module(hw_modules[i], exefs);

    // Main thread: stack in the stack region, entry at rtld (the first module), x1 = own handle.
    u64 stack_size = (kern::g_process.main_thread_stack + 0xFFF) & ~0xFFFull;
    u64 stack_base = kern::mem::kStackBase;
    kern::mem::map(stack_base, stack_size, kern::mem::Stack, kern::mem::RW);
    auto t = kern::create_thread(hw_modules[0].text_lo, 0, stack_base + stack_size,
                                 (s32)kern::g_process.main_thread_priority, (s32)kern::g_process.main_thread_core);
    t->handle = kern::g_handles.add(t);
    t->ctx->x[1] = t->handle;
    t->name = "MainThread";
    hw_log("boot: main thread entry 0x%llx, stack 0x%llx+0x%llx", (unsigned long long)t->entry,
           (unsigned long long)stack_base, (unsigned long long)stack_size);
    kern::start_thread(t.get());
    start_thread_dumper();
    WaitForSingleObject(t->host_handle, INFINITE);
    hw_log("main thread exited");
    return 0;
}
