// host1x command stream parser. Opcodes: switchbrew "Host1x" / Tegra X1 TRM.
#include "host1x.h"

#include <windows.h>

#include <algorithm>
#include <bit>

#include "gpu/nvdrv_device.h"

namespace host1x {

namespace {

constexpr u32 kClassHost = 0x1;
constexpr u32 kThiIncrSyncpt = 0x0, kThiMethod0 = 0x10, kThiMethod1 = 0x11;
constexpr u32 kNvdecExecute = 0xC0, kVicExecute = 0xC0;

void write(u32 engine_class, ChannelState* st, u32 method, u32 arg) {
    if (st->cur_class == kClassHost) return;  // host class: syncpoint waits are already satisfied
    if (method < 0x20) st->thi[method] = arg;
    if (method == kThiIncrSyncpt) {
        // Syncpoint increments are applied by the channel from the submit's increment list.
        return;
    }
    if (method != kThiMethod1) return;
    u32 m = st->thi[kThiMethod0];
    if (m < 0x400) st->regs[m] = arg;
    // Per-engine execute timing, logged every 200 executes (these run on the submitting guest thread).
    struct Timing {
        double sum = 0, worst = 0;
        u32 n = 0;
    };
    static Timing t_nvdec, t_vic;
    auto timed = [&](Timing& t, const char* name, auto&& fn) {
        LARGE_INTEGER f, a, b;
        QueryPerformanceFrequency(&f);
        QueryPerformanceCounter(&a);
        fn();
        QueryPerformanceCounter(&b);
        double ms = (b.QuadPart - a.QuadPart) * 1000.0 / f.QuadPart;
        t.sum += ms;
        t.worst = std::max(t.worst, ms);
        if (++t.n == 200) {
            hw_log("%s: 200 executes: avg %.2f ms, worst %.1f ms", name, t.sum / t.n, t.worst);
            t = {};
        }
    };
    if (engine_class == 0xF0 && m == kNvdecExecute) {
        st->executes++;
        timed(t_nvdec, "nvdec", [&] { video::nvdec_execute(*st); });
    } else if (engine_class == 0x5D && m == kVicExecute) {
        st->executes++;
        timed(t_vic, "vic", [&] { video::vic_execute(*st); });
    }
}

}  // namespace

void process(u32 engine_class, const u32* w, u32 count, ChannelState* st) {
    u32 i = 0;
    while (i < count) {
        u32 h = w[i++];
        u32 op = h >> 28, off = (h >> 16) & 0xFFF, val = h & 0xFFFF;
        switch (op) {
        case 0: {  // SETCLASS: offset, class, mask
            st->cur_class = (h >> 6) & 0x3FF;
            u32 mask = h & 0x3F;
            while (mask && i < count) {
                u32 b = (u32)std::countr_zero(mask);
                mask &= mask - 1;
                write(engine_class, st, off + b, w[i++]);
            }
            break;
        }
        case 1:  // INCR
            for (u32 k = 0; k < val && i < count; k++) write(engine_class, st, off + k, w[i++]);
            break;
        case 2:  // NONINCR
            for (u32 k = 0; k < val && i < count; k++) write(engine_class, st, off, w[i++]);
            break;
        case 3: {  // MASK
            u32 mask = val;
            while (mask && i < count) {
                u32 b = (u32)std::countr_zero(mask);
                mask &= mask - 1;
                write(engine_class, st, off + b, w[i++]);
            }
            break;
        }
        case 4:  // IMM
            write(engine_class, st, off, val);
            break;
        case 0xE:  // EXTEND
            break;
        default: {
            static int warned = 0;
            if (warned++ < 8) hw_log("host1x: unsupported opcode %u (0x%08x)", op, h);
            return;
        }
        }
    }
}

}  // namespace host1x
