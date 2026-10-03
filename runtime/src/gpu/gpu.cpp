// GPU command processor: GPFIFO entries -> pushbuffer methods -> engines.
#include "gpu.h"

#include <windows.h>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "maxwell_3d.h"
#include "nvdrv_device.h"
#include "swizzle.h"

extern "C" u64 hw_cntvct();

namespace gpu {

static Renderer* g_renderer;
Renderer* renderer() {
    static std::once_flag once;
    std::call_once(once, [] { g_renderer = create_renderer(create_address_space()); });
    return g_renderer;
}

static bool trace_enabled() {
    static const bool t = getenv("HWDER_GPU_TRACE") != nullptr;
    return t;
}

static u64 gpu_time_ns() { return hw_cntvct() * 625 / 12; }

// Logs engine operations that take longer than `limit` ms (stall diagnostics).
struct OpTimer {
    const char* what;
    double limit;
    u64 extra;
    LARGE_INTEGER a;
    OpTimer(const char* w, u64 x = 0, double l = 30.0) : what(w), limit(l), extra(x) { QueryPerformanceCounter(&a); }
    ~OpTimer() {
        LARGE_INTEGER b, f;
        QueryPerformanceCounter(&b);
        QueryPerformanceFrequency(&f);
        double ms = (b.QuadPart - a.QuadPart) * 1000.0 / f.QuadPart;
        if (ms > limit) hw_log("gpu: slow: %s (%llu) took %.0f ms", what, (unsigned long long)extra, ms);
    }
};

#define LOG_ONCE(...)                       \
    do {                                    \
        static bool once_ = false;          \
        if (!once_) {                       \
            once_ = true;                   \
            hw_log(__VA_ARGS__);            \
        }                                   \
    } while (0)

// ------------------------------------------------------------------ macro (MME) interpreter
class Macro {
public:
    u32 code[0x2000] = {};
    u32 start[0x80] = {};
    u32 code_ptr = 0, start_ptr = 0;

    using Send = void (*)(void* ctx, u32 method, u32 value);

    // Execute macro `index` with parameters; writes go to send(method, value).
    void run(u32 index, const std::vector<u32>& params, void* ctx, Send send) {
        u32 regs[8] = {};
        u32 pc = start[index & 0x7F];
        u32 method = 0, increment = 0;
        size_t next_param = 1;
        regs[1] = params.empty() ? 0 : params[0];
        bool carry = false;
        auto fetch = [&]() -> u32 {
            if (next_param < params.size()) return params[next_param++];
            LOG_ONCE("mme: macro %u fetched past its %zu parameters", index, params.size());
            return 0;
        };
        auto set_reg = [&](u32 r, u32 v) {
            if (r) regs[r] = v;
        };
        auto do_send = [&](u32 v) {
            send(ctx, method, v);
            method += increment;
        };
        auto set_method = [&](u32 v) {
            method = v & 0xFFF;
            increment = (v >> 12) & 0x3F;
        };
        bool exiting = false;
        s32 delayed_jump = -1;
        for (u32 steps = 0; steps < 0x100000; steps++) {
            u32 in = code[pc & 0x1FFF];
            u32 op = in & 7, assign = (in >> 4) & 7, dst = (in >> 8) & 7, ra = (in >> 11) & 7, rb = (in >> 14) & 7;
            s32 imm = (s32)in >> 14;
            bool is_exit = (in >> 7) & 1;
            u32 next_pc = pc + 1;
            bool branch_taken = false;
            if (op == 7) {  // branch
                bool cond_nz = (in >> 4) & 1, annul = (in >> 5) & 1;
                bool take = cond_nz ? regs[ra] != 0 : regs[ra] == 0;
                if (take) {
                    if (annul) {
                        pc = pc + imm;
                        continue;
                    }
                    delayed_jump = (s32)(pc + imm);
                    branch_taken = true;
                }
            } else {
                u32 result = 0;
                switch (op) {
                case 0: {  // ALU
                    u32 a = regs[ra], b = regs[rb];
                    u32 alu = (in >> 17) & 0x1F;
                    u64 r;
                    switch (alu) {
                    case 0: r = (u64)a + b; carry = r >> 32; result = (u32)r; break;
                    case 1: r = (u64)a + b + carry; carry = r >> 32; result = (u32)r; break;
                    case 2: r = (u64)a - b; carry = a >= b; result = (u32)r; break;
                    case 3: r = (u64)a - b - !carry; carry = (u64)a >= (u64)b + !carry; result = (u32)r; break;
                    case 8: result = a ^ b; break;
                    case 9: result = a | b; break;
                    case 10: result = a & b; break;
                    case 11: result = a & ~b; break;
                    case 12: result = ~(a & b); break;
                    default: LOG_ONCE("mme: alu op %u", alu); break;
                    }
                    break;
                }
                case 1: result = regs[ra] + imm; break;  // AddImmediate
                case 2: case 3: case 4: {               // bitfield ops
                    u32 src_bit = (in >> 17) & 0x1F, size = (in >> 22) & 0x1F, dst_bit = (in >> 27) & 0x1F;
                    u32 mask = size >= 32 ? 0xFFFFFFFFu : ((1u << size) - 1);
                    if (op == 2) {  // insert: (rb >> src_bit) & mask placed at dst_bit in ra
                        u32 ins = (regs[rb] >> src_bit) & mask;
                        result = (regs[ra] & ~(mask << dst_bit)) | (ins << dst_bit);
                    } else if (op == 3) {  // extract (rb >> ra) & mask << dst_bit
                        result = ((regs[rb] >> regs[ra]) & mask) << dst_bit;
                    } else {  // extract (rb >> src_bit) & mask << ra
                        result = ((regs[rb] >> src_bit) & mask) << regs[ra];
                    }
                    break;
                }
                case 5: result = read_reg_(ctx_read_, ctx, regs[ra] + imm); break;  // Read
                default: LOG_ONCE("mme: opcode %u", op); break;
                }
                switch (assign) {
                case 0: set_reg(dst, fetch()); break;
                case 1: set_reg(dst, result); break;
                case 2: set_reg(dst, result); set_method(result); break;
                case 3: set_reg(dst, fetch()); do_send(result); break;
                case 4: set_reg(dst, result); do_send(result); break;
                case 5: set_reg(dst, fetch()); set_method(result); break;
                case 6: set_reg(dst, result); set_method(result); do_send(fetch()); break;
                case 7: set_reg(dst, result); set_method(result); do_send((result >> 12) & 0x3F); break;
                }
            }
            if (exiting) return;  // the delay slot after an exit just executed
            if (is_exit) exiting = true;
            if (delayed_jump >= 0 && !branch_taken) {
                pc = (u32)delayed_jump;
                delayed_jump = -1;
                continue;
            }
            pc = next_pc;
        }
        hw_log("mme: macro %u did not terminate", index);
    }
    u32 (*ctx_read_)(void*, u32) = nullptr;
    static u32 read_reg_(u32 (*f)(void*, u32), void* ctx, u32 reg) { return f ? f(ctx, reg) : 0; }
};

// ------------------------------------------------------------------ engines
struct Channel;

struct Engine3D {
    u32 regs[m3d::kNumRegs] = {};
    Macro mme;
    // pending macro call
    s32 macro_index = -1;
    std::vector<u32> macro_params;
    // draw state
    bool indexed = false;
    u32 instance = 0;
    std::vector<u8> inline_indices;
    MemoryManager* mm = nullptr;
    Channel* ch = nullptr;
    // Constant buffer bindings are channel state; the renderer holds one set, re-sent on channel switch.
    struct CbufBind {
        u64 va = 0;
        u32 size = 0;
        bool valid = false;
    } cbufs[5][18];
};

static Engine3D* g_cbuf_owner = nullptr;  // engine whose bindings the renderer currently holds (GPU thread only)

static void sync_cbufs(Engine3D& e) {
    if (g_cbuf_owner == &e) return;
    g_cbuf_owner = &e;
    for (u32 s = 0; s < 5; s++)
        for (u32 i = 0; i < 18; i++)
            renderer()->bind_const_buffer(s, i, e.cbufs[s][i].va, e.cbufs[s][i].size, e.cbufs[s][i].valid);
}

struct Engine2D {
    u32 regs[0x400] = {};
};

struct EngineCompute {
    u32 regs[0xE00] = {};
};

struct EngineDma {
    u32 regs[0x800] = {};
};

struct EngineI2M {
    u32 regs[0x80] = {};
    // current upload
    u64 dst = 0;
    u32 line_length = 0, line_count = 0;
    std::vector<u8> data;
    bool active = false;
    bool block_linear = false;
    u32 pitch = 0, width = 0, height = 0, depth = 1, block_h = 0, block_d = 0, origin_x = 0, origin_y = 0, layer = 0;
};

struct Channel {
    MemoryManager* mm;
    u32 syncpt = 0;  // channel syncpoint (from the last submission; for tracing)
    u32 bound_class[8] = {};
    Engine3D e3d;
    Engine2D e2d;
    EngineCompute compute;
    EngineDma dma;
    EngineI2M i2m3d, i2m_compute, i2m;
    // puller (GPFIFO class) state
    u32 sem_addr_hi = 0, sem_addr_lo = 0, sem_payload = 0, syncpt_payload = 0, reference = 0;
};

// ------------------------------------------------------------------ inline to memory
static void i2m_launch(Channel* ch, EngineI2M& e, const u32* r, u32 launch) {
    e.dst = ((u64)r[2] << 32) | r[3];
    e.line_length = r[0];
    e.line_count = r[1];
    e.pitch = r[4];
    e.block_linear = (launch & 1) == 0;
    e.block_h = (r[5] >> 4) & 0xF;
    e.block_d = (r[5] >> 8) & 0xF;
    e.width = r[6];
    e.height = r[7];
    e.depth = r[8] ? r[8] : 1;
    e.layer = r[9];
    e.origin_x = r[10];
    e.origin_y = r[11];
    e.data.clear();
    e.data.reserve((size_t)e.line_length * e.line_count);
    e.active = true;
}
static void i2m_data(Channel* ch, EngineI2M& e, u32 value) {
    if (!e.active) return;
    OpTimer t("inline-to-memory", e.line_length * (u64)e.line_count);
    u8* p = (u8*)&value;
    e.data.insert(e.data.end(), p, p + 4);
    size_t total = (size_t)e.line_length * e.line_count;
    if (e.data.size() < total) return;
    e.active = false;
    Renderer* rnd = renderer();
    MemoryManager* mm = ch->mm;
    if (!e.block_linear) {
        if (e.line_count <= 1 || e.pitch == e.line_length) {
            rnd->invalidate_region((u64)mm->translate(e.dst), total);
            mm->write(e.dst, e.data.data(), total);
        } else {
            for (u32 l = 0; l < e.line_count; l++) mm->write(e.dst + (u64)l * e.pitch, &e.data[(size_t)l * e.line_length], e.line_length);
            rnd->invalidate_region((u64)mm->translate(e.dst), (u64)e.pitch * e.line_count);
        }
        return;
    }
    BlockLinear bl{e.width, e.height, e.depth, e.block_h, e.block_d};
    u64 size = bl.size();
    u8* base = mm->translate(e.dst);
    if (!base || mm->contiguous_size(e.dst, size) < size) {
        LOG_ONCE("i2m: block-linear destination not contiguous");
        return;
    }
    rnd->flush_region((u64)base, size);
    swizzle(bl, e.data.data(), base, e.line_length, e.origin_x, e.origin_y, e.line_length, e.line_count, e.layer);
    rnd->invalidate_region((u64)base, size);
}

// ------------------------------------------------------------------ semaphores / reports
static void write_report(Channel* ch, u64 addr, u32 payload, bool short_form) {
    if (short_form) {
        ch->mm->write(addr, &payload, 4);
    } else {
        struct {
            u64 value;
            u64 timestamp;
        } r{payload, gpu_time_ns()};
        ch->mm->write(addr, &r, 16);
    }
    renderer()->invalidate_region((u64)ch->mm->translate(addr), short_form ? 4 : 16);
}

static void report_semaphore(Engine3D& e, u32 query) {
    Channel* ch = e.ch;
    u64 addr = m3d::addr(e.regs, m3d::ReportSemaphore);
    u32 payload = e.regs[m3d::ReportSemaphore + 2];
    u32 op = query & 3;
    bool short_form = (query >> 28) & 1;
    u32 counter = (query >> 23) & 0x1F;
    switch (op) {
    case 0: {  // release
        renderer()->flush_commands();
        if (short_form) {
            renderer()->on_gpu_idle([ch, addr, payload] { write_report(ch, addr, payload, true); });
        } else {
            renderer()->on_gpu_idle([ch, addr, payload] { write_report(ch, addr, payload, false); });
        }
        break;
    }
    case 1: {  // acquire: wait until memory == payload
        renderer()->flush_commands();
        for (int spin = 0;; spin++) {
            u32 v = 0;
            ch->mm->read(addr, &v, 4);
            if (v == payload) break;
            if (spin > 1000) std::this_thread::yield();
        }
        break;
    }
    case 2: {  // report counter value
        u64 value = 0;
        {
            static u32 seen = 0;
            if (counter < 32 && !(seen & (1u << counter))) {
                seen |= 1u << counter;
                hw_log("3d: report counter type 0x%x first requested (short %d)", counter, (int)short_form);
            }
        }
        switch (counter) {
        case 0: value = 0; break;           // zero
        case 1: value = 0; break;           // vertices generated...
        // Samples passed (occlusion). The game fades head-up markers (names, health bars, attack
        // counters) by the visible fraction of a probe quad, so "1 sample" rendered them invisible.
        // Until real occlusion queries exist, report the probe as fully visible.
        case 0xF: value = getenv("HWDER_OCCLUSION_ONE") ? 1 : 0x100000; break;
        case 0x12: value = 0; break;        // primitives generated
        default: value = 0; break;
        }
        if (counter == 0) value = payload;
        renderer()->flush_commands();
        renderer()->on_gpu_idle([ch, addr, value, short_form] {
            if (short_form) {
                u32 v = (u32)value;
                ch->mm->write(addr, &v, 4);
            } else {
                struct {
                    u64 value;
                    u64 timestamp;
                } r{value, gpu_time_ns()};
                ch->mm->write(addr, &r, 16);
            }
        });
        break;
    }
    default:
        LOG_ONCE("3d: report trap");
        break;
    }
}

// ------------------------------------------------------------------ 3D engine
static void draw(Engine3D& e, bool indexed, u32 first, u32 count, u32 topology) {
    DrawParams p;
    p.indexed = indexed;
    p.topology = topology;
    p.first = first;
    p.count = count;
    p.base_vertex = e.regs[m3d::GlobalBaseVertexIndex];
    p.base_instance = e.regs[m3d::GlobalBaseInstanceIndex] + e.instance;
    p.instance = e.instance;
    p.instance_count = 1;
    if (trace_enabled()) hw_log("3d: draw %s first %u count %u topo %u inst %u", indexed ? "indexed" : "arrays", first, count, topology, p.base_instance);
    sync_cbufs(e);
    {
        OpTimer t("draw", count);
        renderer()->draw(e.regs, p);
    }
}

static void method_3d(Engine3D& e, u32 method, u32 value);

static void macro_send(void* ctx, u32 method, u32 value) { method_3d(*(Engine3D*)ctx, method, value); }
static u32 macro_read(void* ctx, u32 reg) { return reg < m3d::kNumRegs ? ((Engine3D*)ctx)->regs[reg] : 0; }

static void flush_macro(Engine3D& e) {
    if (e.macro_index < 0) return;
    u32 index = (u32)e.macro_index;
    e.macro_index = -1;
    e.mme.ctx_read_ = macro_read;
    std::vector<u32> params;
    params.swap(e.macro_params);
    e.mme.run(index, params, &e, macro_send);
}

static void method_3d(Engine3D& e, u32 method, u32 value) {
    using namespace m3d;
    {
        static const char* t = getenv("HWDER_GPU_TRACE");
        static u64 n = 0;
        if (t && t[0] == '2') {
            n++;
            if (n < 200000) hw_log("m3d %03x = %08x", method, value);  // HWDER_GPU_TRACE=2: raw 3D method stream
        }
    }
    if (method >= MacroStart) {  // macro call (even = start, odd = parameter)
        u32 index = (method - MacroStart) >> 1;
        if (!(method & 1) || e.macro_index < 0) {
            // A parameter write with no macro in flight also starts that macro (as on hardware).
            flush_macro(e);
            e.macro_index = (s32)index;
            e.macro_params.clear();
        }
        e.macro_params.push_back(value);
        return;
    }
    if (method >= kNumRegs) return;
    e.regs[method] = value;
    switch (method) {
    case LoadMmeInstructionRamPointer: e.mme.code_ptr = value; break;
    case LoadMmeInstructionRam: e.mme.code[e.mme.code_ptr++ & 0x1FFF] = value; break;
    case LoadMmeStartAddressRamPointer: e.mme.start_ptr = value; break;
    case LoadMmeStartAddressRam: e.mme.start[e.mme.start_ptr++ & 0x7F] = value; break;
    case I2mLaunchDma: i2m_launch(e.ch, e.ch->i2m3d, &e.regs[I2mLineLengthIn], value); break;
    case I2mLoadInlineData: i2m_data(e.ch, e.ch->i2m3d, value); break;
    case SyncInfo: {
        u32 id = value & 0xFFFF;
        bool cond = (value >> 20) & 1;  // 0: increment when all previous work is done
        (void)cond;
        renderer()->flush_commands();
        renderer()->on_gpu_idle([id] { nv::syncpoint_increment(id); });
        break;
    }
    case WaitForIdle:
        renderer()->flush_commands();
        break;
    case ConstBufferData + 0: case ConstBufferData + 1: case ConstBufferData + 2: case ConstBufferData + 3:
    case ConstBufferData + 4: case ConstBufferData + 5: case ConstBufferData + 6: case ConstBufferData + 7:
    case ConstBufferData + 8: case ConstBufferData + 9: case ConstBufferData + 10: case ConstBufferData + 11:
    case ConstBufferData + 12: case ConstBufferData + 13: case ConstBufferData + 14: case ConstBufferData + 15: {
        u64 base = addr(e.regs, ConstBuffer + 1);
        u32& pos = e.regs[ConstBuffer + 3];
        u8* p = e.mm->translate(base + pos);
        if (p) {
            memcpy(p, &value, 4);
            renderer()->invalidate_region((u64)p, 4);
        }
        pos += 4;
        break;
    }
    case DrawBegin: {
        u32 inst = (value >> 26) & 3;
        if (inst == 0) e.instance = 0;
        else if (inst == 1) e.instance++;
        e.inline_indices.clear();
        break;
    }
    case DrawEnd: {
        u32 topology = e.regs[DrawBegin] & 0xFFFF;
        if (!e.inline_indices.empty()) {
            DrawParams p;
            p.indexed = true;
            p.topology = topology;
            p.count = (u32)(e.inline_indices.size() / 4);
            p.base_vertex = e.regs[GlobalBaseVertexIndex];
            p.base_instance = e.regs[GlobalBaseInstanceIndex] + e.instance;
            p.instance = e.instance;
            sync_cbufs(e);
            OpTimer t("inline-index draw", p.count);
            renderer()->draw_inline_indices(e.regs, p, e.inline_indices.data(), 4);
            e.inline_indices.clear();
        } else if (e.indexed) {
            draw(e, true, e.regs[IndexBuffer + 5], e.regs[IndexBuffer + 6], topology);
        } else {
            draw(e, false, e.regs[VertexBufferFirst], e.regs[VertexBufferCount], topology);
        }
        break;
    }
    case VertexBufferCount: e.indexed = false; break;
    case IndexBuffer + 6: e.indexed = true; break;
    case DrawInlineIndex: {
        u8* p = (u8*)&value;
        e.inline_indices.insert(e.inline_indices.end(), p, p + 4);
        break;
    }
    case InlineIndex2x16: {
        u32 a = value & 0xFFFF, b = value >> 16;
        for (u32 v : {a, b}) {
            u8* p = (u8*)&v;
            e.inline_indices.insert(e.inline_indices.end(), p, p + 4);
        }
        break;
    }
    case InlineIndex4x8: {
        for (int i = 0; i < 4; i++) {
            u32 v = (value >> (i * 8)) & 0xFF;
            u8* p = (u8*)&v;
            e.inline_indices.insert(e.inline_indices.end(), p, p + 4);
        }
        break;
    }
    case VertexArrayInstanceFirst: case VertexArrayInstanceSubsequent: {
        if (method == VertexArrayInstanceFirst) e.instance = 0;
        else e.instance++;
        draw(e, false, value & 0xFFFF, (value >> 16) & 0xFFF, value >> 28);
        break;
    }
    case IndexBuffer32First: case IndexBuffer16First: case IndexBuffer8First:
    case IndexBuffer32Subsequent: case IndexBuffer16Subsequent: case IndexBuffer8Subsequent: {
        if (method <= IndexBuffer8First) e.instance = 0;
        else e.instance++;
        static const u32 fmt[] = {2, 1, 0};  // IndexFormat: 0 u8, 1 u16, 2 u32
        e.regs[IndexBuffer + 4] = fmt[(method - IndexBuffer32First) % 3];
        draw(e, true, value & 0xFFFF, (value >> 16) & 0xFFF, value >> 28);
        break;
    }
    case ClearSurface: {
        if (trace_enabled()) hw_log("3d: clear 0x%x", value);
        sync_cbufs(e);
        OpTimer t("clear", value);
        renderer()->clear(e.regs, value);
        break;
    }
    case ReportSemaphore + 3:
        report_semaphore(e, value);
        break;
    case BindGroups + 4: case BindGroups + 12: case BindGroups + 20: case BindGroups + 28: case BindGroups + 36:
    {
        u32 stage = (method - BindGroups) / 8, slot = (value >> 4) & 0x1F;
        if (slot < 18) e.cbufs[stage][slot] = {addr(e.regs, ConstBuffer + 1), e.regs[ConstBuffer], (value & 1) != 0};
        if (g_cbuf_owner == &e)
            renderer()->bind_const_buffer(stage, slot, addr(e.regs, ConstBuffer + 1), e.regs[ConstBuffer], value & 1);
    }
        break;
    default:
        break;
    }
}

// ------------------------------------------------------------------ compute / DMA / 2D
static void method_compute(Channel* ch, u32 method, u32 value) {
    EngineCompute& e = ch->compute;
    if (method < 0xE00) e.regs[method] = value;
    switch (method) {
    case 0x6C: i2m_launch(ch, ch->i2m_compute, &e.regs[0x60], value); break;
    case 0x6D: i2m_data(ch, ch->i2m_compute, value); break;
    case 0xAF: {  // LAUNCH: QMD at LAUNCH_DESC_LOC << 8
        u64 qmd = (u64)e.regs[0xAD] << 8;
        if (trace_enabled()) hw_log("compute: launch qmd 0x%llx", (unsigned long long)qmd);
        OpTimer t("dispatch", qmd);
        renderer()->dispatch(e.regs, qmd);
        break;
    }
    case 0xB2: {  // sync point increment
        u32 id = value & 0xFFFF;
        renderer()->flush_commands();
        renderer()->on_gpu_idle([id] { nv::syncpoint_increment(id); });
        break;
    }
    }
}

static void dma_copy(Channel* ch) {
    EngineDma& e = ch->dma;
    MemoryManager* mm = ch->mm;
    Renderer* rnd = renderer();
    u32 launch = e.regs[0xC0];
    u64 src = ((u64)e.regs[0x100] << 32) | e.regs[0x101];
    u64 dst = ((u64)e.regs[0x102] << 32) | e.regs[0x103];
    u32 pitch_in = e.regs[0x104], pitch_out = e.regs[0x105];
    u32 len = e.regs[0x106], lines = e.regs[0x107];
    bool multi_line = (launch >> 9) & 1;
    bool src_linear = (launch >> 7) & 1, dst_linear = (launch >> 8) & 1;
    bool remap = (launch >> 10) & 1;
    if (!multi_line) lines = 1;
    u32 bpp = 1;
    if (remap) {  // component remapping: each "pixel" has (num components) x (component size)
        u32 rc = e.regs[0x1C2];
        u32 comp_size = ((rc >> 16) & 3) + 1, num_src = ((rc >> 20) & 3) + 1;
        bpp = comp_size * num_src;
        u32 swz = rc & 0xFFF;
        bool const_fill = ((swz & 7) == 4 || (swz & 7) == 5);  // CONST_A / CONST_B into X
        if (const_fill && src_linear && dst_linear) {
            u32 cv = (swz & 7) == 4 ? e.regs[0x1C0] : e.regs[0x1C1];
            u64 total = (u64)len * bpp;
            for (u32 l = 0; l < lines; l++) {
                u64 row = dst + (u64)l * pitch_out;
                rnd->invalidate_region((u64)mm->translate(row), total);
                for (u64 x = 0; x < total; x += comp_size) mm->write(row + x, &cv, comp_size);
            }
            return;
        }
    }
    u64 row_bytes = (u64)len * bpp;
    if (src_linear && dst_linear) {
        if (!multi_line || (pitch_in == row_bytes && pitch_out == row_bytes)) {
            u64 total = multi_line ? row_bytes * lines : row_bytes;
            u8* s = mm->translate(src);
            u8* d = mm->translate(dst);
            rnd->flush_region((u64)s, total);
            rnd->invalidate_region((u64)d, total);
            if (s && d && mm->contiguous_size(src, total) == total && mm->contiguous_size(dst, total) == total) {
                memmove(d, s, total);
            } else {
                std::vector<u8> tmp(total);
                mm->read(src, tmp.data(), total);
                mm->write(dst, tmp.data(), total);
            }
        } else {
            std::vector<u8> tmp(row_bytes);
            for (u32 l = 0; l < lines; l++) {
                rnd->flush_region((u64)mm->translate(src + (u64)l * pitch_in), row_bytes);
                mm->read(src + (u64)l * pitch_in, tmp.data(), row_bytes);
                mm->write(dst + (u64)l * pitch_out, tmp.data(), row_bytes);
                rnd->invalidate_region((u64)mm->translate(dst + (u64)l * pitch_out), row_bytes);
            }
        }
        return;
    }
    auto block = [&](u32 base) {
        u32 bs = e.regs[base];
        BlockLinear bl{e.regs[base + 1] * (remap ? bpp : 1), e.regs[base + 2], e.regs[base + 3] ? e.regs[base + 3] : 1,
                       (bs >> 4) & 0xF, (bs >> 8) & 0xF};
        return bl;
    };
    if (!src_linear && dst_linear) {  // block-linear -> pitch
        BlockLinear bl = block(0x1CA);
        u32 origin = e.regs[0x1CF];
        u32 ox = (origin & 0xFFFF) * (remap ? bpp : 1), oy = origin >> 16;
        u32 layer = e.regs[0x1CE];
        u64 size = bl.size();
        u8* s = mm->translate(src);
        if (!s || mm->contiguous_size(src, size) < size) { LOG_ONCE("dma: non-contiguous block-linear source"); return; }
        rnd->flush_region((u64)s, size);
        std::vector<u8> tmp((size_t)pitch_out * lines);
        deswizzle(bl, s, tmp.data(), pitch_out, ox, oy, (u32)row_bytes, lines, layer);
        for (u32 l = 0; l < lines; l++) mm->write(dst + (u64)l * pitch_out, &tmp[(size_t)l * pitch_out], row_bytes);
        rnd->invalidate_region((u64)mm->translate(dst), (u64)pitch_out * lines);
        return;
    }
    if (src_linear && !dst_linear) {  // pitch -> block-linear
        BlockLinear bl = block(0x1C3);
        u32 origin = e.regs[0x1C8];
        u32 ox = (origin & 0xFFFF) * (remap ? bpp : 1), oy = origin >> 16;
        u32 layer = e.regs[0x1C7];
        u64 size = bl.size();
        u8* d = mm->translate(dst);
        if (!d || mm->contiguous_size(dst, size) < size) { LOG_ONCE("dma: non-contiguous block-linear destination"); return; }
        std::vector<u8> tmp((size_t)pitch_in * lines);
        rnd->flush_region((u64)mm->translate(src), (u64)pitch_in * lines);
        mm->read(src, tmp.data(), tmp.size());
        rnd->flush_region((u64)d, size);
        swizzle(bl, tmp.data(), d, pitch_in, ox, oy, (u32)row_bytes, lines, layer);
        rnd->invalidate_region((u64)d, size);
        return;
    }
    // block-linear -> block-linear: go through a linear temporary
    BlockLinear sbl = block(0x1CA), dbl = block(0x1C3);
    u32 so = e.regs[0x1CF], dor = e.regs[0x1C8];
    u8* s = mm->translate(src);
    u8* d = mm->translate(dst);
    if (!s || !d) return;
    rnd->flush_region((u64)s, sbl.size());
    std::vector<u8> tmp((size_t)row_bytes * lines);
    deswizzle(sbl, s, tmp.data(), (u32)row_bytes, (so & 0xFFFF) * (remap ? bpp : 1), so >> 16, (u32)row_bytes, lines, e.regs[0x1CE]);
    rnd->flush_region((u64)d, dbl.size());
    swizzle(dbl, tmp.data(), d, (u32)row_bytes, (dor & 0xFFFF) * (remap ? bpp : 1), dor >> 16, (u32)row_bytes, lines, e.regs[0x1C7]);
    rnd->invalidate_region((u64)d, dbl.size());
}

static void method_dma(Channel* ch, u32 method, u32 value) {
    EngineDma& e = ch->dma;
    if (method < 0x800) e.regs[method] = value;
    if (method == 0xC0) {  // LAUNCH_DMA
        if (trace_enabled()) hw_log("dma: launch 0x%x", value);
        if ((value & 3) != 0) {
            OpTimer t("dma copy", ((u64)e.regs[0x103] << 32) | e.regs[0x102]);
            dma_copy(ch);
        }
        u32 sem = (value >> 3) & 3;
        if (sem) {
            u64 a = ((u64)e.regs[0x90] << 32) | e.regs[0x91];
            u32 payload = e.regs[0x92];
            renderer()->flush_commands();
            renderer()->on_gpu_idle([ch, a, payload, sem] { write_report(ch, a, payload, sem == 1); });
        }
    }
}

static void method_2d(Channel* ch, u32 method, u32 value) {
    Engine2D& e = ch->e2d;
    if (method < 0x400) e.regs[method] = value;
    if (method == 0x237) {  // PIXELS_FROM_MEMORY_SRC_Y0 (int part) triggers the blit
        OpTimer t("2d blit");
        if (!renderer()->blit_2d(e.regs)) LOG_ONCE("2d: blit not handled by renderer");
    }
}

static void method_i2m(Channel* ch, u32 method, u32 value) {
    EngineI2M& e = ch->i2m;
    if (method < 0x80) e.regs[method] = value;
    if (method == 0x6C) i2m_launch(ch, e, &e.regs[0x60], value);
    else if (method == 0x6D) i2m_data(ch, e, value);
}

// ------------------------------------------------------------------ puller
static void call_method(Channel* ch, u32 subch, u32 method, u32 value) {
    static const char* tr = getenv("HWDER_GPU_TRACE");
    static const bool trace_host = tr && tr[0] == '3';  // host methods + class binds, per channel
    if (trace_host && (method < 0x40 || method == 0xB2 || method == 0xAF || method == 0x586))
        hw_log("gpu[sp%u]: subch %u class 0x%x method 0x%03x = 0x%08x", ch->syncpt, subch, ch->bound_class[subch], method, value);
    if (method < 0x40) {  // GPFIFO class (host) methods
        switch (method) {
        case 0x00:  // SET_OBJECT
            ch->bound_class[subch] = value & 0xFFFF;
            if (trace_enabled()) hw_log("gpu: subchannel %u = class 0x%x", subch, value & 0xFFFF);
            break;
        case 0x04: ch->sem_addr_hi = value; break;
        case 0x05: ch->sem_addr_lo = value; break;
        case 0x06: ch->sem_payload = value; break;
        case 0x07: {  // SEMAPHORED
            u64 a = ((u64)ch->sem_addr_hi << 32) | ch->sem_addr_lo;
            u32 op = value & 0x1F;
            if (op == 2) {  // release
                bool short_form = (value >> 12) & 1;
                u32 payload = ch->sem_payload;
                renderer()->flush_commands();
                renderer()->on_gpu_idle([ch, a, payload, short_form] { write_report(ch, a, payload, short_form); });
            } else if (op == 1 || op == 4 || op == 8) {  // acquire (equal / >= / and)
                renderer()->flush_commands();
                u64 t0 = GetTickCount64();
                bool logged = false;
                for (int spin = 0;; spin++) {
                    u32 v = 0;
                    ch->mm->read(a, &v, 4);
                    if ((op == 1 && v == ch->sem_payload) || (op == 4 && (s32)(v - ch->sem_payload) >= 0) ||
                        (op == 8 && (v & ch->sem_payload)))
                        break;
                    if (spin > 1000) std::this_thread::yield();
                    if (!logged && GetTickCount64() - t0 > 100) {
                        logged = true;
                        hw_log("gpu[sp%u]: semaphore acquire op %u at %llx waiting >100 ms (value %u, want %u)", ch->syncpt, op,
                               (unsigned long long)a, v, ch->sem_payload);
                    }
                }
            } else if (op == 0x10) {  // reduction
                u32 v = 0;
                ch->mm->read(a, &v, 4);
                v += ch->sem_payload;
                ch->mm->write(a, &v, 4);
            }
            break;
        }
        case 0x14: ch->reference = value; break;
        case 0x1C: ch->syncpt_payload = value; break;
        case 0x1D: {  // SYNCPOINTB: bit 0 operation (0 wait, 1 incr), bits 8-19 index
            u32 id = (value >> 8) & 0xFFF;
            if (value & 1) {
                renderer()->flush_commands();
                renderer()->on_gpu_idle([id] { nv::syncpoint_increment(id); });
            } else {
                renderer()->flush_commands();
                if (!nv::syncpoint_wait(id, ch->syncpt_payload, 100 * 1000000)) {
                    hw_log("gpu[sp%u]: pushbuffer syncpoint wait %u >= %u (value %u) >100 ms", ch->syncpt, id, ch->syncpt_payload,
                           nv::syncpoint_read(id));
                    nv::syncpoint_wait(id, ch->syncpt_payload, -1);
                }
            }
            break;
        }
        case 0x1E:  // WFI
            renderer()->flush_commands();
            break;
        default:
            break;
        }
        return;
    }
    switch (ch->bound_class[subch]) {
    case 0xB197: method_3d(ch->e3d, method, value); break;
    case 0xB1C0: method_compute(ch, method, value); break;
    case 0xB0B5: method_dma(ch, method, value); break;
    case 0x902D: method_2d(ch, method, value); break;
    case 0xA140: method_i2m(ch, method, value); break;
    default: LOG_ONCE("gpu: method 0x%x on unbound subchannel %u", method, subch); break;
    }
}

static void process_pushbuffer(Channel* ch, const u32* words, u32 count) {
    u32 i = 0;
    while (i < count) {
        u32 h = words[i++];
        u32 method = h & 0xFFF, subch = (h >> 13) & 7, n = (h >> 16) & 0x1FFF, op = h >> 29;
        switch (op) {
        case 1:  // incrementing
            for (u32 k = 0; k < n && i < count; k++) call_method(ch, subch, method + k, words[i++]);
            break;
        case 3:  // non-incrementing
            for (u32 k = 0; k < n && i < count; k++) call_method(ch, subch, method, words[i++]);
            break;
        case 4:  // immediate data in the header
            call_method(ch, subch, method, n);
            break;
        case 5:  // increment once
            for (u32 k = 0; k < n && i < count; k++) call_method(ch, subch, method + (k ? 1 : 0), words[i++]);
            break;
        case 0: {  // GRP0: tert op in bits 16-17 (0: legacy increment)
            if (h == 0) break;  // NOP
            u32 tert = (h >> 16) & 3;
            if (tert == 0) {
                n = (h >> 18) & 0x7FF;
                for (u32 k = 0; k < n && i < count; k++) call_method(ch, subch, method + k, words[i++]);
            }
            break;
        }
        case 2: {  // GRP2: legacy non-incrementing
            u32 tert = (h >> 16) & 3;
            if (tert == 0) {
                n = (h >> 18) & 0x7FF;
                for (u32 k = 0; k < n && i < count; k++) call_method(ch, subch, method, words[i++]);
            }
            break;
        }
        case 7:  // end of segment
            return;
        default:
            LOG_ONCE("gpu: pushbuffer sec_op %u", op);
            break;
        }
        // A macro call's parameters end with the command that carries them.
        if (ch->e3d.macro_index >= 0) flush_macro(ch->e3d);
    }
}

// ------------------------------------------------------------------ GPU thread
static struct {
    std::mutex m;
    std::condition_variable cv, drained;
    std::deque<std::pair<Channel*, Submission>> queue;
    std::thread thread;
} g_q;

static double g_busy_ms;
double gpu_busy_ms_reset() {
    double v = g_busy_ms;
    g_busy_ms = 0;
    return v;
}
static void execute_inner(Channel* ch, Submission& s);
static void execute(Channel* ch, Submission& s) {
    LARGE_INTEGER f, a, b;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&a);
    execute_inner(ch, s);
    QueryPerformanceCounter(&b);
    double ms = (b.QuadPart - a.QuadPart) * 1000.0 / f.QuadPart;
    g_busy_ms += ms;
    if (ms > 100.0)
        hw_log("gpu: submission on syncpt %u took %.0f ms (%zu entries)", s.syncpoint, ms, s.entries.size());
}
static void execute_inner(Channel* ch, Submission& s) {
    ch->syncpt = s.syncpoint;
    static const char* tr = getenv("HWDER_GPU_TRACE");
    if (tr && tr[0] == '3')
        hw_log("gpu[sp%u]: --- submission: %zu entries, wait %d, increment %d x%u", ch->syncpt, s.entries.size(), (int)s.wait,
               (int)s.increment, s.increment_count);
    if (s.wait) {
        // Bounded so a fence that can only be produced by later queued work cannot wedge the GPU thread.
        while (!nv::syncpoint_wait(s.wait_fence.id, s.wait_fence.value, 100 * 1000000)) {
            LOG_ONCE("gpu: submission waiting >100 ms for syncpt %u >= %u (value %u) - continuing", s.wait_fence.id,
                     s.wait_fence.value, nv::syncpoint_read(s.wait_fence.id));
            break;
        }
    }
    std::vector<u32> buf;
    for (const GpEntry& e : s.entries) {
        u32 n = e.words();
        if (!n) continue;  // control entry
        u64 va = e.address();
        const u32* words = (const u32*)ch->mm->translate(va);
        if (!words || ch->mm->contiguous_size(va, (u64)n * 4) < (u64)n * 4) {
            buf.resize(n);
            ch->mm->read(va, buf.data(), (u64)n * 4);
            words = buf.data();
        }
        OpTimer t("pushbuffer entry", n, 60.0);
        process_pushbuffer(ch, words, n);
    }
    renderer()->flush_commands();
    if (s.increment) {
        u32 id = s.syncpoint, count = s.increment_count;
        renderer()->on_gpu_idle([id, count] {
            for (u32 i = 0; i < count; i++) nv::syncpoint_increment(id);
        });
    }
}

static void gpu_thread() {
    SetThreadDescription(GetCurrentThread(), L"HWDER GPU");
    for (;;) {
        std::pair<Channel*, Submission> item;
        {
            std::unique_lock<std::mutex> l(g_q.m);
            g_q.cv.wait(l, [] { return !g_q.queue.empty(); });
            item = std::move(g_q.queue.front());
            g_q.queue.pop_front();
        }
        g_q.drained.notify_all();
        execute(item.first, item.second);
    }
}

Channel* create_channel(MemoryManager* mm) {
    static std::once_flag once;
    std::call_once(once, [] {
        renderer();
        g_q.thread = std::thread(gpu_thread);
        g_q.thread.detach();
    });
    auto* ch = new Channel;
    ch->mm = mm;
    ch->e3d.mm = mm;
    ch->e3d.ch = ch;
    return ch;
}

// Backpressure: a real GPFIFO is finite, so a guest that outruns the GPU blocks in SubmitGpfifo instead
// of letting its fences lag by seconds (its sync waits time out and it dereferences failed allocations).
constexpr size_t kMaxQueuedSubmissions = 24;

void channel_submit(Channel* ch, Submission&& s) {
    {
        std::unique_lock<std::mutex> l(g_q.m);
        if (g_q.queue.size() >= kMaxQueuedSubmissions) {
            static int logged = 0;
            if (logged++ < 5) hw_log("gpu: guest waiting for the GPU thread (%zu submissions queued)", g_q.queue.size());
            g_q.drained.wait(l, [] { return g_q.queue.size() < kMaxQueuedSubmissions; });
        }
        g_q.queue.emplace_back(ch, std::move(s));
    }
    g_q.cv.notify_one();
}

}  // namespace gpu
