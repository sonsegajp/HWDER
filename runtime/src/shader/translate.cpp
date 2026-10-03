// Maxwell (SM 5.x) -> SPIR-V translator: entry points, CFG discovery and program assembly.
// Instruction semantics live in emit_*.cpp; shared state in translator.h.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>

#include "shader/translator.h"

namespace shader {

static u64 fnv(u64 h, const void* p, size_t n) {
    const u8* b = (const u8*)p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

static bool is_program_end(u64 w) {
    // BRA-to-self (or to the preceding EXIT) that terminates every program.
    return (w >> 52) == 0xE24 && ((w >> 20) & 0xFFFFFF) >= 0xFFFFE0 && (w & 0x1F) == 0xF;
}

u64 hash_program(Environment& env, Stage stage) {
    u64 h = 0xcbf29ce484222325ull ^ u64(stage);
    if (stage != Stage::Compute) h = fnv(h, env.sph(), 0x50);
    for (u32 addr = 0; addr < 0x100000; addr += 8) {
        u64 w = env.read_instruction(addr);
        h = fnv(h, &w, 8);
        if ((addr / 8) % 4 != 0 && is_program_end(w)) break;
    }
    return h;
}

void log_once(const std::string& msg) {
    static std::mutex m;
    static std::unordered_set<std::string> seen;
    std::lock_guard<std::mutex> lk(m);
    if (seen.insert(msg).second) hw_log("[shader] %s", msg.c_str());
}

// ------------------------------------------------------------------------------------------- CFG

static u32 next_pc(u32 pc) {
    pc += 8;
    if ((pc / 8) % 4 == 0) pc += 8;
    return pc;
}

int Translator::intern_stack(const Stack& s) {
    std::vector<u64> key;
    for (auto& e : s) key.push_back(u64(e.kind) << 32 | e.target);
    auto it = stack_ids.find(key);
    if (it != stack_ids.end()) return it->second;
    int id = int(stacks.size());
    stacks.push_back(s);
    stack_ids[key] = id;
    return id;
}

// Applies control-flow effects of one instruction. Returns successors through the out parameters.
// kind: 0 = continue linearly, 1 = jump (uncond), 2 = conditional (jump or fall through), 3 = terminate,
// 4 = conditional terminate (continue)
int Translator::flow(u32 pc, Inst in, Op op, Stack& st, u32& target, Stack& tstack) {
    u32 guard = in.bits(16, 3);
    bool gneg = in.bit(19);
    if (guard == 7 && gneg) return 0;  // never executes
    bool cond = guard != 7;
    // Relative targets are byte offsets that may land on a scheduling-control slot (every 4th 8-byte word);
    // the hardware then executes the instruction following it. Without the skip the CFG decoded the sched
    // word as INVALID and ended the program there (HW:DE's UI vertex shader 09461e7ed853c548 has 18 such
    // branches; on the greyed-out menu buttons' path the colour/texcoord outputs were never written).
    auto rel = [&] {
        u32 t = u32(s32(pc + 8) + in.sbits(20, 24));
        if ((t / 8) % 4 == 0) t += 8;
        return t;
    };
    auto pop_to = [&](u8 kind) -> bool {
        for (int i = int(st.size()) - 1; i >= 0; i--)
            if (st[i].kind == kind) {
                target = st[i].target;
                tstack.assign(st.begin(), st.begin() + i);
                return true;
            }
        return false;
    };
    switch (op) {
    case Op::SSY:
    case Op::PBK:
    case Op::PCNT:
        st.push_back({u8(op == Op::SSY ? 0 : op == Op::PBK ? 1 : 2), rel()});
        return 0;
    case Op::BRA: {
        if (in.bit(5)) throw Unsupported("BRA.CONST");
        target = rel();
        tstack = st;
        if (in.bits(0, 5) != 0xF) cond = true;
        return cond ? 2 : 1;
    }
    case Op::SYNC:
    case Op::BRK:
    case Op::CONT:
    case Op::RET: {
        u8 kind = op == Op::SYNC ? 0 : op == Op::BRK ? 1 : op == Op::CONT ? 2 : 3;
        if (!pop_to(kind)) {
            // Unbalanced token: treat as a no-op (SYNC) or program end.
            if (op == Op::SYNC) return 0;
            return 3;
        }
        if (in.bits(0, 5) != 0xF) cond = true;
        return cond ? 2 : 1;
    }
    case Op::CAL: {
        if (cond) throw Unsupported("predicated CAL");
        target = rel();
        tstack = st;
        tstack.push_back({3, next_pc(pc)});
        return 1;
    }
    case Op::EXIT:
        if (in.bits(0, 5) != 0xF) cond = true;
        return cond ? 4 : 3;
    case Op::KIL:
        if (in.bits(0, 5) != 0xF) cond = true;
        return cond ? 0 : 3;
    case Op::BRX:
    case Op::JMX:
    case Op::JMP:
    case Op::JCAL:
        throw Unsupported(op_name(op));
    case Op::INVALID:
        return 3;
    default:
        return 0;
    }
}

void Translator::build_cfg() {
    // Pass 1: discover leaders. Pass 2: cut blocks at leaders. States are (pc, CRS stack).
    std::set<u32> leaders{8};
    for (int pass = 0; pass < 2; pass++) {
        blocks.clear();
        block_ids.clear();
        std::vector<std::pair<u32, Stack>> work{{8, {}}};
        std::set<std::pair<u32, int>> visited;
        int steps = 0;
        while (!work.empty()) {
            auto [pc, st] = work.back();
            work.pop_back();
            int sid = intern_stack(st);
            if (!visited.insert({pc, sid}).second) continue;
            Block blk;
            blk.start = pc;
            blk.sid = sid;
            for (;;) {
                if (++steps > 400000 || blocks.size() > 30000) throw Unsupported("CFG too large");
                u64 raw = env.read_instruction(pc);
                Inst in{raw};
                Op op = decode(raw);
                if (is_program_end(raw)) op = Op::INVALID;
                blk.insts.push_back(pc);
                u32 target = 0;
                Stack tstack;
                int k = flow(pc, in, op, st, target, tstack);
                u32 nx = next_pc(pc);
                if (k == 1 || k == 2) {
                    leaders.insert(target);
                    work.push_back({target, tstack});
                    if (k == 2) {
                        leaders.insert(nx);
                        work.push_back({nx, st});
                    }
                    break;
                }
                if (k == 3) break;
                if (pass == 1 && leaders.count(nx)) {
                    work.push_back({nx, st});
                    break;
                }
                pc = nx;
            }
            if (pass == 1) {
                block_ids[{blk.start, blk.sid}] = int(blocks.size());
                blocks.push_back(std::move(blk));
            }
        }
    }
}

// ----------------------------------------------------------------------------------- translation

int Translator::block_of(u32 pc, const Stack& st) {
    int sid = intern_stack(st);
    auto it = block_ids.find({pc, sid});
    if (it == block_ids.end()) throw Unsupported("missing successor block");
    return it->second;
}

void Translator::emit_block_body(const Block& blk) {
    Stack st = stacks[blk.sid];
    for (size_t i = 0; i < blk.insts.size(); i++) {
        u32 pc = blk.insts[i];
        u64 raw = env.read_instruction(pc);
        Inst in{raw};
        Op op = decode(raw);
        if (is_program_end(raw)) op = Op::INVALID;
        cur_pc = pc;
        u32 target = 0;
        Stack tstack;
        Stack before = st;
        int k = flow(pc, in, op, st, target, tstack);
        u32 nx = next_pc(pc);
        bool last = i + 1 == blk.insts.size();
        switch (k) {
        case 0:
            if (op == Op::KIL) {
                emit_kill(in);
            } else if (op != Op::SSY && op != Op::PBK && op != Op::PCNT && op != Op::SYNC) {
                emit_predicated(in, op);
            }
            if (last) {
                if (dispatcher) set_next(block_of(nx, st));
                else hw_log("[shader] straight-line block fell off the end");
            }
            break;
        case 1:
            if (dispatcher) set_next(block_of(target, tstack));
            break;
        case 2: {
            Id c = guard_cond(in, op == Op::BRA || op == Op::SYNC || op == Op::BRK || op == Op::CONT || op == Op::RET);
            Id t = b.c_uint(u32(block_of(target, tstack)));
            Id f = b.c_uint(u32(block_of(nx, st)));
            b.store(pc_var, b.op(spv::OpSelect, t_u32, {c, t, f}));
            break;
        }
        case 3:
            if (op == Op::KIL) {
                uses_discard = true;
                b.terminate(spv::OpKill);
            } else {
                emit_epilogue();
                b.terminate(spv::OpReturn);
            }
            return;
        case 4: {
            Id c = guard_cond(in, true);
            if_then(c, [&] {
                emit_epilogue();
                b.terminate(spv::OpReturn);
            });
            if (last && dispatcher) set_next(block_of(nx, st));
            break;
        }
        }
        (void)before;
    }
}

void Translator::set_next(int block) { b.store(pc_var, b.c_uint(u32(block))); }

void Translator::if_then(Id cond, const std::function<void()>& body) {
    Id then_l = b.id(), merge_l = b.id();
    b.op0(spv::OpSelectionMerge, {merge_l, 0});
    b.terminate(spv::OpBranchConditional, {cond, then_l, merge_l});
    b.label(then_l);
    body();
    if (b.block_open) b.branch(merge_l);
    b.label(merge_l);
}

Id Translator::guard_cond(Inst in, bool with_cc) {
    Id c = 0;
    u32 g = in.bits(16, 3);
    if (g != 7 || in.bit(19)) c = pred(g, in.bit(19));
    if (with_cc && in.bits(0, 5) != 0xF) {
        Id t = flow_test(in.bits(0, 5));
        c = c ? b.op(spv::OpLogicalAnd, t_bool, {c, t}) : t;
    }
    return c ? c : b.c_bool(true);
}

void Translator::emit_predicated(Inst in, Op op) {
    u32 g = in.bits(16, 3);
    if (g == 7 && !in.bit(19)) {
        emit_inst(in, op);
        return;
    }
    if_then(pred(g, in.bit(19)), [&] { emit_inst(in, op); });
}

void Translator::emit_kill(Inst in) {
    // Debug: HWDER_SHADER_NO_KIL=1 turns every (conditional) KIL into a no-op (alpha-test diagnosis).
    static const bool no_kill = getenv("HWDER_SHADER_NO_KIL") != nullptr;
    if (no_kill) return;
    uses_discard = true;
    Id c = guard_cond(in, true);
    if_then(c, [&] { b.terminate(spv::OpKill); });
}

void Translator::run() {
    build_cfg();
    dispatcher = blocks.size() > 1;
    begin_program();
    Id fn_t = b.t_func(t_void);
    main_fn = b.begin_function(t_void, fn_t);
    prologue();
    if (!dispatcher) {
        emit_block_body(blocks[0]);
        if (b.block_open) {
            emit_epilogue();
            b.terminate(spv::OpReturn);
        }
    } else {
        pc_var = b.local_var(b.t_ptr(spv::StorageClassFunction, t_u32), b.c_uint(0));
        Id header = b.id(), body = b.id(), cont = b.id(), merge = b.id(), sw_merge = b.id();
        b.branch(header);
        b.label(header);
        b.op0(spv::OpLoopMerge, {merge, cont, 0});
        b.branch(body);
        b.label(body);
        Id v = b.load(t_u32, pc_var);
        std::vector<Id> labels(blocks.size());
        std::vector<u32> sw{v, sw_merge};
        for (size_t i = 0; i < blocks.size(); i++) {
            labels[i] = b.id();
            sw.push_back(u32(i));
            sw.push_back(labels[i]);
        }
        b.op0(spv::OpSelectionMerge, {sw_merge, 0});
        b.terminate(spv::OpSwitch, sw);
        for (size_t i = 0; i < blocks.size(); i++) {
            b.label(labels[i]);
            emit_block_body(blocks[i]);
            if (b.block_open) b.branch(sw_merge);
        }
        b.label(sw_merge);
        b.branch(cont);
        b.label(cont);
        b.branch(header);
        b.label(merge);
        b.terminate(spv::OpUnreachable);
    }
    b.end_function();
    finish_program();
}

std::string listing(Environment& env, Stage stage) {
    std::string out;
    char lbuf[160];
    for (u32 addr = 0; addr < 0x20000; addr += 8) {
        u64 w = env.read_instruction(addr);
        if ((addr / 8) % 4 == 0) {
            snprintf(lbuf, sizeof(lbuf), "%05x: %016llx  (sched)\n", addr, (unsigned long long)w);
            out += lbuf;
            continue;
        }
        Inst ins{w};
        Op op = decode(w);
        snprintf(lbuf, sizeof(lbuf), "%05x: %016llx  %-10s d=R%-3u a=R%-3u b=R%-3u c=R%-3u pred=%u%s imm20=%06llx\n", addr,
                 (unsigned long long)w, op_name(op), ins.bits(0, 8), ins.bits(8, 8), ins.bits(20, 8), ins.bits(39, 8), ins.bits(16, 3),
                 ins.bit(19) ? "!" : "", (unsigned long long)ins.bits(20, 20));
        out += lbuf;
        if (is_program_end(w)) break;
    }
    return out;
}

Program translate(Environment& env, Stage stage, const Options& opt, Environment* vertex_a) {
    if (vertex_a) log_once("VertexA programs are not supported; translating VertexB only");
    // Debug: HWDER_SHADER_FALLBACK=all replaces every stage, =frag only fragment shaders (solid magenta).
    static const char* force = getenv("HWDER_SHADER_FALLBACK");
    if (force && (!strcmp(force, "all") || (stage == Stage::Fragment && !strcmp(force, "frag"))))
        return make_fallback(stage);
    try {
        Translator t(env, stage, opt);
        t.run();
        return std::move(t.prog);
    } catch (const Unsupported& e) {
        log_once(std::string("translation failed, using pass-through: ") + e.what());
    } catch (const std::exception& e) {
        log_once(std::string("translation error, using pass-through: ") + e.what());
    } catch (...) {
        log_once("translation error, using pass-through");
    }
    Program p = make_fallback(stage);
    return p;
}

}  // namespace shader
