// Internal state of the Maxwell -> SPIR-V translator.
//
// Model: the guest register file (R0-R254), predicates (P0-P6) and condition-code flags are SPIR-V
// Function variables (drivers promote them to SSA). Control flow is resolved statically: the CRS stack
// (SSY/PBK/PCNT/CAL tokens) is part of the CFG state, so SYNC/BRK/CONT/RET become direct jumps (code
// reached with different stacks is duplicated). A single-block program is emitted straight-line; anything
// else becomes a loop+switch dispatcher over basic blocks, which is always valid structured control flow.
#pragma once
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include "shader/decode.h"
#include "shader/internal.h"

namespace shader {

struct Unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};
void log_once(const std::string& msg);

struct StackEntry {
    u8 kind;  // 0 SSY, 1 PBK, 2 PCNT, 3 RET
    u32 target;
};
using Stack = std::vector<StackEntry>;

struct Block {
    u32 start;
    int sid;
    std::vector<u32> insts;
};

enum class Form { None, R, C, I, RC, CR };

class Translator {
public:
    Translator(Environment& env, Stage stage, const Options& opt);
    void run();

    Program prog;
    Environment& env;
    Stage stage;
    Options opt;
    SpvBuilder b;
    u8 sph[0x50] = {};
    bool is_vs = false, is_fs = false;

    // ---- CFG
    bool dispatcher = false;
    Id pc_var = 0, main_fn = 0;
    u32 cur_pc = 0;
    bool uses_discard = false;
    std::vector<Stack> stacks;
    std::map<std::vector<u64>, int> stack_ids;
    std::vector<Block> blocks;
    std::map<std::pair<u32, int>, int> block_ids;
    int intern_stack(const Stack& s);
    int flow(u32 pc, Inst in, Op op, Stack& st, u32& target, Stack& tstack);
    void build_cfg();
    int block_of(u32 pc, const Stack& st);
    void emit_block_body(const Block& blk);
    void set_next(int block);
    void if_then(Id cond, const std::function<void()>& body);
    Id guard_cond(Inst in, bool with_cc);
    void emit_predicated(Inst in, Op op);
    void emit_kill(Inst in);

    // ---- types
    Id t_void, t_bool, t_u32, t_s32, t_f32, t_v2f, t_v3f, t_v4f, t_v2u, t_v3u, t_v4u, t_v4s, t_v2b;
    Id ptr_fn_u32, ptr_fn_bool;

    // ---- register file
    Id regs[256] = {}, preds[8] = {}, flags[4] = {};  // flags: Z S C O
    Id R(u32 r);
    Id F(u32 r);
    void setR(u32 r, Id v);
    void setF(u32 r, Id v);
    Id pred(u32 p, bool neg = false);
    void set_pred(u32 p, Id v);
    Id flag(int i);
    void set_flag(int i, Id v);
    Id flow_test(u32 cc);

    // ---- value helpers
    Id u(u32 v) { return b.c_uint(v); }
    Id fc(float v) { return b.c_float(v); }
    Id bc_f(Id v) { return b.bitcast(t_f32, v); }
    Id bc_u(Id v) { return b.bitcast(t_u32, v); }
    Id op2(spv::Op o, Id t, Id a, Id c) { return b.op(o, t, {a, c}); }
    Id fop(spv::Op o, Id t, std::vector<u32> args);  // float arithmetic (NoContraction in pre-raster stages)
    Id glsl(Id t, u32 inst, std::vector<u32> args) { return b.ext(t, inst, args); }
    Id fabs_neg(Id v, bool abs, bool neg, Id t = 0);
    Id fsat(Id v, Id t = 0);
    Id select(Id t, Id c, Id a, Id z) { return b.op(spv::OpSelect, t, {c, a, z}); }
    Id lnot(Id v) { return b.op(spv::OpLogicalNot, t_bool, {v}); }
    Id land(Id a, Id c) { return op2(spv::OpLogicalAnd, t_bool, a, c); }
    Id lor(Id a, Id c) { return op2(spv::OpLogicalOr, t_bool, a, c); }
    Id lxor(Id a, Id c) { return op2(spv::OpLogicalNotEqual, t_bool, a, c); }
    Id bool_op(u32 bop, Id a, Id c);
    Id fcompare(u32 cmp, Id a, Id c, Id bt = 0);
    Id icompare(u32 cmp, bool is_signed, Id a, Id c);
    Id iadd(Id a, Id c) { return op2(spv::OpIAdd, t_u32, a, c); }
    Id bfe(Id v, Id off, Id cnt, bool is_signed);

    // ---- operand decoding
    Form form_of(Op op);
    u32 imm20_u(Inst in) { return u32(in.bits(20, 19)) | (in.bit(56) ? 0xFFF80000u : 0u); }
    u32 fimm20(Inst in) { return (in.bits(20, 19) << 12) | (u32(in.bit(56)) << 31); }
    Id cbuf_operand(Inst in);
    Id src_b(Inst in, Form f);      // 32-bit integer view of operand B
    Id fsrc_b(Inst in, Form f);     // float view of operand B
    Id src_c(Inst in, Form f);      // operand C (u32)

    // ---- instructions
    void emit_inst(Inst in, Op op);
    void emit_float(Inst in, Op op);
    void emit_half(Inst in, Op op);
    void emit_int(Inst in, Op op);
    void emit_conv(Inst in, Op op);
    void emit_pred(Inst in, Op op);
    void emit_mem(Inst in, Op op);
    void emit_tex(Inst in, Op op);
    void emit_warp(Inst in, Op op);
    void unsupported(Op op);

    // ---- program interface
    void begin_program();
    void prologue();
    void emit_epilogue();
    void finish_program();

    // constant buffers
    struct Cbuf {
        Id var = 0;
        u32 size = 0;
    };
    std::map<u32, Cbuf> cbufs;
    Id cbuf_struct = 0, cbuf_ptr_u32 = 0;
    Id cbuf_var(u32 index);
    Id cbuf_load(u32 index, u32 offset);
    Id cbuf_load_dyn(u32 index, Id byte_offset);

    // textures
    struct Tex {
        TextureType type;
        bool depth;
        u32 cbuf_index, cbuf_offset;
        Id var, image_t, sampled_t;
    };
    std::vector<Tex> textures;
    Tex& texture(u32 cbuf_offset, TextureType type, bool depth);

    // io
    Id in_generic[32] = {}, out_generic[32] = {}, in_generic_t[32] = {};
    u32 in_generic_kind[32] = {};  // 0 float 1 sint 2 uint
    bool in_centroid[32] = {};
    Id position = 0, frag_coord = 0, front_facing = 0, point_coord = 0, vertex_index = 0, instance_index = 0;
    Id point_size = 0, clip_distance = 0, layer_out = 0, viewport_out = 0, primitive_id = 0;
    Id frag_out[8] = {}, frag_depth = 0, sample_mask = 0, helper_inv = 0;
    Id push_var = 0, subgroup_id = 0, local_id = 0, workgroup_id = 0;
    std::vector<Id> io_vars;
    Id io_var(spv::StorageClass sc, Id type, int location, int builtin);
    Id input_generic(u32 index);
    Id output_generic(u32 index);
    Id push_member(u32 index, Id type);
    Id load_attribute(u32 addr);           // vertex/fragment input attribute word (u32 bits)
    void store_attribute(u32 addr, Id v);  // output attribute word
    Id ipa(Inst in);
    Id lane_id();

    // local / shared memory
    Id local_mem = 0, shared_mem = 0;
    u32 local_words = 0;
};

}  // namespace shader
