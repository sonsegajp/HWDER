// ALU instruction semantics: float, half2, integer, conversions, predicates.
#include <cstring>

#include "shader/translator.h"

namespace shader {

// ------------------------------------------------------------------------------------- helpers

Form Translator::form_of(Op op) {
    static Form table[size_t(Op::INVALID) + 1];
    static bool init = false;
    if (!init) {
        for (size_t i = 0; i <= size_t(Op::INVALID); i++) {
            std::string n = op_name(Op(i));
            auto ends = [&](const char* s) {
                size_t l = strlen(s);
                return n.size() > l && n.compare(n.size() - l, l, s) == 0;
            };
            table[i] = ends("_rc") ? Form::RC : ends("_cr") ? Form::CR : ends("_r") ? Form::R
                     : ends("_c") ? Form::C : ends("_i") ? Form::I : Form::None;
        }
        init = true;
    }
    return table[size_t(op)];
}

Id Translator::cbuf_operand(Inst in) { return cbuf_load(in.bits(34, 5), in.bits(20, 14) * 4); }

Id Translator::src_b(Inst in, Form f) {
    switch (f) {
    case Form::R: return R(in.bits(20, 8));
    case Form::C:
    case Form::CR: return cbuf_operand(in);
    case Form::I: return u(imm20_u(in));
    case Form::RC: return R(in.bits(39, 8));
    default: return u(0);
    }
}
Id Translator::fsrc_b(Inst in, Form f) {
    if (f == Form::I) return bc_f(u(fimm20(in)));
    return bc_f(src_b(in, f));
}
Id Translator::src_c(Inst in, Form f) {
    if (f == Form::RC) return cbuf_operand(in);
    return R(in.bits(39, 8));
}

Id Translator::fop(spv::Op o, Id t, std::vector<u32> args) {
    Id r = b.op(o, t, std::move(args));
    if (!is_fs) b.decorate(r, spv::DecorationNoContraction);
    return r;
}

Id Translator::fabs_neg(Id v, bool abs, bool neg, Id t) {
    if (!t) t = t_f32;
    if (abs) v = glsl(t, GLSLstd450FAbs, {v});
    if (neg) v = b.op(spv::OpFNegate, t, {v});
    return v;
}

Id Translator::fsat(Id v, Id t) {
    if (!t || t == t_f32) return glsl(t_f32, GLSLstd450NClamp, {v, fc(0), fc(1)});
    Id z = b.c_composite(t, {fc(0), fc(0)}), o = b.c_composite(t, {fc(1), fc(1)});
    return glsl(t, GLSLstd450NClamp, {v, z, o});
}

Id Translator::bool_op(u32 bop, Id a, Id c) {
    switch (bop) {
    case 0: return land(a, c);
    case 1: return lor(a, c);
    case 2: return lxor(a, c);
    default: return a;
    }
}

Id Translator::fcompare(u32 cmp, Id a, Id c, Id bt) {
    if (!bt) bt = t_bool;
    auto konst = [&](bool v) {
        return bt == t_bool ? b.c_bool(v) : b.c_composite(bt, {b.c_bool(v), b.c_bool(v)});
    };
    auto nan_any = [&] {
        return b.op(spv::OpLogicalOr, bt, {b.op(spv::OpIsNan, bt, {a}), b.op(spv::OpIsNan, bt, {c})});
    };
    static const spv::Op ops[16] = {spv::OpNop, spv::OpFOrdLessThan, spv::OpFOrdEqual, spv::OpFOrdLessThanEqual,
                                    spv::OpFOrdGreaterThan, spv::OpFOrdNotEqual, spv::OpFOrdGreaterThanEqual,
                                    spv::OpNop, spv::OpNop, spv::OpFUnordLessThan, spv::OpFUnordEqual,
                                    spv::OpFUnordLessThanEqual, spv::OpFUnordGreaterThan, spv::OpFUnordNotEqual,
                                    spv::OpFUnordGreaterThanEqual, spv::OpNop};
    switch (cmp & 15) {
    case 0: return konst(false);
    case 15: return konst(true);
    case 7: return b.op(spv::OpLogicalNot, bt, {nan_any()});
    case 8: return nan_any();
    default: return b.op(ops[cmp & 15], bt, {a, c});
    }
}

Id Translator::icompare(u32 cmp, bool sg, Id a, Id c) {
    switch (cmp & 7) {
    case 0: return b.c_bool(false);
    case 1: return op2(sg ? spv::OpSLessThan : spv::OpULessThan, t_bool, a, c);
    case 2: return op2(spv::OpIEqual, t_bool, a, c);
    case 3: return op2(sg ? spv::OpSLessThanEqual : spv::OpULessThanEqual, t_bool, a, c);
    case 4: return op2(sg ? spv::OpSGreaterThan : spv::OpUGreaterThan, t_bool, a, c);
    case 5: return op2(spv::OpINotEqual, t_bool, a, c);
    case 6: return op2(sg ? spv::OpSGreaterThanEqual : spv::OpUGreaterThanEqual, t_bool, a, c);
    default: return b.c_bool(true);
    }
}

Id Translator::bfe(Id v, Id off, Id cnt, bool sg) {
    return b.op(sg ? spv::OpBitFieldSExtract : spv::OpBitFieldUExtract, t_u32, {v, off, cnt});
}

void Translator::unsupported(Op op) { log_once(std::string("unimplemented instruction ") + op_name(op) + " (no-op)"); }

// Sets Z/S from a 32-bit integer result and clears C/O.
static void cc_zs(Translator& t, Id res) {
    t.set_flag(0, t.op2(spv::OpIEqual, t.t_bool, res, t.u(0)));
    t.set_flag(1, t.op2(spv::OpSLessThan, t.t_bool, res, t.u(0)));
}
static void cc_float(Translator& t, Id f) {
    t.set_flag(0, t.op2(spv::OpFOrdEqual, t.t_bool, f, t.fc(0)));
    t.set_flag(1, t.op2(spv::OpFOrdLessThan, t.t_bool, f, t.fc(0)));
    t.set_flag(2, t.b.c_bool(false));
    t.set_flag(3, t.b.c_bool(false));
}

// ------------------------------------------------------------------------------------ dispatch

void Translator::emit_inst(Inst in, Op op) {
    switch (op) {
    case Op::NOP:
    case Op::DEPBAR:
    case Op::MEMBAR:
    case Op::CCTL:
    case Op::CCTLL:
    case Op::CCTLT:
    case Op::SSY:
    case Op::PBK:
    case Op::PCNT:
    case Op::SYNC:
    case Op::BPT:
    case Op::SAM:
    case Op::RAM:
    case Op::RTT:
    case Op::IDE:
    case Op::PLONGJMP:
    case Op::SETLMEMBASE:
    case Op::SETCRSPTR:
        return;
    case Op::FADD_r: case Op::FADD_c: case Op::FADD_i: case Op::FADD32I:
    case Op::FMUL_r: case Op::FMUL_c: case Op::FMUL_i: case Op::FMUL32I:
    case Op::FFMA_r: case Op::FFMA_rc: case Op::FFMA_cr: case Op::FFMA_i: case Op::FFMA32I:
    case Op::FMNMX_r: case Op::FMNMX_c: case Op::FMNMX_i:
    case Op::MUFU: case Op::RRO_r: case Op::RRO_c: case Op::RRO_i:
    case Op::FSETP_r: case Op::FSETP_c: case Op::FSETP_i:
    case Op::FSET_r: case Op::FSET_c: case Op::FSET_i:
    case Op::FCMP_r: case Op::FCMP_rc: case Op::FCMP_cr: case Op::FCMP_i:
    case Op::FSWZADD: case Op::FCHK_r: case Op::FCHK_c: case Op::FCHK_i:
        return emit_float(in, op);
    case Op::HADD2_r: case Op::HADD2_c: case Op::HADD2_i: case Op::HADD2_32I:
    case Op::HMUL2_r: case Op::HMUL2_c: case Op::HMUL2_i: case Op::HMUL2_32I:
    case Op::HFMA2_r: case Op::HFMA2_rc: case Op::HFMA2_cr: case Op::HFMA2_i: case Op::HFMA2_32I:
    case Op::HSET2_r: case Op::HSET2_c: case Op::HSET2_i:
    case Op::HSETP2_r: case Op::HSETP2_c: case Op::HSETP2_i:
        return emit_half(in, op);
    case Op::F2F_r: case Op::F2F_c: case Op::F2F_i:
    case Op::F2I_r: case Op::F2I_c: case Op::F2I_i:
    case Op::I2F_r: case Op::I2F_c: case Op::I2F_i:
    case Op::I2I_r: case Op::I2I_c: case Op::I2I_i:
        return emit_conv(in, op);
    case Op::PSETP: case Op::PSET: case Op::CSETP: case Op::CSET:
    case Op::P2R_r: case Op::P2R_c: case Op::P2R_i:
    case Op::R2P_r: case Op::R2P_c: case Op::R2P_i:
        return emit_pred(in, op);
    case Op::LDC: case Op::ALD: case Op::AST: case Op::IPA: case Op::S2R: case Op::CS2R:
    case Op::LDL: case Op::STL: case Op::LDS: case Op::STS: case Op::LD: case Op::ST:
    case Op::LDG: case Op::STG: case Op::ATOM: case Op::ATOMS: case Op::RED: case Op::ATOM_cas:
    case Op::ATOMS_cas: case Op::OUT_r: case Op::OUT_c: case Op::OUT_i: case Op::BAR: case Op::PIXLD:
    case Op::AL2P: case Op::ISBERD:
        return emit_mem(in, op);
    case Op::TEX: case Op::TEX_b: case Op::TEXS: case Op::TLD: case Op::TLD_b: case Op::TLDS:
    case Op::TLD4: case Op::TLD4_b: case Op::TLD4S: case Op::TXQ: case Op::TXQ_b: case Op::TMML:
    case Op::TMML_b: case Op::TXD: case Op::TXD_b: case Op::SULD: case Op::SUST: case Op::SURED:
    case Op::SUATOM: case Op::SUATOM_cas: case Op::TXA:
        return emit_tex(in, op);
    case Op::VOTE: case Op::VOTE_vtg: case Op::SHFL:
        return emit_warp(in, op);
    default:
        return emit_int(in, op);
    }
}

// --------------------------------------------------------------------------------------- float

void Translator::emit_float(Inst in, Op op) {
    Form f = form_of(op);
    u32 d = in.bits(0, 8);
    switch (op) {
    case Op::FADD_r: case Op::FADD_c: case Op::FADD_i: case Op::FADD32I: {
        bool is32 = op == Op::FADD32I;
        Id a = fabs_neg(F(in.bits(8, 8)), in.bit(is32 ? 54 : 46), in.bit(is32 ? 56 : 48));
        Id bb = is32 ? bc_f(u(in.bits(20, 32))) : fsrc_b(in, f);
        bb = fabs_neg(bb, in.bit(is32 ? 57 : 49), in.bit(is32 ? 53 : 45));
        Id r = fop(spv::OpFAdd, t_f32, {a, bb});
        if (!is32 && in.bit(50)) r = fsat(r);
        setF(d, r);
        if (in.bit(is32 ? 52 : 47)) cc_float(*this, r);
        return;
    }
    case Op::FMUL_r: case Op::FMUL_c: case Op::FMUL_i: case Op::FMUL32I: {
        bool is32 = op == Op::FMUL32I;
        Id a = F(in.bits(8, 8));
        Id bb = is32 ? bc_f(u(in.bits(20, 32))) : fsrc_b(in, f);
        if (!is32 && in.bit(48)) bb = b.op(spv::OpFNegate, t_f32, {bb});
        u32 fmz = in.bits(is32 ? 53 : 44, 2);
        bool sat = in.bit(is32 ? 55 : 50);
        if (!is32) {
            static const float scales[8] = {1, 0.5f, 0.25f, 0.125f, 8, 4, 2, 1};
            u32 sc = in.bits(41, 3);
            if (sc) a = fop(spv::OpFMul, t_f32, {a, fc(scales[sc])});
        }
        Id r = fop(spv::OpFMul, t_f32, {a, bb});
        if (fmz == 2 && !sat) {
            Id z = lor(op2(spv::OpFOrdEqual, t_bool, a, fc(0)), op2(spv::OpFOrdEqual, t_bool, bb, fc(0)));
            r = select(t_f32, z, fc(0), r);
        }
        if (sat) r = fsat(r);
        setF(d, r);
        if (in.bit(is32 ? 52 : 47)) cc_float(*this, r);
        return;
    }
    case Op::FFMA_r: case Op::FFMA_rc: case Op::FFMA_cr: case Op::FFMA_i: case Op::FFMA32I: {
        bool is32 = op == Op::FFMA32I;
        Id a = F(in.bits(8, 8));
        Id bb, c;
        if (is32) {
            bb = bc_f(u(in.bits(20, 32)));
            c = F(d);
            if (in.bit(56)) a = b.op(spv::OpFNegate, t_f32, {a});
            if (in.bit(57)) c = b.op(spv::OpFNegate, t_f32, {c});
        } else {
            bb = fsrc_b(in, f);
            c = bc_f(src_c(in, f));
            if (in.bit(48)) bb = b.op(spv::OpFNegate, t_f32, {bb});
            if (in.bit(49)) c = b.op(spv::OpFNegate, t_f32, {c});
        }
        u32 fmz = in.bits(53, 2);
        bool sat = in.bit(is32 ? 55 : 50);
        Id r = glsl(t_f32, GLSLstd450Fma, {a, bb, c});
        if (fmz == 2 && !sat) {
            Id z = lor(op2(spv::OpFOrdEqual, t_bool, a, fc(0)), op2(spv::OpFOrdEqual, t_bool, bb, fc(0)));
            r = select(t_f32, z, c, r);
        }
        if (sat) r = fsat(r);
        setF(d, r);
        if (in.bit(is32 ? 52 : 47)) cc_float(*this, r);
        return;
    }
    case Op::FMNMX_r: case Op::FMNMX_c: case Op::FMNMX_i: {
        Id a = fabs_neg(F(in.bits(8, 8)), in.bit(46), in.bit(48));
        Id bb = fabs_neg(fsrc_b(in, f), in.bit(49), in.bit(45));
        Id mn = glsl(t_f32, GLSLstd450NMin, {a, bb});
        Id mx = glsl(t_f32, GLSLstd450NMax, {a, bb});
        Id p = pred(in.bits(39, 3), in.bit(42));
        Id r = select(t_f32, p, mn, mx);
        setF(d, r);
        if (in.bit(47)) cc_float(*this, r);
        return;
    }
    case Op::MUFU: {
        Id a = fabs_neg(F(in.bits(8, 8)), in.bit(46), in.bit(48));
        Id r;
        switch (in.bits(20, 4)) {
        case 0: r = glsl(t_f32, GLSLstd450Cos, {a}); break;
        case 1: r = glsl(t_f32, GLSLstd450Sin, {a}); break;
        case 2: r = glsl(t_f32, GLSLstd450Exp2, {a}); break;
        case 3: r = glsl(t_f32, GLSLstd450Log2, {a}); break;
        case 4: r = b.op(spv::OpFDiv, t_f32, {fc(1), a}); break;
        case 5: r = glsl(t_f32, GLSLstd450InverseSqrt, {a}); break;
        case 8: r = glsl(t_f32, GLSLstd450Sqrt, {a}); break;
        case 6: {  // RCP64H: approximate on the high word of a double
            r = b.op(spv::OpFDiv, t_f32, {fc(1), a});
            break;
        }
        case 7: r = glsl(t_f32, GLSLstd450InverseSqrt, {a}); break;
        default:
            log_once("MUFU op " + std::to_string(in.bits(20, 4)));
            r = a;
        }
        if (in.bit(50)) r = fsat(r);
        setF(d, r);
        return;
    }
    case Op::RRO_r: case Op::RRO_c: case Op::RRO_i:
        setF(d, fabs_neg(fsrc_b(in, f), in.bit(49), in.bit(45)));
        return;
    case Op::FSETP_r: case Op::FSETP_c: case Op::FSETP_i: {
        Id a = fabs_neg(F(in.bits(8, 8)), in.bit(7), in.bit(43));
        Id bb = fabs_neg(fsrc_b(in, f), in.bit(44), in.bit(6));
        Id cmp = fcompare(in.bits(48, 4), a, bb);
        Id bp = pred(in.bits(39, 3), in.bit(42));
        u32 bop = in.bits(45, 2);
        set_pred(in.bits(3, 3), bool_op(bop, cmp, bp));
        set_pred(in.bits(0, 3), bool_op(bop, lnot(cmp), bp));
        return;
    }
    case Op::FSET_r: case Op::FSET_c: case Op::FSET_i: {
        Id a = fabs_neg(F(in.bits(8, 8)), in.bit(54), in.bit(43));
        Id bb = fabs_neg(fsrc_b(in, f), in.bit(44), in.bit(53));
        Id cmp = fcompare(in.bits(48, 4), a, bb);
        Id res = bool_op(in.bits(45, 2), cmp, pred(in.bits(39, 3), in.bit(42)));
        Id r = select(t_u32, res, u(in.bit(52) ? 0x3F800000u : 0xFFFFFFFFu), u(0));
        setR(d, r);
        if (in.bit(47)) cc_zs(*this, r);
        return;
    }
    case Op::FCMP_r: case Op::FCMP_rc: case Op::FCMP_cr: case Op::FCMP_i: {
        Id a = F(in.bits(8, 8));
        Id bb = fsrc_b(in, f);
        Id c = bc_f(src_c(in, f));
        Id cmp = fcompare(in.bits(48, 4), c, fc(0));
        setF(d, select(t_f32, cmp, a, bb));
        return;
    }
    case Op::FSWZADD: {
        Id a = F(in.bits(8, 8));
        Id bb = F(in.bits(20, 8));
        Id lane = op2(spv::OpBitwiseAnd, t_u32, lane_id(), u(3));
        Id sh = op2(spv::OpShiftLeftLogical, t_u32, lane, u(1));
        Id sel = op2(spv::OpBitwiseAnd, t_u32, op2(spv::OpShiftRightLogical, t_u32, u(in.bits(28, 8)), sh), u(3));
        Id lut_a = b.c_composite(t_v4f, {fc(-1), fc(1), fc(-1), fc(0)});
        Id lut_b = b.c_composite(t_v4f, {fc(-1), fc(-1), fc(1), fc(-1)});
        Id ma = b.op(spv::OpVectorExtractDynamic, t_f32, {lut_a, sel});
        Id mb = b.op(spv::OpVectorExtractDynamic, t_f32, {lut_b, sel});
        setF(d, fop(spv::OpFAdd, t_f32, {fop(spv::OpFMul, t_f32, {a, ma}), fop(spv::OpFMul, t_f32, {bb, mb})}));
        return;
    }
    case Op::FCHK_r: case Op::FCHK_c: case Op::FCHK_i:
        set_pred(in.bits(0, 3), b.c_bool(false));
        return;
    default:
        unsupported(op);
    }
}

// --------------------------------------------------------------------------------------- half2

namespace {
// swizzle: 0 H1_H0, 1 F32, 2 H0_H0, 3 H1_H1
Id unpack_h2(Translator& t, Id v, u32 swz) {
    if (swz == 1) {
        Id f = t.bc_f(v);
        return t.b.op(spv::OpCompositeConstruct, t.t_v2f, {f, f});
    }
    Id h = t.glsl(t.t_v2f, GLSLstd450UnpackHalf2x16, {v});
    if (swz == 2) return t.b.op(spv::OpVectorShuffle, t.t_v2f, {h, h, 0, 0});
    if (swz == 3) return t.b.op(spv::OpVectorShuffle, t.t_v2f, {h, h, 1, 1});
    return h;
}
Id h2_imm(Inst in) {
    return 0;
}
u32 h2_imm_bits(Inst in) {
    return (in.bits(20, 9) << 6) | (u32(in.bit(29)) << 15) | (in.bits(30, 9) << 22) | (u32(in.bit(56)) << 31);
}
// merge: 0 H1_H0, 1 F32, 2 MRG_H0, 3 MRG_H1
Id merge_h2(Translator& t, u32 dest, Id r, u32 merge) {
    switch (merge) {
    case 1: return t.bc_u(t.b.op(spv::OpCompositeExtract, t.t_f32, {r, 0}));
    case 2:
    case 3: {
        Id old = t.glsl(t.t_v2f, GLSLstd450UnpackHalf2x16, {t.R(dest)});
        Id v = merge == 2 ? t.b.op(spv::OpVectorShuffle, t.t_v2f, {r, old, 0, 3})
                          : t.b.op(spv::OpVectorShuffle, t.t_v2f, {old, r, 0, 3});
        return t.glsl(t.t_u32, GLSLstd450PackHalf2x16, {v});
    }
    default: return t.glsl(t.t_u32, GLSLstd450PackHalf2x16, {r});
    }
}
}  // namespace

void Translator::emit_half(Inst in, Op op) {
    u32 d = in.bits(0, 8);
    Id t2 = t_v2f;
    auto absneg2 = [&](Id v, bool abs, bool neg) { return fabs_neg(v, abs, neg, t2); };
    auto zero2 = [&] { return b.c_composite(t2, {fc(0), fc(0)}); };
    auto fmz_fix = [&](Id r, Id a, Id bb, Id repl) {
        Id za = b.op(spv::OpFOrdEqual, t_v2b, {a, zero2()});
        Id zb = b.op(spv::OpFOrdEqual, t_v2b, {bb, zero2()});
        return select(t2, b.op(spv::OpLogicalOr, t_v2b, {za, zb}), repl, r);
    };
    switch (op) {
    case Op::HADD2_r: case Op::HADD2_c: case Op::HADD2_i: case Op::HADD2_32I:
    case Op::HMUL2_r: case Op::HMUL2_c: case Op::HMUL2_i: case Op::HMUL2_32I: {
        bool add = op == Op::HADD2_r || op == Op::HADD2_c || op == Op::HADD2_i || op == Op::HADD2_32I;
        bool is32 = op == Op::HADD2_32I || op == Op::HMUL2_32I;
        u32 merge = is32 ? 0 : in.bits(49, 2);
        u32 swz_a = is32 ? in.bits(53, 2) : in.bits(47, 2);
        bool abs_a = false, neg_a = false, abs_b = false, neg_b = false, sat = false;
        u32 fmz = 0;
        Id bv;
        u32 swz_b = 0;
        if (op == Op::HADD2_r || op == Op::HMUL2_r) {
            bv = R(in.bits(20, 8));
            swz_b = in.bits(28, 2);
            sat = in.bit(32);
            neg_b = in.bit(31);
            abs_b = in.bit(30);
            abs_a = in.bit(44);
            neg_a = add && in.bit(43);
        } else if (op == Op::HADD2_c || op == Op::HMUL2_c) {
            bv = cbuf_operand(in);
            swz_b = 1;
            sat = in.bit(52);
            abs_b = in.bit(54);
            neg_b = add && in.bit(56);
            abs_a = in.bit(44);
            neg_a = in.bit(43);
        } else if (op == Op::HADD2_i || op == Op::HMUL2_i) {
            bv = u(h2_imm_bits(in));
            sat = in.bit(52);
            abs_a = in.bit(44);
            neg_a = in.bit(43);
        } else {
            bv = u(in.bits(20, 32));
            sat = in.bit(52);
            neg_a = add && in.bit(56);
        }
        if (!add && !is32) fmz = in.bits(39, 2);
        if (!add && is32) fmz = in.bits(55, 2);
        Id a = absneg2(unpack_h2(*this, R(in.bits(8, 8)), swz_a), abs_a, neg_a);
        Id bb = absneg2(unpack_h2(*this, bv, swz_b), abs_b, neg_b);
        Id r = b.op(add ? spv::OpFAdd : spv::OpFMul, t2, {a, bb});
        if (!add && fmz == 2 && !sat) r = fmz_fix(r, a, bb, zero2());
        if (sat) r = fsat(r, t2);
        setR(d, merge_h2(*this, d, r, merge));
        return;
    }
    case Op::HFMA2_r: case Op::HFMA2_rc: case Op::HFMA2_cr: case Op::HFMA2_i: case Op::HFMA2_32I: {
        u32 merge = 0, swz_a = 0, swz_b = 0, swz_c = 0, prec = 0;
        bool neg_b = false, neg_c = false, sat = false;
        Id bv, cv;
        if (op == Op::HFMA2_32I) {
            swz_a = in.bits(53, 2);
            neg_c = in.bit(52);
            prec = in.bits(55, 2);
            bv = u(in.bits(20, 32));
            cv = R(d);
        } else {
            merge = in.bits(49, 2);
            swz_a = in.bits(47, 2);
            if (op == Op::HFMA2_r) {
                swz_b = in.bits(28, 2);
                sat = in.bit(32);
                neg_b = in.bit(31);
                neg_c = in.bit(30);
                swz_c = in.bits(35, 2);
                prec = in.bits(37, 2);
                bv = R(in.bits(20, 8));
                cv = R(in.bits(39, 8));
            } else {
                neg_c = in.bit(51);
                sat = in.bit(52);
                neg_b = in.bit(56);
                prec = in.bits(57, 2);
                if (op == Op::HFMA2_rc) {
                    swz_b = in.bits(53, 2);
                    swz_c = 1;
                    bv = R(in.bits(39, 8));
                    cv = cbuf_operand(in);
                } else if (op == Op::HFMA2_cr) {
                    swz_b = 1;
                    swz_c = in.bits(53, 2);
                    bv = cbuf_operand(in);
                    cv = R(in.bits(39, 8));
                } else {
                    swz_b = 0;
                    swz_c = in.bits(53, 2);
                    neg_b = false;
                    bv = u(h2_imm_bits(in));
                    cv = R(in.bits(39, 8));
                }
            }
        }
        Id a = unpack_h2(*this, R(in.bits(8, 8)), swz_a);
        Id bb = absneg2(unpack_h2(*this, bv, swz_b), false, neg_b);
        Id c = absneg2(unpack_h2(*this, cv, swz_c), false, neg_c);
        Id r = glsl(t2, GLSLstd450Fma, {a, bb, c});
        if (prec == 2 && !sat) r = fmz_fix(r, a, bb, c);
        if (sat) r = fsat(r, t2);
        setR(d, merge_h2(*this, d, r, merge));
        return;
    }
    case Op::HSET2_r: case Op::HSET2_c: case Op::HSET2_i:
    case Op::HSETP2_r: case Op::HSETP2_c: case Op::HSETP2_i: {
        bool setp = op == Op::HSETP2_r || op == Op::HSETP2_c || op == Op::HSETP2_i;
        Id bv;
        u32 swz_b = 0, cmp;
        bool neg_b = false, abs_b = false, bf = false, h_and = false;
        if (op == Op::HSET2_r || op == Op::HSETP2_r) {
            bv = R(in.bits(20, 8));
            swz_b = in.bits(28, 2);
            abs_b = in.bit(30);
            neg_b = in.bit(31);
            cmp = in.bits(35, 4);
            bf = in.bit(49);
            h_and = in.bit(49);
        } else {
            bv = op == Op::HSET2_c || op == Op::HSETP2_c ? cbuf_operand(in) : u(h2_imm_bits(in));
            swz_b = op == Op::HSET2_c || op == Op::HSETP2_c ? 1 : 0;
            cmp = in.bits(49, 4);
            bf = in.bit(53);
            h_and = in.bit(53);
            if (op == Op::HSET2_c || op == Op::HSETP2_c) {
                neg_b = in.bit(56);
                abs_b = setp && in.bit(54);
            }
        }
        Id a = absneg2(unpack_h2(*this, R(in.bits(8, 8)), in.bits(47, 2)), in.bit(44), in.bit(43));
        Id bb = absneg2(unpack_h2(*this, bv, swz_b), abs_b, neg_b);
        Id c2 = fcompare(cmp, a, bb, t_v2b);
        Id lo = b.op(spv::OpCompositeExtract, t_bool, {c2, 0});
        Id hi = b.op(spv::OpCompositeExtract, t_bool, {c2, 1});
        Id p = pred(in.bits(39, 3), in.bit(42));
        u32 bop = in.bits(45, 2);
        lo = bool_op(bop, lo, p);
        hi = bool_op(bop, hi, p);
        if (setp) {
            if (h_and) {
                Id r = land(lo, hi);
                set_pred(in.bits(3, 3), r);
                set_pred(in.bits(0, 3), lnot(r));
            } else {
                set_pred(in.bits(3, 3), lo);
                set_pred(in.bits(0, 3), hi);
            }
        } else {
            u32 tv = bf ? 0x3C00 : 0xFFFF;
            Id r = op2(spv::OpBitwiseOr, t_u32, select(t_u32, lo, u(tv), u(0)), select(t_u32, hi, u(tv << 16), u(0)));
            setR(d, r);
        }
        return;
    }
    default:
        unsupported(op);
    }
}

// ------------------------------------------------------------------------------------- integer

void Translator::emit_int(Inst in, Op op) {
    Form f = form_of(op);
    u32 d = in.bits(0, 8);
    Id a_reg = 0;
    auto A = [&] {
        if (!a_reg) a_reg = R(in.bits(8, 8));
        return a_reg;
    };
    auto ineg = [&](Id v) { return b.op(spv::OpSNegate, t_u32, {v}); };
    switch (op) {
    case Op::MOV_r: case Op::MOV_c: case Op::MOV_i:
        setR(d, src_b(in, f));
        return;
    case Op::MOV32I:
        setR(d, u(in.bits(20, 32)));
        return;
    case Op::SEL_r: case Op::SEL_c: case Op::SEL_i:
        setR(d, select(t_u32, pred(in.bits(39, 3), in.bit(42)), A(), src_b(in, f)));
        return;
    case Op::IADD_r: case Op::IADD_c: case Op::IADD_i: case Op::IADD32I: {
        bool is32 = op == Op::IADD32I;
        Id a = A();
        Id bb = is32 ? u(in.bits(20, 32)) : src_b(in, f);
        bool na = in.bit(is32 ? 56 : 49), nb = !is32 && in.bit(48);
        bool po = is32 ? in.bits(55, 2) == 3 : (na && nb);
        bool x = in.bit(is32 ? 53 : 43), cc = in.bit(is32 ? 52 : 47);
        if (po) {
            bb = iadd(bb, u(1));
        } else {
            if (na) a = ineg(a);
            if (nb) bb = ineg(bb);
        }
        Id r = iadd(a, bb);
        if (x) r = iadd(r, select(t_u32, flag(2), u(1), u(0)));
        setR(d, r);
        if (cc) {
            cc_zs(*this, r);
            set_flag(2, op2(spv::OpULessThan, t_bool, r, a));
            Id sa = op2(spv::OpSLessThan, t_bool, a, u(0)), sb = op2(spv::OpSLessThan, t_bool, bb, u(0));
            Id sr = op2(spv::OpSLessThan, t_bool, r, u(0));
            set_flag(3, land(op2(spv::OpLogicalEqual, t_bool, sa, sb), lxor(sa, sr)));
        }
        return;
    }
    case Op::IADD3_r: case Op::IADD3_c: case Op::IADD3_i: {
        Id a = A(), bb = src_b(in, f), c = src_c(in, f);
        if (in.bit(51)) a = ineg(a);
        if (in.bit(50)) bb = ineg(bb);
        if (in.bit(49)) c = ineg(c);
        if (op == Op::IADD3_r) {
            auto half = [&](Id v, u32 h) {
                if (h == 1) return op2(spv::OpBitwiseAnd, t_u32, v, u(0xFFFF));
                if (h == 2) return op2(spv::OpShiftRightLogical, t_u32, v, u(16));
                return v;
            };
            a = half(a, in.bits(35, 2));
            bb = half(bb, in.bits(33, 2));
            c = half(c, in.bits(31, 2));
        }
        Id r = iadd(a, bb);
        if (op == Op::IADD3_r) {
            u32 mode = in.bits(37, 2);
            if (mode == 1) r = op2(spv::OpShiftRightLogical, t_u32, r, u(16));
            if (mode == 2) r = op2(spv::OpShiftLeftLogical, t_u32, r, u(16));
        }
        r = iadd(r, c);
        setR(d, r);
        if (in.bit(47)) cc_zs(*this, r);
        return;
    }
    case Op::ISCADD_r: case Op::ISCADD_c: case Op::ISCADD_i: case Op::ISCADD32I: {
        bool is32 = op == Op::ISCADD32I;
        Id a = A();
        Id bb = is32 ? u(in.bits(20, 32)) : src_b(in, f);
        u32 scale = is32 ? in.bits(53, 5) : in.bits(39, 5);
        bool na = !is32 && in.bit(49), nb = !is32 && in.bit(48);
        if (na && nb) {
            bb = iadd(bb, u(1));
        } else {
            if (na) a = ineg(a);
            if (nb) bb = ineg(bb);
        }
        Id r = iadd(op2(spv::OpShiftLeftLogical, t_u32, a, u(scale)), bb);
        setR(d, r);
        if (in.bit(is32 ? 52 : 47)) cc_zs(*this, r);
        return;
    }
    case Op::SHL_r: case Op::SHL_c: case Op::SHL_i: {
        Id s = src_b(in, f);
        Id r;
        if (in.bit(39)) {
            r = op2(spv::OpShiftLeftLogical, t_u32, A(), op2(spv::OpBitwiseAnd, t_u32, s, u(31)));
        } else {
            Id safe = op2(spv::OpULessThan, t_bool, s, u(32));
            r = select(t_u32, safe, op2(spv::OpShiftLeftLogical, t_u32, A(), op2(spv::OpBitwiseAnd, t_u32, s, u(31))), u(0));
        }
        setR(d, r);
        if (in.bit(47)) cc_zs(*this, r);
        return;
    }
    case Op::SHR_r: case Op::SHR_c: case Op::SHR_i: {
        Id s = src_b(in, f);
        bool sg = in.bit(48);
        Id a = A();
        if (in.bit(40)) a = b.op(spv::OpBitReverse, t_u32, {a});
        Id sm = op2(spv::OpBitwiseAnd, t_u32, s, u(31));
        Id r = op2(sg ? spv::OpShiftRightArithmetic : spv::OpShiftRightLogical, t_u32, a, sm);
        if (!in.bit(39)) {
            Id safe = op2(spv::OpULessThan, t_bool, s, u(32));
            Id over = sg ? op2(spv::OpShiftRightArithmetic, t_u32, a, u(31)) : u(0);
            r = select(t_u32, safe, r, over);
        }
        setR(d, r);
        if (in.bit(47)) cc_zs(*this, r);
        return;
    }
    case Op::SHF_l_r: case Op::SHF_l_i: case Op::SHF_r_r: case Op::SHF_r_i: {
        bool right = op == Op::SHF_r_r || op == Op::SHF_r_i;
        Id lo = A();
        Id hi = R(in.bits(39, 8));
        Id s = op == Op::SHF_l_i || op == Op::SHF_r_i ? u(imm20_u(in)) : R(in.bits(20, 8));
        bool wrap = in.bit(50);
        bool is64 = in.bits(37, 2) != 0;
        s = op2(spv::OpBitwiseAnd, t_u32, s, u(wrap ? (is64 ? 63 : 31) : 63));
        if (!wrap && !is64) s = glsl(t_u32, GLSLstd450UMin, {s, u(32)});
        // 64-bit funnel via two 32-bit halves: value = hi:lo
        Id s31 = op2(spv::OpBitwiseAnd, t_u32, s, u(31));
        Id ge32 = op2(spv::OpUGreaterThanEqual, t_bool, s, u(32));
        Id zero_s = op2(spv::OpIEqual, t_bool, s31, u(0));
        Id inv = op2(spv::OpISub, t_u32, u(32), s31);
        Id r;
        if (right) {
            Id x = op2(spv::OpBitwiseOr, t_u32, op2(spv::OpShiftRightLogical, t_u32, lo, s31),
                       op2(spv::OpShiftLeftLogical, t_u32, hi, op2(spv::OpBitwiseAnd, t_u32, inv, u(31))));
            x = select(t_u32, zero_s, lo, x);
            Id y = op2(in.bits(48, 1) ? spv::OpShiftRightArithmetic : spv::OpShiftRightLogical, t_u32, hi, s31);
            r = select(t_u32, ge32, y, x);
        } else {
            Id x = op2(spv::OpBitwiseOr, t_u32, op2(spv::OpShiftLeftLogical, t_u32, hi, s31),
                       op2(spv::OpShiftRightLogical, t_u32, lo, op2(spv::OpBitwiseAnd, t_u32, inv, u(31))));
            x = select(t_u32, zero_s, hi, x);
            Id y = op2(spv::OpShiftLeftLogical, t_u32, lo, s31);
            r = select(t_u32, ge32, y, x);
        }
        setR(d, r);
        return;
    }
    case Op::LOP_r: case Op::LOP_c: case Op::LOP_i: case Op::LOP32I: {
        bool is32 = op == Op::LOP32I;
        Id a = A();
        Id bb = is32 ? u(in.bits(20, 32)) : src_b(in, f);
        if (in.bit(is32 ? 55 : 39)) a = b.op(spv::OpNot, t_u32, {a});
        if (in.bit(is32 ? 56 : 40)) bb = b.op(spv::OpNot, t_u32, {bb});
        u32 lop = in.bits(is32 ? 53 : 41, 2);
        Id r = lop == 0 ? op2(spv::OpBitwiseAnd, t_u32, a, bb)
             : lop == 1 ? op2(spv::OpBitwiseOr, t_u32, a, bb)
             : lop == 2 ? op2(spv::OpBitwiseXor, t_u32, a, bb) : bb;
        if (!is32) {
            u32 pop = in.bits(44, 2);
            if (in.bits(48, 3) != 7) {
                Id pv = pop == 0 ? b.c_bool(false) : pop == 1 ? b.c_bool(true)
                      : pop == 2 ? op2(spv::OpIEqual, t_bool, r, u(0)) : op2(spv::OpINotEqual, t_bool, r, u(0));
                set_pred(in.bits(48, 3), pv);
            }
        }
        setR(d, r);
        if (in.bit(is32 ? 52 : 47)) {
            cc_zs(*this, r);
            set_flag(2, b.c_bool(false));
            set_flag(3, b.c_bool(false));
        }
        return;
    }
    case Op::LOP3_r: case Op::LOP3_c: case Op::LOP3_i: {
        Id a = A();
        Id bb = op == Op::LOP3_i ? u(imm20_u(in)) : src_b(in, op == Op::LOP3_c ? Form::C : Form::R);
        Id c = R(in.bits(39, 8));
        u32 lut = op == Op::LOP3_r ? in.bits(28, 8) : in.bits(48, 8);
        // Sum of products over the 8 minterms of (a, b, c).
        Id r = u(0);
        Id na = b.op(spv::OpNot, t_u32, {a}), nb = b.op(spv::OpNot, t_u32, {bb}), nc = b.op(spv::OpNot, t_u32, {c});
        if (lut == 0xFF) r = u(0xFFFFFFFF);
        else if (lut != 0) {
            Id acc = 0;
            for (u32 m = 0; m < 8; m++) {
                if (!((lut >> m) & 1)) continue;
                Id t1 = op2(spv::OpBitwiseAnd, t_u32, (m & 4) ? a : na, (m & 2) ? bb : nb);
                t1 = op2(spv::OpBitwiseAnd, t_u32, t1, (m & 1) ? c : nc);
                acc = acc ? op2(spv::OpBitwiseOr, t_u32, acc, t1) : t1;
            }
            r = acc;
        }
        if (op == Op::LOP3_r && in.bits(48, 3) != 7) {
            u32 pop = in.bits(36, 2);
            Id pv = pop == 0 ? b.c_bool(false) : pop == 1 ? b.c_bool(true)
                  : pop == 2 ? op2(spv::OpIEqual, t_bool, r, u(0)) : op2(spv::OpINotEqual, t_bool, r, u(0));
            set_pred(in.bits(48, 3), pv);
        }
        setR(d, r);
        if (in.bit(47)) cc_zs(*this, r);
        return;
    }
    case Op::XMAD_r: case Op::XMAD_rc: case Op::XMAD_cr: case Op::XMAD_i: {
        Id bv, cv;
        u32 mode, half_b = 0;
        bool psl = false, mrg = false;
        if (op == Op::XMAD_r) {
            bv = R(in.bits(20, 8));
            cv = R(in.bits(39, 8));
            half_b = in.bit(35);
            psl = in.bit(36);
            mrg = in.bit(37);
            mode = in.bits(50, 3);
        } else if (op == Op::XMAD_i) {
            bv = u(in.bits(20, 16));
            cv = R(in.bits(39, 8));
            psl = in.bit(36);
            mrg = in.bit(37);
            mode = in.bits(50, 3);
        } else if (op == Op::XMAD_rc) {
            bv = R(in.bits(39, 8));
            cv = cbuf_operand(in);
            mode = in.bits(50, 2);
            half_b = in.bit(52);
        } else {
            bv = cbuf_operand(in);
            cv = R(in.bits(39, 8));
            mode = in.bits(50, 2);
            half_b = in.bit(52);
            psl = in.bit(55);
            mrg = in.bit(56);
        }
        bool sa = in.bit(48), sb = in.bit(49);
        Id ha = bfe(A(), u(in.bit(53) ? 16 : 0), u(16), sa);
        Id hb = bfe(bv, u(half_b ? 16 : 0), u(16), sb);
        Id prod = op2(spv::OpIMul, t_u32, ha, hb);
        if (psl) prod = op2(spv::OpShiftLeftLogical, t_u32, prod, u(16));
        Id c = cv;
        switch (mode) {
        case 1: c = op2(spv::OpBitwiseAnd, t_u32, cv, u(0xFFFF)); break;
        case 2: c = op2(spv::OpShiftRightLogical, t_u32, cv, u(16)); break;
        case 4: c = iadd(op2(spv::OpShiftLeftLogical, t_u32, bv, u(16)), cv); break;
        case 3: log_once("XMAD CSFU"); break;
        default: break;
        }
        Id r = iadd(prod, c);
        if (mrg) r = b.op(spv::OpBitFieldInsert, t_u32, {r, bv, u(16), u(16)});
        setR(d, r);
        if (in.bit(47)) cc_zs(*this, r);
        return;
    }
    case Op::IMNMX_r: case Op::IMNMX_c: case Op::IMNMX_i: {
        bool sg = in.bit(48);
        Id a = A(), bb = src_b(in, f);
        Id mn = glsl(t_u32, sg ? GLSLstd450SMin : GLSLstd450UMin, {a, bb});
        Id mx = glsl(t_u32, sg ? GLSLstd450SMax : GLSLstd450UMax, {a, bb});
        Id r = select(t_u32, pred(in.bits(39, 3), in.bit(42)), mn, mx);
        setR(d, r);
        if (in.bit(47)) cc_zs(*this, r);
        return;
    }
    case Op::BFE_r: case Op::BFE_c: case Op::BFE_i: {
        Id s = src_b(in, f);
        Id base = A();
        if (in.bit(40)) base = b.op(spv::OpBitReverse, t_u32, {base});
        Id off = op2(spv::OpBitwiseAnd, t_u32, s, u(0xFF));
        Id cnt = bfe(s, u(8), u(8), false);
        // Clamp to the valid range of OpBitField*Extract.
        off = glsl(t_u32, GLSLstd450UMin, {off, u(32)});
        cnt = glsl(t_u32, GLSLstd450UMin, {cnt, op2(spv::OpISub, t_u32, u(32), off)});
        Id r = bfe(base, off, cnt, in.bit(48));
        r = select(t_u32, op2(spv::OpIEqual, t_bool, cnt, u(0)), u(0), r);
        setR(d, r);
        if (in.bit(47)) cc_zs(*this, r);
        return;
    }
    case Op::BFI_r: case Op::BFI_rc: case Op::BFI_cr: case Op::BFI_i: {
        Id s = src_b(in, f);
        Id base = src_c(in, f);
        Id ins = A();
        Id off = glsl(t_u32, GLSLstd450UMin, {op2(spv::OpBitwiseAnd, t_u32, s, u(0xFF)), u(32)});
        Id cnt = glsl(t_u32, GLSLstd450UMin, {bfe(s, u(8), u(8), false), op2(spv::OpISub, t_u32, u(32), off)});
        Id r = b.op(spv::OpBitFieldInsert, t_u32, {base, ins, off, cnt});
        setR(d, r);
        if (in.bit(47)) cc_zs(*this, r);
        return;
    }
    case Op::POPC_r: case Op::POPC_c: case Op::POPC_i: {
        Id v = src_b(in, f);
        if (in.bit(40)) v = b.op(spv::OpNot, t_u32, {v});
        setR(d, b.op(spv::OpBitCount, t_u32, {v}));
        return;
    }
    case Op::FLO_r: case Op::FLO_c: case Op::FLO_i: {
        Id v = src_b(in, f);
        if (in.bit(40)) v = b.op(spv::OpNot, t_u32, {v});
        Id r = glsl(t_u32, in.bit(48) ? GLSLstd450FindSMsb : GLSLstd450FindUMsb, {v});
        if (in.bit(41)) {
            Id nf = op2(spv::OpIEqual, t_bool, r, u(0xFFFFFFFF));
            r = select(t_u32, nf, r, op2(spv::OpBitwiseXor, t_u32, r, u(31)));
        }
        setR(d, r);
        return;
    }
    case Op::IMUL_r: case Op::IMUL_c: case Op::IMUL_i: case Op::IMUL32I: {
        bool is32 = op == Op::IMUL32I;
        Id bb = is32 ? u(in.bits(20, 32)) : src_b(in, f);
        bool hi = in.bit(is32 ? 53 : 39);
        bool sa = in.bit(is32 ? 54 : 40), sb = in.bit(is32 ? 55 : 41);
        Id r;
        if (hi) {
            Id st = b.type(spv::OpTypeStruct, {t_u32, t_u32});
            Id m = b.op(sa || sb ? spv::OpSMulExtended : spv::OpUMulExtended, st, {A(), bb});
            r = b.op(spv::OpCompositeExtract, t_u32, {m, 1});
        } else {
            r = op2(spv::OpIMul, t_u32, A(), bb);
        }
        setR(d, r);
        return;
    }
    case Op::IMAD_r: case Op::IMAD_rc: case Op::IMAD_cr: case Op::IMAD_i: case Op::IMAD32I: {
        bool is32 = op == Op::IMAD32I;
        Id bb = is32 ? u(in.bits(20, 32)) : src_b(in, f);
        Id c = is32 ? R(d) : src_c(in, f);
        Id r = iadd(op2(spv::OpIMul, t_u32, A(), bb), c);
        setR(d, r);
        return;
    }
    case Op::ICMP_r: case Op::ICMP_rc: case Op::ICMP_cr: case Op::ICMP_i: {
        Id bb = src_b(in, f), c = src_c(in, f);
        Id cmp = icompare(in.bits(49, 3), in.bit(48), c, u(0));
        setR(d, select(t_u32, cmp, A(), bb));
        return;
    }
    case Op::ISETP_r: case Op::ISETP_c: case Op::ISETP_i: {
        Id cmp = icompare(in.bits(49, 3), in.bit(48), A(), src_b(in, f));
        Id bp = pred(in.bits(39, 3), in.bit(42));
        u32 bop = in.bits(45, 2);
        set_pred(in.bits(3, 3), bool_op(bop, cmp, bp));
        set_pred(in.bits(0, 3), bool_op(bop, lnot(cmp), bp));
        return;
    }
    case Op::ISET_r: case Op::ISET_c: case Op::ISET_i: {
        Id cmp = icompare(in.bits(49, 3), in.bit(48), A(), src_b(in, f));
        Id res = bool_op(in.bits(45, 2), cmp, pred(in.bits(39, 3), in.bit(42)));
        Id r = select(t_u32, res, u(in.bit(44) ? 0x3F800000u : 0xFFFFFFFFu), u(0));
        setR(d, r);
        if (in.bit(47)) cc_zs(*this, r);
        return;
    }
    case Op::PRMT_r: case Op::PRMT_rc: case Op::PRMT_cr: case Op::PRMT_i: {
        Id sel = src_b(in, f);
        Id c = src_c(in, f);
        Id a = A();
        u32 mode = in.bits(48, 3);
        if (mode != 0) log_once("PRMT mode " + std::to_string(mode));
        Id r = u(0);
        for (u32 i = 0; i < 4; i++) {
            Id s = bfe(sel, u(i * 4), u(4), false);
            Id idx = op2(spv::OpBitwiseAnd, t_u32, s, u(7));
            Id src = select(t_u32, op2(spv::OpULessThan, t_bool, idx, u(4)), a, c);
            Id byte = bfe(src, op2(spv::OpIMul, t_u32, op2(spv::OpBitwiseAnd, t_u32, idx, u(3)), u(8)), u(8), false);
            Id sgn = op2(spv::OpINotEqual, t_bool, op2(spv::OpBitwiseAnd, t_u32, s, u(8)), u(0));
            Id rep = select(t_u32, op2(spv::OpINotEqual, t_bool, op2(spv::OpBitwiseAnd, t_u32, byte, u(0x80)), u(0)),
                            u(0xFF), u(0));
            byte = select(t_u32, sgn, rep, byte);
            r = op2(spv::OpBitwiseOr, t_u32, r, op2(spv::OpShiftLeftLogical, t_u32, byte, u(i * 8)));
        }
        setR(d, r);
        return;
    }
    case Op::LEA_lo_r: case Op::LEA_lo_c: case Op::LEA_lo_i: {
        Id a = A();
        if (in.bit(45)) a = ineg(a);
        Id r = iadd(op2(spv::OpShiftLeftLogical, t_u32, a, u(in.bits(39, 5))), src_b(in, f));
        setR(d, r);
        return;
    }
    case Op::LEA_hi_r: case Op::LEA_hi_c: {
        // hi: (a:c) >> (32 - scale) + b
        Id a = A();
        Id hi = op == Op::LEA_hi_r ? R(in.bits(39, 8)) : R(in.bits(39, 8));
        u32 scale = op == Op::LEA_hi_r ? in.bits(28, 5) : in.bits(51, 5);
        Id bb = op == Op::LEA_hi_r ? R(in.bits(20, 8)) : cbuf_operand(in);
        Id sh = scale == 0 ? hi
                           : op2(spv::OpBitwiseOr, t_u32, op2(spv::OpShiftRightLogical, t_u32, a, u(32 - scale)),
                                 op2(spv::OpShiftLeftLogical, t_u32, hi, u(scale)));
        setR(d, iadd(sh, bb));
        return;
    }
    case Op::LEPC:
        setR(d, u(cur_pc));
        return;
    case Op::GETCRSPTR:
    case Op::GETLMEMBASE:
        setR(d, u(0));
        return;
    case Op::VMNMX: case Op::VADD: case Op::VMAD: case Op::VSET: case Op::VSETP: case Op::VSHL:
    case Op::VSHR: case Op::VABSDIFF: case Op::VABSDIFF4: case Op::IDP_r: case Op::IDP_i:
    case Op::IMADSP_r: case Op::IMADSP_rc: case Op::IMADSP_cr: case Op::IMADSP_i:
    default:
        unsupported(op);
    }
}

// --------------------------------------------------------------------------------- conversions

void Translator::emit_conv(Inst in, Op op) {
    Form f = form_of(op);
    u32 d = in.bits(0, 8);
    switch (op) {
    case Op::F2F_r: case Op::F2F_c: case Op::F2F_i: {
        u32 dst = in.bits(8, 2), src = in.bits(10, 2);
        if (dst == 3 || src == 3) {
            unsupported(op);
            return;
        }
        Id v;
        if (src == 1) {
            Id h = glsl(t_v2f, GLSLstd450UnpackHalf2x16, {f == Form::I ? u(imm20_u(in)) : src_b(in, f)});
            v = b.op(spv::OpCompositeExtract, t_f32, {h, in.bit(41) ? 1u : 0u});
        } else {
            v = fsrc_b(in, f);
        }
        v = fabs_neg(v, in.bit(49), in.bit(45));
        if (src == dst) {
            switch (in.bits(39, 4) & 0xB) {
            case 8: v = glsl(t_f32, GLSLstd450RoundEven, {v}); break;
            case 9: v = glsl(t_f32, GLSLstd450Floor, {v}); break;
            case 10: v = glsl(t_f32, GLSLstd450Ceil, {v}); break;
            case 11: v = glsl(t_f32, GLSLstd450Trunc, {v}); break;
            default: break;
            }
        }
        if (in.bit(50)) v = fsat(v);
        if (dst == 1) setR(d, glsl(t_u32, GLSLstd450PackHalf2x16, {b.op(spv::OpCompositeConstruct, t_v2f, {v, fc(0)})}));
        else setF(d, v);
        return;
    }
    case Op::F2I_r: case Op::F2I_c: case Op::F2I_i: {
        u32 dfmt = in.bits(8, 2), sfmt = in.bits(10, 2);
        bool sg = in.bit(12);
        if (sfmt == 3 || dfmt == 3) {
            unsupported(op);
            return;
        }
        Id v;
        if (sfmt == 1) {
            Id h = glsl(t_v2f, GLSLstd450UnpackHalf2x16, {src_b(in, f)});
            v = b.op(spv::OpCompositeExtract, t_f32, {h, in.bit(41) ? 1u : 0u});
        } else {
            v = fsrc_b(in, f);
        }
        v = fabs_neg(v, in.bit(45), in.bit(49));
        switch (in.bits(39, 2)) {
        case 0: v = glsl(t_f32, GLSLstd450RoundEven, {v}); break;
        case 1: v = glsl(t_f32, GLSLstd450Floor, {v}); break;
        case 2: v = glsl(t_f32, GLSLstd450Ceil, {v}); break;
        case 3: v = glsl(t_f32, GLSLstd450Trunc, {v}); break;
        }
        Id nan = b.op(spv::OpIsNan, t_bool, {v});
        Id r;
        if (sg) {
            float lo = dfmt == 1 ? -32768.f : -2147483648.f, hi = dfmt == 1 ? 32767.f : 2147483520.f;
            Id c = glsl(t_f32, GLSLstd450FClamp, {v, fc(lo), fc(hi)});
            r = b.bitcast(t_u32, b.op(spv::OpConvertFToS, t_s32, {c}));
            if (dfmt != 1) {
                Id big = op2(spv::OpFOrdGreaterThanEqual, t_bool, v, fc(2147483648.f));
                r = select(t_u32, big, u(0x7FFFFFFF), r);
            }
        } else {
            float hi = dfmt == 1 ? 65535.f : 4294967040.f;
            Id c = glsl(t_f32, GLSLstd450FClamp, {v, fc(0), fc(hi)});
            r = b.op(spv::OpConvertFToU, t_u32, {c});
            if (dfmt != 1) {
                Id big = op2(spv::OpFOrdGreaterThanEqual, t_bool, v, fc(4294967296.f));
                r = select(t_u32, big, u(0xFFFFFFFF), r);
            }
        }
        r = select(t_u32, nan, u(0), r);
        setR(d, r);
        return;
    }
    case Op::I2F_r: case Op::I2F_c: case Op::I2F_i: {
        u32 ffmt = in.bits(8, 2), ifmt = in.bits(10, 2);
        bool sg = in.bit(13);
        if (ffmt == 3 || ifmt == 3) {
            unsupported(op);
            return;
        }
        Id v = src_b(in, f);
        u32 sel = in.bits(41, 2);
        if (ifmt == 0) v = bfe(v, u(sel * 8), u(8), sg);
        else if (ifmt == 1) v = bfe(v, u(sel * 8), u(16), sg);
        if (in.bit(49)) {
            if (sg) v = glsl(t_u32, GLSLstd450SAbs, {v});
        }
        if (in.bit(45)) v = b.op(spv::OpSNegate, t_u32, {v});
        Id r = sg ? b.op(spv::OpConvertSToF, t_f32, {b.bitcast(t_s32, v)}) : b.op(spv::OpConvertUToF, t_f32, {v});
        if (ffmt == 1) setR(d, glsl(t_u32, GLSLstd450PackHalf2x16, {b.op(spv::OpCompositeConstruct, t_v2f, {r, fc(0)})}));
        else setF(d, r);
        return;
    }
    case Op::I2I_r: case Op::I2I_c: case Op::I2I_i: {
        u32 dfmt = in.bits(8, 2), sfmt = in.bits(10, 2);
        bool dsg = in.bit(12), ssg = in.bit(13);
        Id v = src_b(in, f);
        u32 sel = in.bits(41, 2);
        if (sfmt == 0) v = bfe(v, u(sel * 8), u(8), ssg);
        else if (sfmt == 1) v = bfe(v, u(sel * 8), u(16), ssg);
        if (in.bit(49)) v = glsl(t_u32, GLSLstd450SAbs, {v});
        if (in.bit(45)) v = b.op(spv::OpSNegate, t_u32, {v});
        if (in.bit(50)) {
            // Saturate into the destination range.
            s32 lo = 0, hi = 0;
            u32 bits = dfmt == 0 ? 8 : dfmt == 1 ? 16 : 32;
            if (bits < 32) {
                if (dsg) {
                    lo = -(1 << (bits - 1));
                    hi = (1 << (bits - 1)) - 1;
                } else {
                    lo = 0;
                    hi = (1 << bits) - 1;
                }
                v = ssg ? glsl(t_u32, GLSLstd450SClamp, {v, u(u32(lo)), u(u32(hi))})
                        : glsl(t_u32, GLSLstd450UMin, {v, u(u32(hi))});
            }
        } else if (dfmt < 2) {
            v = bfe(v, u(0), u(dfmt == 0 ? 8 : 16), dsg);
        }
        setR(d, v);
        if (in.bit(47)) cc_zs(*this, v);
        return;
    }
    default:
        unsupported(op);
    }
}

// ---------------------------------------------------------------------------------- predicates

void Translator::emit_pred(Inst in, Op op) {
    switch (op) {
    case Op::PSETP:
    case Op::PSET: {
        Id pa = pred(in.bits(12, 3), in.bit(15));
        Id pb = pred(in.bits(29, 3), in.bit(32));
        Id pc = pred(in.bits(39, 3), in.bit(42));
        Id r1 = bool_op(in.bits(24, 2), pa, pb);
        u32 bop2 = in.bits(45, 2);
        if (op == Op::PSETP) {
            set_pred(in.bits(3, 3), bool_op(bop2, r1, pc));
            set_pred(in.bits(0, 3), bool_op(bop2, lnot(r1), pc));
        } else {
            Id r = select(t_u32, bool_op(bop2, r1, pc), u(in.bit(44) ? 0x3F800000u : 0xFFFFFFFFu), u(0));
            setR(in.bits(0, 8), r);
        }
        return;
    }
    case Op::CSETP:
    case Op::CSET: {
        Id t = flow_test(in.bits(8, 5));
        Id bp = pred(in.bits(39, 3), in.bit(42));
        u32 bop = in.bits(45, 2);
        if (op == Op::CSETP) {
            set_pred(in.bits(3, 3), bool_op(bop, t, bp));
            set_pred(in.bits(0, 3), bool_op(bop, lnot(t), bp));
        } else {
            setR(in.bits(0, 8), select(t_u32, bool_op(bop, t, bp), u(in.bit(44) ? 0x3F800000u : 0xFFFFFFFFu), u(0)));
        }
        return;
    }
    case Op::P2R_r: case Op::P2R_c: case Op::P2R_i: {
        u32 mask = op == Op::P2R_i ? imm20_u(in) : 0xFF;
        bool cc_mode = in.bit(40);
        u32 shift = in.bits(41, 2) * 8;
        Id ins = u(0);
        for (u32 i = 0; i < 8; i++) {
            if (!((mask >> i) & 1)) continue;
            Id c = cc_mode ? (i < 4 ? flag(int(i)) : b.c_bool(false)) : pred(i);
            ins = op2(spv::OpBitwiseOr, t_u32, ins, select(t_u32, c, u(1u << (i + shift)), u(0)));
        }
        Id keep = op2(spv::OpBitwiseAnd, t_u32, R(in.bits(8, 8)), u(~((mask & 0xFF) << shift)));
        setR(in.bits(0, 8), op2(spv::OpBitwiseOr, t_u32, keep, ins));
        return;
    }
    case Op::R2P_r: case Op::R2P_c: case Op::R2P_i: {
        Form f = form_of(op);
        u32 mask = op == Op::R2P_i ? imm20_u(in) : 0xFF;
        Id m = op == Op::R2P_i ? u(mask) : src_b(in, f);
        bool cc_mode = in.bit(40);
        u32 shift = in.bits(41, 2) * 8;
        Id src = R(in.bits(8, 8));
        for (u32 i = 0; i < (cc_mode ? 4u : 7u); i++) {
            Id bit = op2(spv::OpINotEqual, t_bool, bfe(src, u(i + shift), u(1), false), u(0));
            Id en = op2(spv::OpINotEqual, t_bool, bfe(m, u(i), u(1), false), u(0));
            Id old = cc_mode ? flag(int(i)) : pred(i);
            Id nv = select(t_bool, en, bit, old);
            if (cc_mode) set_flag(int(i), nv);
            else set_pred(i, nv);
        }
        return;
    }
    default:
        unsupported(op);
    }
}

}  // namespace shader
