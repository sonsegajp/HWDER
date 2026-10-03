"""AArch64 SIMD & floating point -> C translation (see a64.py for conventions).

Vector registers are `V128` unions in c->v[]. Lane views:
  b/h/s/d (unsigned 8/16/32/64), sb/sh/ss/sd (signed), f (float), fd (double).
Writes of 64-bit vectors (Q=0) and of scalars zero the remaining upper bits.
"""
import struct

from .a64 import Unimplemented, bits, bit, sext, mask, hx, cond_expr, rd, wr, LD_FN, ST_FN

U = {8: "b", 16: "h", 32: "s", 64: "d"}
S = {8: "sb", 16: "sh", 32: "ss", 64: "sd"}
UT = {8: "u8", 16: "u16", 32: "u32", 64: "u64"}
STY = {8: "s8", 16: "s16", 32: "s32", 64: "s64"}
F = {32: "f", 64: "fd"}
FT = {32: "float", 64: "double"}
FSUF = {32: "F", 64: "D"}


def fconst(val, w):
    if val != val:
        raise Unimplemented("nan const")
    return f"({FT[w]}){float(val).hex()}"


def vfp_expand_imm(imm8, w):
    sign = imm8 >> 7
    b6 = (imm8 >> 6) & 1
    if w == 32:
        e, fbits = 8, 23
    else:
        e, fbits = 11, 52
    exp = ((b6 ^ 1) << (e - 1)) | ((mask(e - 3) if b6 else 0) << 2) | ((imm8 >> 4) & 3)
    frac = (imm8 & 0xF) << (fbits - 4)
    v = (sign << (e + fbits)) | (exp << fbits) | frac
    if w == 32:
        return struct.unpack("<f", struct.pack("<I", v))[0]
    return struct.unpack("<d", struct.pack("<Q", v))[0]


def adv_simd_expand_imm(op, cmode, imm8):
    c3 = cmode >> 1
    if c3 == 0:
        return imm8 * 0x0000000100000001
    if c3 == 1:
        return (imm8 << 8) * 0x0000000100000001
    if c3 == 2:
        return (imm8 << 16) * 0x0000000100000001
    if c3 == 3:
        return (imm8 << 24) * 0x0000000100000001
    if c3 == 4:
        return imm8 * 0x0001000100010001
    if c3 == 5:
        return (imm8 << 8) * 0x0001000100010001
    if c3 == 6:
        if cmode & 1 == 0:
            return ((imm8 << 8) | 0xFF) * 0x0000000100000001
        return ((imm8 << 16) | 0xFFFF) * 0x0000000100000001
    if cmode & 1 == 0 and op == 0:
        return imm8 * 0x0101010101010101
    if cmode & 1 == 0 and op == 1:
        v = 0
        for i in range(8):
            if imm8 & (1 << i):
                v |= 0xFF << (8 * i)
        return v
    if cmode & 1 == 1 and op == 0:
        f = struct.unpack("<I", struct.pack("<f", vfp_expand_imm(imm8, 32)))[0]
        return f * 0x0000000100000001
    return struct.unpack("<Q", struct.pack("<d", vfp_expand_imm(imm8, 64)))[0]


def vec_op(d, n_elems, lane_out, expr, srcs, keep_d=False, q=1):
    """Emit a lane-wise vector op. srcs: dict name->register number; expr uses name_[i]."""
    decl = ", ".join(f"{k}_ = c->v[{v}]" for k, v in srcs.items())
    init = "c->v[" + str(d) + "]" if keep_d else "V_ZERO"
    if keep_d and not q:
        init = f"V_LO(c->v[{d}])"
    return (f"{{ V128 {decl + ', ' if decl else ''}r_ = {init}; "
            f"for (int i = 0; i < {n_elems}; i++) r_.{lane_out}[i] = {expr}; c->v[{d}] = r_; }}")


def set_scalar(d, w, val, lane=None):
    lane = lane or {8: "b", 16: "h", 32: "f", 64: "fd"}[w]
    return f"{{ {lane_type(lane)} t_ = {val}; c->v[{d}] = V_ZERO; c->v[{d}].{lane}[0] = t_; }}"


def lane_type(lane):
    return {"b": "u8", "h": "u16", "s": "u32", "d": "u64", "sb": "s8", "sh": "s16", "ss": "s32",
            "sd": "s64", "f": "float", "fd": "double"}[lane]


def crypto(w):
    """ARMv8 AES / SHA1 / SHA256 instructions, or None if w is not one."""
    rn, rdn, rm = bits(w, 9, 5), bits(w, 4, 0), bits(w, 20, 16)
    d, n, m = f"c->v[{rdn}]", f"c->v[{rn}]", f"c->v[{rm}]"
    if w & 0xFFFFCC00 == 0x4E284800:  # AESE / AESD / AESMC / AESIMC
        op = bits(w, 13, 12)
        e = [f"AESE({d}, {n})", f"AESD({d}, {n})", f"AESMC({n})", f"AESIMC({n})"][op]
        return f"{d} = {e};"
    if w & 0xFFE08C00 == 0x5E000000:  # SHA three-register
        op = bits(w, 14, 12)
        e = {0: f"SHA1HASH({d}, {n}.s[0], {m}, 0)", 1: f"SHA1HASH({d}, {n}.s[0], {m}, 1)",
             2: f"SHA1HASH({d}, {n}.s[0], {m}, 2)", 3: f"SHA1SU0({d}, {n}, {m})",
             4: f"SHA256HASH({d}, {n}, {m}, 0)", 5: f"SHA256HASH({n}, {d}, {m}, 1)",
             6: f"SHA256SU1({d}, {n}, {m})"}.get(op)
        if e is None:
            raise Unimplemented("sha3 op")
        return f"{{ V128 r_ = {e}; {d} = r_; }}"
    if w & 0xFFFFCC00 == 0x5E280800:  # SHA two-register
        op = bits(w, 16, 12)
        if op == 0:  # SHA1H: Sd = ROL(Sn, 30)
            return set_scalar(rdn, 32, f"HW_ROL32({n}.s[0], 30)", lane="s")
        if op == 1:
            return f"{{ V128 r_ = SHA1SU1({d}, {n}); {d} = r_; }}"
        if op == 2:
            return f"{{ V128 r_ = SHA256SU0({d}, {n}); {d} = r_; }}"
        raise Unimplemented("sha2 op")
    return None


def translate(w, pc):
    cr = crypto(w)
    if cr is not None:
        return cr
    if w & 0x5F000000 == 0x1E000000:
        # bit 31 is sf for FP<->integer conversions, must be 0 otherwise
        if bit(w, 31) and not (bit(w, 21) == 0 or bits(w, 15, 10) == 0):
            raise Unimplemented("scalar fp M=1")
        return scalar_fp(w)
    if w & 0x5F000000 == 0x1F000000 and bit(w, 31) == 0:
        return fp_3src(w)
    if w & 0x9F000000 == 0x0E000000 or w & 0x9F000000 == 0x0F000000:
        return adv_simd_vector(w)
    if w & 0xDF000000 == 0x5E000000 or w & 0xDF000000 == 0x5F000000:
        return adv_simd_scalar(w)
    if w & 0xBF000000 == 0x0C000000 or w & 0xBF000000 == 0x0D000000:
        return ld_struct(w)
    raise Unimplemented("simd group")


# ====================================================================== scalar FP

def ftype_width(t):
    if t == 0:
        return 32
    if t == 1:
        return 64
    if t == 3:
        return 16
    raise Unimplemented("ftype")


def fr(n, w):
    if w == 16:
        return f"c->v[{n}].h[0]"
    return f"c->v[{n}].{F[w]}[0]"


def scalar_fp(w):
    ftype = bits(w, 23, 22)
    rn, rdn = bits(w, 9, 5), bits(w, 4, 0)
    rm = bits(w, 20, 16)
    if bit(w, 21) == 0:
        return fp_fixed_conv(w)
    op15_10 = bits(w, 15, 10)
    if op15_10 == 0:
        return fp_int_conv(w)
    fw = ftype_width(ftype)
    if bits(w, 14, 10) == 0b10000:  # 1-source
        opcode = bits(w, 20, 15)
        if fw == 16:
            if opcode == 0b000100:
                return set_scalar(rdn, 32, f"H2F(c->v[{rn}].h[0])")
            if opcode == 0b000101:
                return set_scalar(rdn, 64, f"(double)H2F(c->v[{rn}].h[0])")
            raise Unimplemented("half 1src")
        a = fr(rn, fw)
        fn = {0: f"{a}", 1: f"FABS{FSUF[fw]}({a})", 2: f"-{a}", 3: f"FSQRT{FSUF[fw]}({a})",
              8: f"FRINTN{FSUF[fw]}({a})", 9: f"FRINTP{FSUF[fw]}({a})", 10: f"FRINTM{FSUF[fw]}({a})",
              11: f"FRINTZ{FSUF[fw]}({a})", 12: f"FRINTA{FSUF[fw]}({a})", 14: f"FRINTX{FSUF[fw]}({a})",
              15: f"FRINTX{FSUF[fw]}({a})"}
        if opcode in fn:
            return set_scalar(rdn, fw, fn[opcode])
        if opcode == 4:  # to single
            return set_scalar(rdn, 32, f"(float){a}")
        if opcode == 5:
            return set_scalar(rdn, 64, f"(double){a}")
        if opcode == 7:
            return set_scalar(rdn, 16, f"F2H((float){a})")
        raise Unimplemented(f"fp 1src {opcode}")
    if bits(w, 13, 10) == 0b1000:  # compare
        if fw == 16:
            raise Unimplemented("half cmp")
        opc2 = bits(w, 4, 0)
        a = fr(rn, fw)
        b = fconst(0.0, fw) if opc2 & 0b01000 else fr(rm, fw)
        return f"FCMP_FLAGS(c, (double){a}, (double){b});"
    if bits(w, 12, 10) == 0b100:  # FMOV immediate
        if fw == 16:
            raise Unimplemented("half fmov")
        return set_scalar(rdn, fw, fconst(vfp_expand_imm(bits(w, 20, 13), fw), fw))
    if bits(w, 11, 10) == 0b01:  # FCCMP / FCCMPE
        if fw == 16:
            raise Unimplemented("half ccmp")
        cond, nzcv = bits(w, 15, 12), bits(w, 3, 0)
        return (f"if ({cond_expr(cond)}) {{ FCMP_FLAGS(c, (double){fr(rn, fw)}, (double){fr(rm, fw)}); }} "
                f"else {{ SETNZCV(c, {nzcv}); }}")
    if bits(w, 11, 10) == 0b10:  # 2-source
        if fw == 16:
            raise Unimplemented("half 2src")
        opcode = bits(w, 15, 12)
        a, b = fr(rn, fw), fr(rm, fw)
        sfx = FSUF[fw]
        e = {0: f"{a} * {b}", 1: f"{a} / {b}", 2: f"{a} + {b}", 3: f"{a} - {b}",
             4: f"FMAX{sfx}({a}, {b})", 5: f"FMIN{sfx}({a}, {b})",
             6: f"FMAXNM{sfx}({a}, {b})", 7: f"FMINNM{sfx}({a}, {b})", 8: f"-({a} * {b})"}.get(opcode)
        if e is None:
            raise Unimplemented("fp 2src")
        return set_scalar(rdn, fw, e)
    if bits(w, 11, 10) == 0b11:  # FCSEL
        cond = bits(w, 15, 12)
        if fw == 16:
            raise Unimplemented("half fcsel")
        return set_scalar(rdn, fw, f"({cond_expr(cond)}) ? {fr(rn, fw)} : {fr(rm, fw)}")
    raise Unimplemented("scalar fp")


def fp_3src(w):
    fw = ftype_width(bits(w, 23, 22))
    if fw == 16:
        raise Unimplemented("half fma")
    o1, o0 = bit(w, 21), bit(w, 15)
    rm, ra, rn, rdn = bits(w, 20, 16), bits(w, 14, 10), bits(w, 9, 5), bits(w, 4, 0)
    a, b, acc = fr(rn, fw), fr(rm, fw), fr(ra, fw)
    fma = "FMAF" if fw == 32 else "FMAD"
    e = {(0, 0): f"{fma}({a}, {b}, {acc})",
         (0, 1): f"{fma}(-{a}, {b}, {acc})",
         (1, 0): f"{fma}(-{a}, {b}, -{acc})",
         (1, 1): f"{fma}({a}, {b}, -{acc})"}[(o1, o0)]
    return set_scalar(rdn, fw, e)


ROUND_FN = {0: "FRINTN", 1: "FRINTP", 2: "FRINTM", 3: "FRINTZ"}


def f2i(fw, iw, signed, rounding, val):
    """Float -> int with ARM saturation. rounding: N/P/M/Z/A."""
    r = {"N": f"FRINTN{FSUF[fw]}", "P": f"FRINTP{FSUF[fw]}", "M": f"FRINTM{FSUF[fw]}",
         "Z": "", "A": f"FRINTA{FSUF[fw]}"}[rounding]
    v = f"{r}({val})" if r else val
    return f"F2{'S' if signed else 'U'}{iw}((double){v})"


def fp_int_conv(w):
    sf = bit(w, 31)
    ftype = bits(w, 23, 22)
    rmode, opcode = bits(w, 20, 19), bits(w, 18, 16)
    rn, rdn = bits(w, 9, 5), bits(w, 4, 0)
    iw = 64 if sf else 32
    # FMOV general <-> fp
    if opcode in (6, 7) and rmode == 0:
        fw = ftype_width(ftype)
        if opcode == 6:  # fp -> gpr
            lane = {16: "h", 32: "s", 64: "d"}[fw]
            return wr(rdn, sf, f"c->v[{rn}].{lane}[0]")
        lane = {16: "h", 32: "s", 64: "d"}[fw]
        return set_scalar(rdn, fw, f"({lane_type(lane)}){rd(rn, sf)}", lane=lane)
    if opcode in (6, 7) and rmode == 1 and ftype == 2:  # FMOV Xd, Vn.D[1] / Vd.D[1], Xn
        if opcode == 6:
            return wr(rdn, 1, f"c->v[{rn}].d[1]")
        return f"c->v[{rdn}].d[1] = {rd(rn, 1)};"
    fw = ftype_width(ftype)
    if fw == 16:
        raise Unimplemented("half conv")
    a = fr(rn, fw)
    if opcode in (2, 3) and rmode == 0:  # SCVTF / UCVTF
        src = f"(s{iw}){rd(rn, sf)}" if opcode == 2 else f"(u{iw}){rd(rn, sf)}"
        return set_scalar(rdn, fw, f"({FT[fw]}){src}")
    if opcode in (0, 1):
        rounding = "NPMZ"[rmode]
        return wr(rdn, sf, f2i(fw, iw, opcode == 0, rounding, a))
    if opcode in (4, 5) and rmode == 0:
        return wr(rdn, sf, f2i(fw, iw, opcode == 4, "A", a))
    raise Unimplemented("fp/int conv")


def fp_fixed_conv(w):
    sf = bit(w, 31)
    fw = ftype_width(bits(w, 23, 22))
    rmode, opcode = bits(w, 20, 19), bits(w, 18, 16)
    scale = bits(w, 15, 10)
    fbits = 64 - scale
    rn, rdn = bits(w, 9, 5), bits(w, 4, 0)
    iw = 64 if sf else 32
    if fw == 16:
        raise Unimplemented("half fixed")
    if rmode == 0 and opcode in (2, 3):
        src = f"(s{iw}){rd(rn, sf)}" if opcode == 2 else f"(u{iw}){rd(rn, sf)}"
        return set_scalar(rdn, fw, f"({FT[fw]})((double){src} / {float(2 ** fbits).hex()})")
    if rmode == 3 and opcode in (0, 1):
        return wr(rdn, sf, f"F2{'S' if opcode == 0 else 'U'}{iw}((double){fr(rn, fw)} * {float(2 ** fbits).hex()})")
    raise Unimplemented("fixed conv")


# ====================================================================== Advanced SIMD vector

def arr(size, q):
    esize = 8 << size
    return esize, (128 if q else 64) // esize


def adv_simd_vector(w):
    q, u = bit(w, 30), bit(w, 29)
    rm, rn, rdn = bits(w, 20, 16), bits(w, 9, 5), bits(w, 4, 0)
    size = bits(w, 23, 22)
    if w & 0x9F200400 == 0x0E200400:
        return three_same(w, q, u, size, rm, rn, rdn)
    if w & 0x9F3E0C00 == 0x0E200800:
        return two_reg_misc(w, q, u, size, rn, rdn)
    if w & 0x9F3E0C00 == 0x0E300800:
        return across_lanes(w, q, u, size, rn, rdn)
    if w & 0x9F200C00 == 0x0E200000:
        return three_diff(w, q, u, size, rm, rn, rdn)
    if w & 0x9FE08400 == 0x0E000400:
        return simd_copy(w, q, u, rn, rdn)
    if w & 0xBF208C00 == 0x0E000800:
        return permute(w, q, size, rm, rn, rdn)
    if w & 0xBFE08400 == 0x2E000000:
        imm4 = bits(w, 14, 11)
        nbytes = 16 if q else 8
        return (f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}], r_ = V_ZERO; u8 t_[32]; "
                f"memcpy(t_, a_.b, {nbytes}); memcpy(t_ + {nbytes}, b_.b, {nbytes}); "
                f"memcpy(r_.b, t_ + {imm4}, {nbytes}); c->v[{rdn}] = r_; }}")
    if w & 0xBFE08C00 == 0x0E000000:  # TBL / TBX
        length = bits(w, 14, 13) + 1
        tbx = bit(w, 12)
        regs = ", ".join(f"c->v[{(rn + i) % 32}]" for i in range(length))
        n = 16 if q else 8
        return (f"{{ V128 t_[{length}] = {{ {regs} }}; V128 i_ = c->v[{rm}], r_ = "
                f"{'c->v[' + str(rdn) + ']' if tbx else 'V_ZERO'}; "
                f"{'if (!' + str(q) + ') r_.d[1] = 0; ' if tbx else ''}"
                f"for (int i = 0; i < {n}; i++) {{ u32 x_ = i_.b[i]; if (x_ < {16 * length}) r_.b[i] = t_[x_ >> 4].b[x_ & 15]; }} "
                f"c->v[{rdn}] = r_; }}")
    if w & 0x9FF80400 == 0x0F000400:
        return mod_imm(w, q, rdn)
    if w & 0x9F800400 == 0x0F000400:
        return shift_imm(w, q, u, rn, rdn, scalar=False)
    if w & 0x9F000400 == 0x0F000000:
        return indexed(w, q, u, size, rn, rdn, scalar=False)
    raise Unimplemented("adv simd vector")


def three_same(w, q, u, size, rm, rn, rdn):
    opcode = bits(w, 15, 11)
    if opcode >= 0b11000:  # FP
        a_bit, sz = bit(w, 23), bit(w, 22)
        fw = 64 if sz else 32
        if sz and not q:
            raise Unimplemented("reserved")
        n = (128 if q else 64) // fw
        lf = F[fw]
        sfx = FSUF[fw]
        A, B, D = f"a_.{lf}[i]", f"b_.{lf}[i]", f"r_.{lf}[i]"
        it = UT[fw]
        key = (u, a_bit, opcode)
        srcs = {"a": rn, "b": rm}
        cmp_ = lambda e: vec_op(rdn, n, U[fw], f"({e}) ? ({it})~({it})0 : 0", srcs, q=q)
        table = {
            (0, 0, 0b11000): f"FMAXNM{sfx}({A}, {B})",
            (0, 1, 0b11000): f"FMINNM{sfx}({A}, {B})",
            (0, 0, 0b11010): f"{A} + {B}",
            (0, 1, 0b11010): f"{A} - {B}",
            (1, 0, 0b11011): f"{A} * {B}",
            (0, 0, 0b11011): f"FMULX{sfx}({A}, {B})",
            (1, 0, 0b11111): f"{A} / {B}",
            (0, 0, 0b11110): f"FMAX{sfx}({A}, {B})",
            (0, 1, 0b11110): f"FMIN{sfx}({A}, {B})",
            (0, 0, 0b11111): f"FRECPS{sfx}({A}, {B})",
            (0, 1, 0b11111): f"FRSQRTS{sfx}({A}, {B})",
            (1, 1, 0b11010): f"FABS{sfx}({A} - {B})",
        }
        if key in table:
            return vec_op(rdn, n, lf, table[key], srcs, q=q)
        if key == (0, 0, 0b11001):
            return vec_op(rdn, n, lf, f"FMA{sfx}(a_.{lf}[i], b_.{lf}[i], d_.{lf}[i])",
                          {"a": rn, "b": rm, "d": rdn}, q=q)
        if key == (0, 1, 0b11001):
            return vec_op(rdn, n, lf, f"FMA{sfx}(-a_.{lf}[i], b_.{lf}[i], d_.{lf}[i])",
                          {"a": rn, "b": rm, "d": rdn}, q=q)
        if key == (0, 0, 0b11100):
            return cmp_(f"{A} == {B}")
        if key == (1, 0, 0b11100):
            return cmp_(f"{A} >= {B}")
        if key == (1, 1, 0b11100):
            return cmp_(f"{A} > {B}")
        if key == (1, 0, 0b11101):
            return cmp_(f"FABS{sfx}({A}) >= FABS{sfx}({B})")
        if key == (1, 1, 0b11101):
            return cmp_(f"FABS{sfx}({A}) > FABS{sfx}({B})")
        pair_ops = {(1, 0, 0b11010): "{x} + {y}", (1, 0, 0b11110): f"FMAX{sfx}({{x}}, {{y}})",
                    (1, 1, 0b11110): f"FMIN{sfx}({{x}}, {{y}})", (1, 0, 0b11000): f"FMAXNM{sfx}({{x}}, {{y}})",
                    (1, 1, 0b11000): f"FMINNM{sfx}({{x}}, {{y}})"}
        if key in pair_ops:
            return pairwise(rdn, rn, rm, n, lf, pair_ops[key])
        raise Unimplemented(f"fp three same {key}")
    esize, n = arr(size, q)
    lu, ls = U[esize], S[esize]
    it = UT[esize]
    A, B = f"a_.{lu}[i]", f"b_.{lu}[i]"
    SA, SB = f"a_.{ls}[i]", f"b_.{ls}[i]"
    srcs = {"a": rn, "b": rm}
    ones = f"({it})~({it})0"
    if opcode == 0b00011:  # logical
        n64 = 2 if q else 1
        if u == 0:
            e = {0: "a_.d[i] & b_.d[i]", 1: "a_.d[i] & ~b_.d[i]", 2: "a_.d[i] | b_.d[i]",
                 3: "a_.d[i] | ~b_.d[i]"}[size]
            if size == 2 and rn == rm:
                return f"c->v[{rdn}] = {'c->v[' + str(rn) + ']' if q else 'V_LO(c->v[' + str(rn) + '])'};"
            return vec_op(rdn, n64, "d", e, srcs, q=q)
        e = {0: "a_.d[i] ^ b_.d[i]",
             1: "(d_.d[i] & a_.d[i]) | (~d_.d[i] & b_.d[i])",
             2: "(d_.d[i] & ~b_.d[i]) | (a_.d[i] & b_.d[i])",
             3: "(d_.d[i] & b_.d[i]) | (a_.d[i] & ~b_.d[i])"}[size]
        return vec_op(rdn, n64, "d", e, {"a": rn, "b": rm, "d": rdn}, q=q)
    table = {
        (0, 0b10000): f"({it})({A} + {B})",
        (1, 0b10000): f"({it})({A} - {B})",
        (0, 0b10011): f"({it})({A} * {B})",
        (1, 0b10001): f"({A} == {B}) ? {ones} : 0",
        (0, 0b10001): f"({A} & {B}) ? {ones} : 0",
        (0, 0b00110): f"({SA} > {SB}) ? {ones} : 0",
        (1, 0b00110): f"({A} > {B}) ? {ones} : 0",
        (0, 0b00111): f"({SA} >= {SB}) ? {ones} : 0",
        (1, 0b00111): f"({A} >= {B}) ? {ones} : 0",
        (0, 0b01100): f"({it})(({SA} > {SB}) ? {SA} : {SB})",
        (1, 0b01100): f"({A} > {B}) ? {A} : {B}",
        (0, 0b01101): f"({it})(({SA} < {SB}) ? {SA} : {SB})",
        (1, 0b01101): f"({A} < {B}) ? {A} : {B}",
        (0, 0b01000): f"({it})SSHL{esize}({SA}, (s8)b_.{lu}[i])",
        (1, 0b01000): f"({it})USHL{esize}({A}, (s8)b_.{lu}[i])",
    }
    if (u, opcode) in table:
        return vec_op(rdn, n, lu, table[(u, opcode)], srcs, q=q)
    if opcode in (0b00001, 0b00101):  # SQADD / UQADD / SQSUB / UQSUB
        add = opcode == 0b00001
        if esize == 64:
            fn = ("HW_SQADD64" if add else "HW_SQSUB64") if u == 0 else ("HW_UQADD64" if add else "HW_UQSUB64")
            cast = "(s64)" if u == 0 else ""
            return vec_op(rdn, n, lu, f"({it}){fn}({cast}a_.{lu}[i], {cast}b_.{lu}[i])", srcs, q=q)
        op = "+" if add else "-"
        if u == 0:
            e = f"({it})HW_SAT_S((s64)a_.{ls}[i] {op} (s64)b_.{ls}[i], {esize})"
        else:
            e = f"({it})HW_SAT_U((s64)a_.{lu}[i] {op} (s64)b_.{lu}[i], {esize})"
        return vec_op(rdn, n, lu, e, srcs, q=q)
    if opcode == 0b10010:  # MLA / MLS
        op = "-" if u else "+"
        return vec_op(rdn, n, lu, f"({it})(d_.{lu}[i] {op} a_.{lu}[i] * b_.{lu}[i])",
                      {"a": rn, "b": rm, "d": rdn}, q=q)
    if (u, opcode) == (0, 0b10111):
        return pairwise(rdn, rn, rm, n, lu, f"({it})({{x}} + {{y}})")
    if (u, opcode) in ((0, 0b10100), (1, 0b10100), (0, 0b10101), (1, 0b10101)):
        lane = ls if u == 0 else lu
        cmp = ">" if opcode == 0b10100 else "<"
        return pairwise(rdn, rn, rm, n, lane, f"(({{x}}) {cmp} ({{y}}) ? ({{x}}) : ({{y}}))")
    raise Unimplemented(f"int three same u={u} op={opcode:05b}")


def pairwise(rdn, rn, rm, n, lane, op):
    half = n // 2
    x0, y0 = f"a_.{lane}[2 * i]", f"a_.{lane}[2 * i + 1]"
    x1, y1 = f"b_.{lane}[2 * i]", f"b_.{lane}[2 * i + 1]"
    return (f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}], r_ = V_ZERO; "
            f"for (int i = 0; i < {half}; i++) {{ r_.{lane}[i] = {op.format(x=x0, y=y0)}; "
            f"r_.{lane}[{half} + i] = {op.format(x=x1, y=y1)}; }} c->v[{rdn}] = r_; }}")


def two_reg_misc(w, q, u, size, rn, rdn):
    opcode = bits(w, 16, 12)
    esize, n = arr(size, q)
    srcs = {"a": rn}
    if opcode >= 0b01100 and opcode not in (0b10010, 0b10011, 0b10100):  # FP-ish
        a_bit, sz = bit(w, 23), bit(w, 22)
        fw = 64 if sz else 32
        nf = (128 if q else 64) // fw
        lf, sfx, it = F[fw], FSUF[fw], UT[fw]
        A = f"a_.{lf}[i]"
        ones = f"({it})~({it})0"
        key = (u, a_bit, opcode)
        if key == (0, 0, 0b10110):  # FCVTN(2): double->single narrow
            if sz:
                if q:
                    return (f"{{ V128 a_ = c->v[{rn}], r_ = c->v[{rdn}]; r_.f[2] = (float)a_.fd[0]; "
                            f"r_.f[3] = (float)a_.fd[1]; c->v[{rdn}] = r_; }}")
                return (f"{{ V128 a_ = c->v[{rn}], r_ = V_ZERO; r_.f[0] = (float)a_.fd[0]; "
                        f"r_.f[1] = (float)a_.fd[1]; c->v[{rdn}] = r_; }}")
            if q:
                return (f"{{ V128 a_ = c->v[{rn}], r_ = c->v[{rdn}]; "
                        f"for (int i = 0; i < 4; i++) r_.h[4 + i] = F2H(a_.f[i]); c->v[{rdn}] = r_; }}")
            return (f"{{ V128 a_ = c->v[{rn}], r_ = V_ZERO; "
                    f"for (int i = 0; i < 4; i++) r_.h[i] = F2H(a_.f[i]); c->v[{rdn}] = r_; }}")
        if key == (0, 0, 0b10111):  # FCVTL(2)
            if sz:
                off = 2 if q else 0
                return (f"{{ V128 a_ = c->v[{rn}], r_; r_.fd[0] = (double)a_.f[{off}]; "
                        f"r_.fd[1] = (double)a_.f[{off + 1}]; c->v[{rdn}] = r_; }}")
            off = 4 if q else 0
            return (f"{{ V128 a_ = c->v[{rn}], r_; for (int i = 0; i < 4; i++) "
                    f"r_.f[i] = H2F(a_.h[{off} + i]); c->v[{rdn}] = r_; }}")
        table = {
            (0, 1, 0b01111): f"FABS{sfx}({A})",
            (1, 1, 0b01111): f"-{A}",
            (1, 1, 0b11111): f"FSQRT{sfx}({A})",
            (0, 1, 0b11101): f"({FT[fw]})1.0 / {A}",
            (1, 1, 0b11101): f"({FT[fw]})1.0 / FSQRT{sfx}({A})",
            (0, 0, 0b11000): f"FRINTN{sfx}({A})",
            (0, 0, 0b11001): f"FRINTM{sfx}({A})",
            (0, 1, 0b11000): f"FRINTP{sfx}({A})",
            (0, 1, 0b11001): f"FRINTZ{sfx}({A})",
            (1, 0, 0b11000): f"FRINTA{sfx}({A})",
            (1, 0, 0b11001): f"FRINTX{sfx}({A})",
            (1, 1, 0b11001): f"FRINTX{sfx}({A})",
            (0, 0, 0b11101): f"({FT[fw]})a_.{S[fw]}[i]",
            (1, 0, 0b11101): f"({FT[fw]})a_.{U[fw]}[i]",
        }
        if key in table:
            return vec_op(rdn, nf, lf, table[key], srcs, q=q)
        cvt = {(0, 1, 0b11011): ("Z", 1), (1, 1, 0b11011): ("Z", 0), (0, 0, 0b11011): ("M", 1),
               (1, 0, 0b11011): ("M", 0), (0, 1, 0b11010): ("P", 1), (1, 1, 0b11010): ("P", 0),
               (0, 0, 0b11010): ("N", 1), (1, 0, 0b11010): ("N", 0), (0, 0, 0b11100): ("A", 1),
               (1, 0, 0b11100): ("A", 0)}
        if key in cvt:
            r, signed = cvt[key]
            return vec_op(rdn, nf, U[fw], f"({it}){f2i(fw, fw, signed, r, A)}", srcs, q=q)
        cmpz = {(0, 1, 0b01100): ">", (0, 1, 0b01101): "==", (0, 1, 0b01110): "<",
                (1, 1, 0b01100): ">=", (1, 1, 0b01101): "<="}
        if key in cmpz:
            return vec_op(rdn, nf, U[fw], f"({A} {cmpz[key]} 0) ? {ones} : 0", srcs, q=q)
        raise Unimplemented(f"fp two-reg misc {key}")
    lu, ls, it = U[esize], S[esize], UT[esize]
    A, SA = f"a_.{lu}[i]", f"a_.{ls}[i]"
    ones = f"({it})~({it})0"
    if (u, opcode) == (1, 0b00101) and size == 0:  # NOT
        return vec_op(rdn, 2 if q else 1, "d", "~a_.d[i]", srcs, q=q)
    if (u, opcode) == (0, 0b00101):  # CNT
        return vec_op(rdn, n, "b", "(u8)__builtin_popcount(a_.b[i])", srcs, q=q)
    if (u, opcode) == (0, 0b00000):  # REV64
        per = 64 // esize
        return vec_op(rdn, n, lu, f"a_.{lu}[(i / {per}) * {per} + ({per} - 1 - i % {per})]", srcs, q=q)
    if (u, opcode) == (1, 0b00000):  # REV32
        per = 32 // esize
        return vec_op(rdn, n, lu, f"a_.{lu}[(i / {per}) * {per} + ({per} - 1 - i % {per})]", srcs, q=q)
    if (u, opcode) == (0, 0b00001):  # REV16
        return vec_op(rdn, n, "b", "a_.b[i ^ 1]", srcs, q=q)
    table = {
        (0, 0b01011): f"({it})(({SA} < 0) ? -{SA} : {SA})",
        (1, 0b01011): f"({it})(0 - {A})",
        (0, 0b01000): f"({SA} > 0) ? {ones} : 0",
        (0, 0b01001): f"({A} == 0) ? {ones} : 0",
        (0, 0b01010): f"({SA} < 0) ? {ones} : 0",
        (1, 0b01000): f"({SA} >= 0) ? {ones} : 0",
        (1, 0b01001): f"({SA} <= 0) ? {ones} : 0",
    }
    if (u, opcode) in table:
        return vec_op(rdn, n, lu, table[(u, opcode)], srcs, q=q)
    if (u, opcode) == (0, 0b10010):  # XTN(2)
        dw = esize
        sw = esize * 2
        cnt = 64 // dw
        off = cnt if q else 0
        init = f"c->v[{rdn}]" if q else "V_ZERO"
        return (f"{{ V128 a_ = c->v[{rn}], r_ = {init}; for (int i = 0; i < {cnt}; i++) "
                f"r_.{U[dw]}[{off} + i] = ({UT[dw]})a_.{U[sw]}[i]; c->v[{rdn}] = r_; }}")
    if (u, opcode) in ((0, 0b00010), (1, 0b00010)):  # SADDLP / UADDLP
        dw = esize * 2
        lane = ls if u == 0 else lu
        return vec_op(rdn, n // 2, U[dw], f"({UT[dw]})(a_.{lane}[2 * i] + a_.{lane}[2 * i + 1])", srcs, q=q)
    raise Unimplemented(f"int two-reg misc u={u} op={opcode:05b}")


def across_lanes(w, q, u, size, rn, rdn):
    opcode = bits(w, 16, 12)
    esize, n = arr(size, q)
    lu, ls = U[esize], S[esize]
    if opcode == 0b11011:  # ADDV
        return (f"{{ V128 a_ = c->v[{rn}]; {UT[esize]} s_ = 0; for (int i = 0; i < {n}; i++) s_ += a_.{lu}[i]; "
                f"{set_scalar(rdn, esize, 's_', lane=lu)} }}")
    if opcode in (0b01010, 0b11010):  # S/U MAX/MIN V
        lane = lu if u else ls
        ty = UT[esize] if u else STY[esize]
        cmp = ">" if opcode == 0b01010 else "<"
        return (f"{{ V128 a_ = c->v[{rn}]; {ty} s_ = a_.{lane}[0]; for (int i = 1; i < {n}; i++) "
                f"if (a_.{lane}[i] {cmp} s_) s_ = a_.{lane}[i]; {set_scalar(rdn, esize, f'({UT[esize]})s_', lane=lu)} }}")
    if opcode == 0b00011:  # SADDLV / UADDLV
        dw = esize * 2
        lane = lu if u else ls
        return (f"{{ V128 a_ = c->v[{rn}]; s64 s_ = 0; for (int i = 0; i < {n}; i++) s_ += a_.{lane}[i]; "
                f"{set_scalar(rdn, dw, f'({UT[dw]})s_', lane=U[dw])} }}")
    if u == 1 and opcode in (0b01100, 0b01111) and bit(w, 22) == 0 and q:
        mn = bit(w, 23)
        fn = ("FMINNMF" if mn else "FMAXNMF") if opcode == 0b01100 else ("FMINF" if mn else "FMAXF")
        return (f"{{ V128 a_ = c->v[{rn}]; float s_ = {fn}({fn}(a_.f[0], a_.f[1]), {fn}(a_.f[2], a_.f[3])); "
                f"{set_scalar(rdn, 32, 's_')} }}")
    raise Unimplemented(f"across lanes op={opcode:05b}")


def three_diff(w, q, u, size, rm, rn, rdn):
    opcode = bits(w, 15, 12)
    esize = 8 << size
    if opcode == 0b1110 and u == 0:  # PMULL / PMULL2
        if size == 3:
            return f"c->v[{rdn}] = PMULL64(c->v[{rn}].d[{q}], c->v[{rm}].d[{q}]);"
        if size == 0:
            off = 8 if q else 0
            return (f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}], r_; for (int i = 0; i < 8; i++) "
                    f"r_.h[i] = PMUL8(a_.b[{off} + i], b_.b[{off} + i]); c->v[{rdn}] = r_; }}")
        raise Unimplemented("pmull size")
    if esize == 64:
        raise Unimplemented("three diff size")
    dw = esize * 2
    n = 64 // esize
    off = n if q else 0
    sl = S[esize] if u == 0 else U[esize]
    dl = S[dw] if u == 0 else U[dw]
    a = f"(s64)a_.{sl}[{off} + i]" if u == 0 else f"(u64)a_.{sl}[{off} + i]"
    b = f"(s64)b_.{sl}[{off} + i]" if u == 0 else f"(u64)b_.{sl}[{off} + i]"
    if opcode == 0b1100:
        e = f"{a} * {b}"
    elif opcode == 0b0000:
        e = f"{a} + {b}"
    elif opcode == 0b0010:
        e = f"{a} - {b}"
    elif opcode == 0b1000:
        e = f"d_.{dl}[i] + {a} * {b}"
    elif opcode == 0b1010:
        e = f"d_.{dl}[i] - {a} * {b}"
    elif opcode in (0b0001, 0b0011):  # SADDW / SSUBW
        aw = f"a_.{dl}[i]"
        e = f"{aw} {'+' if opcode == 1 else '-'} {b}"
    else:
        raise Unimplemented(f"three diff op={opcode:04b}")
    return (f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}], d_ = c->v[{rdn}], r_; (void)d_; "
            f"for (int i = 0; i < {n}; i++) r_.{U[dw]}[i] = ({UT[dw]})({e}); c->v[{rdn}] = r_; }}")


def simd_copy(w, q, op, rn, rdn):
    imm5, imm4 = bits(w, 20, 16), bits(w, 14, 11)
    size = (imm5 & -imm5).bit_length() - 1
    if size > 3:
        raise Unimplemented("copy size")
    esize = 8 << size
    idx = imm5 >> (size + 1)
    lu = U[esize]
    n = (128 if q else 64) // esize
    if op == 0 and imm4 == 0:  # DUP element
        return f"{{ V128 a_ = c->v[{rn}], r_ = V_ZERO; for (int i = 0; i < {n}; i++) r_.{lu}[i] = a_.{lu}[{idx}]; c->v[{rdn}] = r_; }}"
    if op == 0 and imm4 == 1:  # DUP general
        return f"{{ {UT[esize]} x_ = ({UT[esize]}){rd(rn, esize == 64)}; V128 r_ = V_ZERO; for (int i = 0; i < {n}; i++) r_.{lu}[i] = x_; c->v[{rdn}] = r_; }}"
    if op == 0 and imm4 == 3 and q:  # INS general
        return f"c->v[{rdn}].{lu}[{idx}] = ({UT[esize]}){rd(rn, esize == 64)};"
    if op == 0 and imm4 == 5:  # SMOV
        return wr(rdn, q, f"(s64)c->v[{rn}].{S[esize]}[{idx}]")
    if op == 0 and imm4 == 7:  # UMOV
        return wr(rdn, esize == 64, f"c->v[{rn}].{lu}[{idx}]")
    if op == 1 and q:  # INS element
        idx2 = imm4 >> size
        return f"c->v[{rdn}].{lu}[{idx}] = c->v[{rn}].{lu}[{idx2}];"
    raise Unimplemented("simd copy")


def permute(w, q, size, rm, rn, rdn):
    opcode = bits(w, 14, 12)
    esize, n = arr(size, q)
    lu = U[esize]
    half = n // 2
    if opcode in (1, 5):  # UZP1/2
        p = 0 if opcode == 1 else 1
        return (f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}], r_ = V_ZERO; for (int i = 0; i < {half}; i++) "
                f"{{ r_.{lu}[i] = a_.{lu}[2 * i + {p}]; r_.{lu}[{half} + i] = b_.{lu}[2 * i + {p}]; }} c->v[{rdn}] = r_; }}")
    if opcode in (2, 6):  # TRN1/2
        p = 0 if opcode == 2 else 1
        return (f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}], r_ = V_ZERO; for (int i = 0; i < {half}; i++) "
                f"{{ r_.{lu}[2 * i] = a_.{lu}[2 * i + {p}]; r_.{lu}[2 * i + 1] = b_.{lu}[2 * i + {p}]; }} c->v[{rdn}] = r_; }}")
    if opcode in (3, 7):  # ZIP1/2
        base = 0 if opcode == 3 else half
        return (f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}], r_ = V_ZERO; for (int i = 0; i < {half}; i++) "
                f"{{ r_.{lu}[2 * i] = a_.{lu}[{base} + i]; r_.{lu}[2 * i + 1] = b_.{lu}[{base} + i]; }} c->v[{rdn}] = r_; }}")
    raise Unimplemented("permute")


def mod_imm(w, q, rdn):
    op = bit(w, 29)
    cmode = bits(w, 15, 12)
    imm8 = (bits(w, 18, 16) << 5) | bits(w, 9, 5)
    if bit(w, 11):
        raise Unimplemented("fmov half vec")
    if cmode == 0b1111 and op == 1 and not q:
        raise Unimplemented("reserved mod imm")
    imm = adv_simd_expand_imm(op, cmode, imm8)
    is_orr = cmode < 0b1100 and (cmode & 1) == 1
    if cmode < 0b1110 and op == 1 and not (cmode < 0b1100 and cmode & 1):  # MVNI
        imm = ~imm & mask(64)
    hi = imm if q else 0
    if cmode < 0b1100 and cmode & 1:
        if op == 0:  # ORR
            return (f"{{ V128 r_ = c->v[{rdn}]; r_.d[0] |= {hx(imm)}; r_.d[1] = {('r_.d[1] | ' + hx(imm)) if q else '0'}; "
                    f"c->v[{rdn}] = r_; }}")
        nimm = ~imm & mask(64)  # BIC
        return (f"{{ V128 r_ = c->v[{rdn}]; r_.d[0] &= {hx(nimm)}; r_.d[1] = {('r_.d[1] & ' + hx(nimm)) if q else '0'}; "
                f"c->v[{rdn}] = r_; }}")
    return f"{{ V128 r_; r_.d[0] = {hx(imm)}; r_.d[1] = {hx(hi)}; c->v[{rdn}] = r_; }}"


def shift_imm(w, q, u, rn, rdn, scalar):
    immh, immb = bits(w, 22, 19), bits(w, 18, 16)
    opcode = bits(w, 15, 11)
    if immh == 0:
        raise Unimplemented("shift imm immh=0")
    hsb = immh.bit_length() - 1
    esize = 8 << hsb
    full = (immh << 3) | immb
    rshift = 2 * esize - full
    lshift = full - esize
    if scalar:
        if esize != 64 and opcode not in (0b11100, 0b11111):
            raise Unimplemented("scalar shift size")
        n = 1
    else:
        n = (128 if q else 64) // esize
    lu, ls, it = U[esize], S[esize], UT[esize]
    srcs = {"a": rn}

    def emit(lane, e, keep=False):
        if scalar:
            extra = {"d": rdn} if keep else {}
            s2 = dict(srcs, **extra)
            decl = ", ".join(f"{k}_ = c->v[{v}]" for k, v in s2.items())
            return f"{{ V128 {decl}; {set_scalar(rdn, esize, e.replace('[i]', '[0]'), lane=lane)} }}"
        if keep:
            return vec_op(rdn, n, lane, e, dict(srcs, d=rdn), q=q)
        return vec_op(rdn, n, lane, e, srcs, q=q)

    if opcode == 0b00000:  # SSHR / USHR
        if u:
            return emit(lu, f"({it})(a_.{lu}[i] >> {rshift})" if rshift < esize else "0")
        return emit(lu, f"({it})(a_.{ls}[i] >> {min(rshift, esize - 1)})")
    if opcode == 0b00010:  # SSRA / USRA
        sh = f"(a_.{lu}[i] >> {rshift})" if u else f"(a_.{ls}[i] >> {min(rshift, esize - 1)})"
        if u and rshift >= esize:
            sh = "0"
        return emit(lu, f"({it})(d_.{lu}[i] + ({it}){sh})", keep=True)
    if opcode == 0b01010 and u == 0:  # SHL
        return emit(lu, f"({it})(a_.{lu}[i] << {lshift})")
    if opcode == 0b01010 and u == 1:  # SLI
        m = (mask(esize) << lshift) & mask(esize)
        return emit(lu, f"({it})((d_.{lu}[i] & ({it})~({it}){hx(m)}) | (({it})(a_.{lu}[i] << {lshift})))", keep=True)
    if opcode == 0b01000 and u == 1:  # SRI
        m = mask(esize) >> rshift if rshift < esize else 0
        return emit(lu, f"({it})((d_.{lu}[i] & ({it})~({it}){hx(m)}) | ({it})(a_.{lu}[i] >> {rshift}))"
                    if rshift < esize else f"d_.{lu}[i]", keep=True)
    if opcode == 0b10100 and not scalar:  # SSHLL / USHLL (2)
        dw = esize * 2
        src_esize = esize
        cnt = 64 // src_esize
        off = cnt if q else 0
        lane = U[src_esize] if u else S[src_esize]
        return (f"{{ V128 a_ = c->v[{rn}], r_; for (int i = 0; i < {cnt}; i++) "
                f"r_.{U[dw]}[i] = ({UT[dw]})(({'u64' if u else 's64'})a_.{lane}[{off} + i] << {lshift}); c->v[{rdn}] = r_; }}")
    if opcode in (0b10000, 0b10001) and u == 0 and not scalar:  # SHRN / RSHRN (2)
        # immh encodes the destination element size: source is 2x.
        dw = esize
        sw = esize * 2
        sh = 2 * dw - full
        cnt = 64 // dw
        off = cnt if q else 0
        rnd = f" + ((u64)1 << {sh - 1})" if opcode == 0b10001 else ""
        init = f"c->v[{rdn}]" if q else "V_ZERO"
        return (f"{{ V128 a_ = c->v[{rn}], r_ = {init}; for (int i = 0; i < {cnt}; i++) "
                f"r_.{U[dw]}[{off} + i] = ({UT[dw]})(((u64)a_.{U[sw]}[i]{rnd}) >> {sh}); c->v[{rdn}] = r_; }}")
    if opcode in (0b10010, 0b10011) or (u == 1 and opcode in (0b10000, 0b10001)):
        # SQSHRN/SQRSHRN (U0), UQSHRN/UQRSHRN (U1, 1001x), SQSHRUN/SQRSHRUN (U1, 1000x) [+2]
        if scalar:
            raise Unimplemented("scalar saturating narrow")
        dw, sw = esize, esize * 2
        sh = 2 * dw - full
        cnt = 64 // dw
        off = cnt if q else 0
        rounding = opcode & 1
        signed_src = not (u == 1 and opcode in (0b10010, 0b10011))
        unsigned_dst = u == 1
        src = f"(s64)a_.{S[sw]}[i]" if signed_src else f"(s64)a_.{U[sw]}[i]"
        if sw == 64 and not signed_src:
            src = f"(s64)(a_.{U[sw]}[i] >> 1)"  # keep the u64 path in range; shift adjusted below
            sh_expr = f"{sh - 1}" if sh > 0 else "0"
        else:
            sh_expr = f"{sh}"
        rnd = f" + ((s64)1 << ({sh_expr} - 1))" if rounding and sh > 0 else ""
        val = f"(({src}{rnd}) >> {sh_expr})"
        sat = f"HW_SAT_U({val}, {dw})" if unsigned_dst else f"(u64)HW_SAT_S({val}, {dw})"
        init = f"c->v[{rdn}]" if q else "V_ZERO"
        return (f"{{ V128 a_ = c->v[{rn}], r_ = {init}; for (int i = 0; i < {cnt}; i++) "
                f"r_.{U[dw]}[{off} + i] = ({UT[dw]}){sat}; c->v[{rdn}] = r_; }}")
    if opcode == 0b11100:  # SCVTF / UCVTF fixed
        if esize not in (32, 64):
            raise Unimplemented("cvtf fixed size")
        fb = rshift
        lane = lu if u else ls
        return emit(F[esize], f"({FT[esize]})((double)a_.{lane}[i] / {float(2 ** fb).hex()})")
    if opcode == 0b11111:  # FCVTZS / FCVTZU fixed
        if esize not in (32, 64):
            raise Unimplemented("fcvtz fixed size")
        fb = rshift
        return emit(lu, f"({it})F2{'U' if u else 'S'}{esize}((double)a_.{F[esize]}[i] * {float(2 ** fb).hex()})")
    raise Unimplemented(f"shift imm op={opcode:05b} u={u}")


def indexed(w, q, u, size, rn, rdn, scalar):
    opcode = bits(w, 15, 12)
    l, m, h = bit(w, 21), bit(w, 20), bit(w, 11)
    rm4 = bits(w, 19, 16)
    if opcode in (0b0001, 0b0101, 0b1001) and (u == 0 or opcode == 0b1001):
        sz = bit(w, 22)
        if size >> 1 != 1:
            raise Unimplemented("fp indexed half")
        fw = 64 if sz else 32
        idx = (h << 1 | l) if not sz else h
        rm = (m << 4) | rm4
        lf, sfx = F[fw], FSUF[fw]
        n = 1 if scalar else (128 if q else 64) // fw
        B = f"b_.{lf}[{idx}]"
        if opcode == 0b1001:
            e = f"FMULX{sfx}(a_.{lf}[i], {B})" if u else f"a_.{lf}[i] * {B}"
        elif opcode == 0b0001:
            e = f"FMA{sfx}(a_.{lf}[i], {B}, d_.{lf}[i])"
        else:
            e = f"FMA{sfx}(-a_.{lf}[i], {B}, d_.{lf}[i])"
        if scalar:
            e0 = e.replace("[i]", "[0]")
            return (f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}], d_ = c->v[{rdn}]; (void)d_; "
                    f"{set_scalar(rdn, fw, e0)} }}")
        return vec_op(rdn, n, lf, e, {"a": rn, "b": rm, "d": rdn}, q=q)
    if scalar:
        raise Unimplemented("scalar int indexed")
    esize = 8 << size
    if esize == 16:
        idx, rm = (h << 2) | (l << 1) | m, rm4
    elif esize == 32:
        idx, rm = (h << 1) | l, (m << 4) | rm4
    else:
        raise Unimplemented("int indexed size")
    lu, it = U[esize], UT[esize]
    n = (128 if q else 64) // esize
    B = f"b_.{lu}[{idx}]"
    if opcode == 0b1000 and u == 0:
        return vec_op(rdn, n, lu, f"({it})(a_.{lu}[i] * {B})", {"a": rn, "b": rm}, q=q)
    if opcode in (0b0000, 0b0100) and u == 1:
        op = "+" if opcode == 0 else "-"
        return vec_op(rdn, n, lu, f"({it})(d_.{lu}[i] {op} a_.{lu}[i] * {B})", {"a": rn, "b": rm, "d": rdn}, q=q)
    if opcode in (0b1010, 0b0010, 0b0110):  # S/UMULL, S/UMLAL, S/UMLSL by element
        dw = esize * 2
        cnt = 64 // esize
        off = cnt if q else 0
        sl = U[esize] if u else S[esize]
        cast = "u64" if u else "s64"
        prod = f"({cast})a_.{sl}[{off} + i] * ({cast})b_.{sl}[{idx}]"
        e = {0b1010: prod, 0b0010: f"d_.{U[dw]}[i] + {prod}", 0b0110: f"d_.{U[dw]}[i] - {prod}"}[opcode]
        return (f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}], d_ = c->v[{rdn}], r_; (void)d_; "
                f"for (int i = 0; i < {cnt}; i++) r_.{U[dw]}[i] = ({UT[dw]})({e}); c->v[{rdn}] = r_; }}")
    raise Unimplemented(f"indexed op={opcode:04b} u={u}")


# ====================================================================== Advanced SIMD scalar

def adv_simd_scalar(w):
    u = bit(w, 29)
    size = bits(w, 23, 22)
    rm, rn, rdn = bits(w, 20, 16), bits(w, 9, 5), bits(w, 4, 0)
    if w & 0xDF3E0C00 == 0x5E300800:  # scalar pairwise
        opcode = bits(w, 16, 12)
        if u == 0 and opcode == 0b11011 and size == 3:
            return f"{{ V128 a_ = c->v[{rn}]; {set_scalar(rdn, 64, 'a_.d[0] + a_.d[1]', lane='d')} }}"
        if u == 1:
            sz = bit(w, 22)
            fw = 64 if sz else 32
            lf, sfx = F[fw], FSUF[fw]
            x, y = f"a_.{lf}[0]", f"a_.{lf}[1]"
            e = {(0b01101, 0): f"{x} + {y}", (0b01111, 0): f"FMAX{sfx}({x}, {y})", (0b01111, 1): f"FMIN{sfx}({x}, {y})",
                 (0b01100, 0): f"FMAXNM{sfx}({x}, {y})", (0b01100, 1): f"FMINNM{sfx}({x}, {y})"}.get((opcode, bit(w, 23)))
            if e:
                return f"{{ V128 a_ = c->v[{rn}]; {set_scalar(rdn, fw, e)} }}"
        raise Unimplemented("scalar pairwise")
    if w & 0xDF200400 == 0x5E200400:  # scalar three same
        opcode = bits(w, 15, 11)
        if opcode >= 0b11000:
            a_bit, sz = bit(w, 23), bit(w, 22)
            fw = 64 if sz else 32
            lf, sfx, it = F[fw], FSUF[fw], UT[fw]
            A, B = f"a_.{lf}[0]", f"b_.{lf}[0]"
            ones = f"({it})~({it})0"
            key = (u, a_bit, opcode)
            fe = {(1, 1, 0b11010): f"FABS{sfx}({A} - {B})", (0, 0, 0b11011): f"FMULX{sfx}({A}, {B})",
                  (0, 0, 0b11111): f"FRECPS{sfx}({A}, {B})",
                  (0, 1, 0b11111): f"FRSQRTS{sfx}({A}, {B})"}
            if key in fe:
                return f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}]; {set_scalar(rdn, fw, fe[key])} }}"
            ce = {(0, 0, 0b11100): "==", (1, 0, 0b11100): ">=", (1, 1, 0b11100): ">"}
            if key in ce:
                return (f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}]; "
                        f"{set_scalar(rdn, fw, f'({A} {ce[key]} {B}) ? {ones} : 0', lane=U[fw])} }}")
            ae = {(1, 0, 0b11101): ">=", (1, 1, 0b11101): ">"}
            if key in ae:
                return (f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}]; "
                        f"{set_scalar(rdn, fw, f'(FABS{sfx}({A}) {ae[key]} FABS{sfx}({B})) ? {ones} : 0', lane=U[fw])} }}")
            raise Unimplemented(f"scalar fp three same {key}")
        if size != 3:
            raise Unimplemented("scalar int three same size")
        A, B = "a_.d[0]", "b_.d[0]"
        ones = "~(u64)0"
        e = {(0, 0b10000): f"{A} + {B}", (1, 0b10000): f"{A} - {B}",
             (1, 0b10001): f"({A} == {B}) ? {ones} : 0", (0, 0b10001): f"({A} & {B}) ? {ones} : 0",
             (0, 0b00110): f"((s64){A} > (s64){B}) ? {ones} : 0", (1, 0b00110): f"({A} > {B}) ? {ones} : 0",
             (0, 0b00111): f"((s64){A} >= (s64){B}) ? {ones} : 0", (1, 0b00111): f"({A} >= {B}) ? {ones} : 0",
             (0, 0b01000): f"(u64)SSHL64((s64){A}, (s8){B})", (1, 0b01000): f"USHL64({A}, (s8){B})"}.get((u, opcode))
        if e is None:
            raise Unimplemented("scalar int three same")
        return f"{{ V128 a_ = c->v[{rn}], b_ = c->v[{rm}]; {set_scalar(rdn, 64, e, lane='d')} }}"
    if w & 0xDF3E0C00 == 0x5E200800:  # scalar two-reg misc
        opcode = bits(w, 16, 12)
        a_bit, sz = bit(w, 23), bit(w, 22)
        fw = 64 if sz else 32
        lf, sfx, it = F[fw], FSUF[fw], UT[fw]
        A = f"a_.{lf}[0]"
        key = (u, a_bit, opcode)
        cvt = {(0, 1, 0b11011): ("Z", 1), (1, 1, 0b11011): ("Z", 0), (0, 0, 0b11011): ("M", 1),
               (1, 0, 0b11011): ("M", 0), (0, 1, 0b11010): ("P", 1), (1, 1, 0b11010): ("P", 0),
               (0, 0, 0b11010): ("N", 1), (1, 0, 0b11010): ("N", 0), (0, 0, 0b11100): ("A", 1),
               (1, 0, 0b11100): ("A", 0)}
        if key in cvt:
            r, signed = cvt[key]
            return f"{{ V128 a_ = c->v[{rn}]; {set_scalar(rdn, fw, f'({it}){f2i(fw, fw, signed, r, A)}', lane=U[fw])} }}"
        if key in ((0, 0, 0b11101), (1, 0, 0b11101)):
            lane = S[fw] if u == 0 else U[fw]
            return f"{{ V128 a_ = c->v[{rn}]; {set_scalar(rdn, fw, f'({FT[fw]})a_.{lane}[0]')} }}"
        if key == (0, 1, 0b11101):
            return f"{{ V128 a_ = c->v[{rn}]; {set_scalar(rdn, fw, f'({FT[fw]})1.0 / {A}')} }}"
        if key == (1, 1, 0b11101):
            return f"{{ V128 a_ = c->v[{rn}]; {set_scalar(rdn, fw, f'({FT[fw]})1.0 / FSQRT{sfx}({A})')} }}"
        cmpz = {(0, 1, 0b01100): ">", (0, 1, 0b01101): "==", (0, 1, 0b01110): "<",
                (1, 1, 0b01100): ">=", (1, 1, 0b01101): "<="}
        if key in cmpz:
            return (f"{{ V128 a_ = c->v[{rn}]; "
                    f"{set_scalar(rdn, fw, f'({A} {cmpz[key]} 0) ? ({it})~({it})0 : 0', lane=U[fw])} }}")
        if size == 3 and (u, opcode) == (1, 0b01011):
            return f"{{ V128 a_ = c->v[{rn}]; {set_scalar(rdn, 64, '0 - a_.d[0]', lane='d')} }}"
        if (u, a_bit, opcode) == (1, 0, 0b10110) and sz:  # FCVTXN
            return f"{{ V128 a_ = c->v[{rn}]; {set_scalar(rdn, 32, '(float)a_.fd[0]')} }}"
        raise Unimplemented(f"scalar two-reg misc {key}")
    if w & 0xDFE08400 == 0x5E000400:  # scalar copy (DUP element)
        imm5 = bits(w, 20, 16)
        sz = (imm5 & -imm5).bit_length() - 1
        esize = 8 << sz
        idx = imm5 >> (sz + 1)
        return f"{{ V128 a_ = c->v[{rn}]; {set_scalar(rdn, esize, f'a_.{U[esize]}[{idx}]', lane=U[esize])} }}"
    if w & 0xDF800400 == 0x5F000400:
        return shift_imm(w, 1, u, rn, rdn, scalar=True)
    if w & 0xDF000400 == 0x5F000000:
        return indexed(w, 1, u, size, rn, rdn, scalar=True)
    raise Unimplemented("adv simd scalar")


# ====================================================================== structure loads/stores

def ld_struct(w):
    q, l = bit(w, 30), bit(w, 22)
    post = bit(w, 23)
    rm, rn, rt = bits(w, 20, 16), bits(w, 9, 5), bits(w, 4, 0)
    opcode = bits(w, 15, 12)
    size = bits(w, 11, 10)
    base = rd(rn, 1, sp=True)
    if bit(w, 24) == 0:  # multiple structures
        if bit(w, 21) != 0 or (not post and rm != 0):
            raise Unimplemented("ld multiple encoding")
        layout = {0b0000: (4, 4), 0b0010: (1, 4), 0b0100: (3, 3), 0b0110: (1, 3),
                  0b0111: (1, 1), 0b1000: (2, 2), 0b1010: (1, 2)}.get(opcode)
        if layout is None:
            raise Unimplemented("ld multiple opcode")
        selem, rpt = layout
        nregs = selem * rpt if selem == 1 else selem
        esize = 8 << size
        regbytes = 16 if q else 8
        total = nregs * regbytes
        body = []
        if selem == 1:
            for r in range(nregs):
                t = (rt + r) % 32
                if l:
                    if q:
                        body.append(f"c->v[{t}] = LD128(a_ + {r * regbytes});")
                    else:
                        body.append(f"c->v[{t}] = V_ZERO; c->v[{t}].d[0] = LD64(a_ + {r * regbytes});")
                else:
                    body.append(f"ST128(a_ + {r * regbytes}, c->v[{t}]);" if q else
                                f"ST64(a_ + {r * regbytes}, c->v[{t}].d[0]);")
        else:
            ne = regbytes * 8 // esize
            eb = esize // 8
            lu = U[esize]
            if l:
                for r in range(selem):
                    body.append(f"V128 t{r}_ = V_ZERO;")
                for r in range(selem):
                    body.append(f"for (int i = 0; i < {ne}; i++) t{r}_.{lu}[i] = {LD_FN[eb]}(a_ + (i * {selem} + {r}) * {eb});")
                for r in range(selem):
                    body.append(f"c->v[{(rt + r) % 32}] = t{r}_;")
            else:
                for r in range(selem):
                    body.append(f"for (int i = 0; i < {ne}; i++) {ST_FN[eb]}(a_ + (i * {selem} + {r}) * {eb}, c->v[{(rt + r) % 32}].{lu}[i]);")
        wb = ""
        if post:
            wb = wr(rn, 1, f"a_ + {total}" if rm == 31 else f"a_ + c->x[{rm}]", sp=True)
        return f"{{ u64 a_ = {base}; {' '.join(body)} {wb} }}"
    # single structure
    r = bit(w, 21)
    s = bit(w, 12)
    opc3 = bits(w, 15, 13)
    selem = ((opc3 & 1) << 1 | r) + 1
    if not post and rm != 0:
        raise Unimplemented("ld single encoding")
    if opc3 == 0b110 or opc3 == 0b111:  # replicate
        if not l:
            raise Unimplemented("st replicate")
        esize = 8 << size
        eb = esize // 8
        n = (128 if q else 64) // esize
        lu = U[esize]
        body = []
        for i in range(selem):
            t = (rt + i) % 32
            body.append(f"{{ {UT[esize]} x_ = {LD_FN[eb]}(a_ + {i * eb}); V128 r_ = V_ZERO; "
                        f"for (int i = 0; i < {n}; i++) r_.{lu}[i] = x_; c->v[{t}] = r_; }}")
        total = selem * eb
    else:
        scale = opc3 >> 1
        if scale == 0:
            esize, idx = 8, (q << 3) | (s << 2) | size
        elif scale == 1:
            esize, idx = 16, (q << 2) | (s << 1) | (size >> 1)
        elif size == 0:
            esize, idx = 32, (q << 1) | s
        else:
            esize, idx = 64, q
        eb = esize // 8
        lu = U[esize]
        body = []
        for i in range(selem):
            t = (rt + i) % 32
            if l:
                body.append(f"c->v[{t}].{lu}[{idx}] = {LD_FN[eb]}(a_ + {i * eb});")
            else:
                body.append(f"{ST_FN[eb]}(a_ + {i * eb}, c->v[{t}].{lu}[{idx}]);")
        total = selem * eb
    wb = ""
    if post:
        wb = wr(rn, 1, f"a_ + {total}" if rm == 31 else f"a_ + c->x[{rm}]", sp=True)
    return f"{{ u64 a_ = {base}; {' '.join(body)} {wb} }}"
