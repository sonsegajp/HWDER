"""AArch64 -> C instruction translator.

translate(word, pc, ctx) returns a TInsn describing:
  * c:      C statements implementing the instruction (non-control-flow part)
  * kind:   control flow class (None, 'b', 'bcond', 'bl', 'br', 'blr', 'ret', 'svc', 'brk', 'udf')
  * target: static branch target if any
  * cond:   C boolean expression for conditional branches

The generated C relies on runtime/include/hwder/cpu.h. Guest memory is identity
mapped (guest address == host address), so loads/stores are plain pointer accesses.
"""
import struct


class Unimplemented(Exception):
    pass


class TInsn:
    __slots__ = ("c", "kind", "target", "cond", "reg", "link")

    def __init__(self, c="", kind=None, target=None, cond=None, reg=None, link=False):
        self.c, self.kind, self.target, self.cond, self.reg, self.link = c, kind, target, cond, reg, link


def bits(w, hi, lo):
    return (w >> lo) & ((1 << (hi - lo + 1)) - 1)


def bit(w, n):
    return (w >> n) & 1


def sext(v, n):
    return v - (1 << n) if v & (1 << (n - 1)) else v


def mask(n):
    return (1 << n) - 1


def hx(v):
    return f"0x{v:X}ULL"


def hx32(v):
    return f"0x{v:X}u"


COND_NAMES = ["eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "nv"]


def cond_expr(cond):
    base = cond >> 1
    e = ["c->zf", "c->cf", "c->nf", "c->vf",
         "(c->cf && !c->zf)", "(c->nf == c->vf)", "(!c->zf && c->nf == c->vf)", "1"][base]
    if cond & 1 and base != 7:
        return f"!{e}"
    return e


# ------------------------------------------------------------------ register helpers

def rd(n, sf, sp=False):
    """Read general register n (X if sf else W). Register 31 is SP or ZR."""
    if n == 31:
        if sp:
            return "c->sp" if sf else "(u32)c->sp"
        return "(u64)0" if sf else "(u32)0"
    return f"c->x[{n}]" if sf else f"(u32)c->x[{n}]"


def wr(n, sf, val, sp=False):
    if n == 31 and not sp:
        return f"(void)({val});"
    dst = "c->sp" if n == 31 else f"c->x[{n}]"
    if sf:
        return f"{dst} = (u64)({val});"
    return f"{dst} = (u32)({val});"


def T(sf):
    return "u64" if sf else "u32"


def ST(sf):
    return "s64" if sf else "s32"


def shift_reg(n, sf, stype, amt):
    r = rd(n, sf)
    if amt == 0:
        return r
    t, s = T(sf), ST(sf)
    w = 64 if sf else 32
    if stype == 0:
        return f"(({t}){r} << {amt})"
    if stype == 1:
        return f"(({t}){r} >> {amt})"
    if stype == 2:
        return f"(({t})(({s}){r} >> {amt}))"
    return f"ROR{w}({r}, {amt})"


def extend_reg(n, sf, option, amt):
    """Extended register operand for add/sub (extended) and register-offset addressing."""
    t = T(sf)
    r = f"c->x[{n}]" if n != 31 else "(u64)0"
    e = [f"(u8){r}", f"(u16){r}", f"(u32){r}", f"(u64){r}",
         f"(s8){r}", f"(s16){r}", f"(s32){r}", f"(s64){r}"][option]
    e = f"(({t})({e}))"
    if amt:
        e = f"({e} << {amt})"
    return e


def highest_set_bit(v):
    return v.bit_length() - 1


def ror(v, r, w):
    r %= w
    return ((v >> r) | (v << (w - r))) & mask(w) if r else v


def decode_bit_masks(n, imms, immr, m, immediate=True):
    length = highest_set_bit((n << 6) | (~imms & 0x3F))
    if length < 1:
        raise Unimplemented("reserved bitmask")
    levels = mask(length)
    if immediate and (imms & levels) == levels:
        raise Unimplemented("reserved bitmask")
    s = imms & levels
    r = immr & levels
    diff = (s - r) & 0x3F
    esize = 1 << length
    d = diff & levels
    welem = mask(s + 1)
    telem = mask(d + 1)
    wmask = 0
    tmask = 0
    welem = ror(welem, r, esize)
    for i in range(m // esize):
        wmask |= welem << (i * esize)
        tmask |= telem << (i * esize)
    return wmask, tmask


# ------------------------------------------------------------------ memory helpers

LD_FN = {1: "LD8", 2: "LD16", 4: "LD32", 8: "LD64"}
LDS_FN = {1: "LDS8", 2: "LDS16", 4: "LDS32"}
ST_FN = {1: "ST8", 2: "ST16", 4: "ST32", 8: "ST64"}


def fp_load(t, size, addr):
    """Load SIMD&FP register t of `size` bytes, zeroing the rest."""
    if size == 16:
        return f"c->v[{t}] = LD128({addr});"
    return f"c->v[{t}] = V_ZERO; c->v[{t}].{ {1: 'b', 2: 'h', 4: 's', 8: 'd'}[size] }[0] = {LD_FN[size]}({addr});"


def fp_store(t, size, addr):
    if size == 16:
        return f"ST128({addr}, c->v[{t}]);"
    return f"{ST_FN[size]}({addr}, c->v[{t}].{ {1: 'b', 2: 'h', 4: 's', 8: 'd'}[size] }[0]);"


# ------------------------------------------------------------------ translator

class Translator:
    def __init__(self):
        from . import a64_simd
        self.simd = a64_simd

    def translate(self, w, pc):
        op0 = bits(w, 28, 25)
        if op0 in (0b1000, 0b1001):
            return TInsn(self.dp_imm(w, pc))
        if op0 in (0b1010, 0b1011):
            return self.branch_sys(w, pc)
        if op0 & 0b0101 == 0b0100:
            return TInsn(self.ldst(w, pc))
        if op0 & 0b0111 == 0b0101:
            return TInsn(self.dp_reg(w, pc))
        if op0 & 0b0111 == 0b0111:
            return TInsn(self.simd.translate(w, pc))
        if w == 0 or bits(w, 31, 16) == 0:
            return TInsn("", kind="udf")
        raise Unimplemented("unallocated")

    # ---------------------------------------------------------- data processing immediate
    def dp_imm(self, w, pc):
        op = bits(w, 25, 23)
        rdn = bits(w, 4, 0)
        rn = bits(w, 9, 5)
        sf = bit(w, 31)
        if op in (0, 1):  # ADR / ADRP
            imm = sext((bits(w, 23, 5) << 2) | bits(w, 30, 29), 21)
            if bit(w, 31):
                val = ((pc & ~0xFFF) + (imm << 12)) & mask(64)
            else:
                val = (pc + imm) & mask(64)
            return wr(rdn, 1, hx(val))
        if op == 2:  # add/sub immediate
            opc, s, sh = bit(w, 30), bit(w, 29), bit(w, 22)
            imm = bits(w, 21, 10) << (12 if sh else 0)
            a = rd(rn, sf, sp=True)
            t = T(sf)
            if s:
                fn = ("SUBS" if opc else "ADDS") + ("64" if sf else "32")
                return wr(rdn, sf, f"{fn}(c, {a}, ({t}){hx(imm)})")
            opr = "-" if opc else "+"
            if imm == 0:
                return wr(rdn, sf, a, sp=True)
            return wr(rdn, sf, f"({t})({a} {opr} ({t}){hx(imm)})", sp=True)
        if op == 4:  # logical immediate
            opc = bits(w, 30, 29)
            n = bit(w, 22)
            if not sf and n:
                raise Unimplemented("logical imm N=1 32-bit")
            m = 64 if sf else 32
            imm, _ = decode_bit_masks(n, bits(w, 15, 10), bits(w, 21, 16), m)
            t = T(sf)
            a = rd(rn, sf)
            k = f"({t}){hx(imm)}"
            if opc == 0:
                return wr(rdn, sf, f"{a} & {k}", sp=True)
            if opc == 1:
                return wr(rdn, sf, f"{a} | {k}", sp=True)
            if opc == 2:
                return wr(rdn, sf, f"{a} ^ {k}", sp=True)
            return wr(rdn, sf, f"LOGICFLAGS{m}(c, {a} & {k})")
        if op == 5:  # move wide
            opc = bits(w, 30, 29)
            hw = bits(w, 22, 21)
            if not sf and hw > 1:
                raise Unimplemented("movw hw")
            imm = bits(w, 20, 5) << (16 * hw)
            m = 64 if sf else 32
            if opc == 0:
                return wr(rdn, sf, hx((~imm) & mask(m)))
            if opc == 2:
                return wr(rdn, sf, hx(imm))
            if opc == 3:
                keep = (~(0xFFFF << (16 * hw))) & mask(m)
                return wr(rdn, sf, f"({rd(rdn, sf)} & {hx(keep)}) | {hx(imm)}")
            raise Unimplemented("movw opc")
        if op == 6:  # bitfield
            return self.bitfield(w)
        if op == 7:  # EXTR
            if bit(w, 22) != sf or bit(w, 21):
                raise Unimplemented("extr")
            rm = bits(w, 20, 16)
            lsb = bits(w, 15, 10)
            m = 64 if sf else 32
            if lsb == 0:
                return wr(rdn, sf, rd(rm, sf))
            if rn == rm:
                return wr(rdn, sf, f"ROR{m}({rd(rn, sf)}, {lsb})")
            t = T(sf)
            return wr(rdn, sf, f"(({t}){rd(rm, sf)} >> {lsb}) | (({t}){rd(rn, sf)} << {m - lsb})")
        raise Unimplemented("dp_imm")

    def bitfield(self, w):
        sf = bit(w, 31)
        opc = bits(w, 30, 29)
        n = bit(w, 22)
        immr = bits(w, 21, 16)
        imms = bits(w, 15, 10)
        rn, rdn = bits(w, 9, 5), bits(w, 4, 0)
        if n != sf:
            raise Unimplemented("bitfield N")
        m = 64 if sf else 32
        t, s = T(sf), ST(sf)
        src = rd(rn, sf)
        if opc == 2:  # UBFM
            if imms >= immr:
                width = imms - immr + 1
                e = f"(({t}){src} >> {immr})" if immr else f"({t}){src}"
                if width < m:
                    e = f"({e} & {hx(mask(width))})"
                return wr(rdn, sf, e)
            width = imms + 1
            return wr(rdn, sf, f"((({t}){src} & {hx(mask(width))}) << {m - immr})")
        if opc == 0:  # SBFM
            if imms >= immr:
                sh = m - 1 - imms
                return wr(rdn, sf, f"({t})((({s})(({t}){src} << {sh})) >> {sh + immr})")
            sh = m - 1 - imms
            return wr(rdn, sf, f"(({t})((({s})(({t}){src} << {sh})) >> {sh}) << {m - immr})")
        if opc == 1:  # BFM
            dst = rd(rdn, sf)
            if imms >= immr:
                width = imms - immr + 1
                mk = mask(width)
                return wr(rdn, sf, f"({dst} & ({t})~({t}){hx(mk)}) | ((({t}){src} >> {immr}) & {hx(mk)})")
            width = imms + 1
            lsb = m - immr
            mk = mask(width) << lsb
            return wr(rdn, sf, f"({dst} & ({t})~({t}){hx(mk)}) | ((({t}){src} << {lsb}) & {hx(mk & mask(m))})")
        raise Unimplemented("bitfield opc")

    # ---------------------------------------------------------- branches / system
    def branch_sys(self, w, pc):
        if w & 0x7C000000 == 0x14000000:  # B / BL
            tgt = (pc + (sext(bits(w, 25, 0), 26) << 2)) & mask(64)
            if bit(w, 31):
                return TInsn(f"c->x[30] = {hx(pc + 4)};", kind="bl", target=tgt, link=True)
            return TInsn("", kind="b", target=tgt)
        if w & 0xFF000010 == 0x54000000:  # B.cond
            tgt = (pc + (sext(bits(w, 23, 5), 19) << 2)) & mask(64)
            cond = bits(w, 3, 0)
            if cond >= 14:
                return TInsn("", kind="b", target=tgt)
            return TInsn("", kind="bcond", target=tgt, cond=cond_expr(cond))
        if w & 0x7E000000 == 0x34000000:  # CBZ / CBNZ
            sf = bit(w, 31)
            tgt = (pc + (sext(bits(w, 23, 5), 19) << 2)) & mask(64)
            r = rd(bits(w, 4, 0), sf)
            return TInsn("", kind="bcond", target=tgt, cond=f"({r} {'!=' if bit(w, 24) else '=='} 0)")
        if w & 0x7E000000 == 0x36000000:  # TBZ / TBNZ
            b = (bit(w, 31) << 5) | bits(w, 23, 19)
            tgt = (pc + (sext(bits(w, 18, 5), 14) << 2)) & mask(64)
            r = rd(bits(w, 4, 0), 1)
            return TInsn("", kind="bcond", target=tgt,
                         cond=f"((({r} >> {b}) & 1) {'!=' if bit(w, 24) else '=='} 0)")
        if w & 0xFE000000 == 0xD6000000:  # BR / BLR / RET
            opc = bits(w, 24, 21)
            rn = bits(w, 9, 5)
            if bits(w, 20, 16) != 0x1F or bits(w, 15, 10) != 0 or bits(w, 4, 0) != 0:
                raise Unimplemented("branch reg (pauth)")
            if opc == 0:
                return TInsn("", kind="br", reg=rn)
            if opc == 1:
                return TInsn(f"c->x[30] = {hx(pc + 4)};", kind="blr", reg=rn, link=True)
            if opc == 2:
                return TInsn("", kind="ret", reg=rn)
            raise Unimplemented("eret/drps")
        if w & 0xFF000000 == 0xD4000000:
            opc = bits(w, 23, 21)
            ll = bits(w, 1, 0)
            imm = bits(w, 20, 5)
            if opc == 0 and ll == 1:
                return TInsn(f"hw_svc(c, {imm});", kind="svc", target=imm)
            if opc == 1 and ll == 0:
                return TInsn(f"hw_brk(c, {hx(pc)}, {imm});", kind="brk", target=imm)
            raise Unimplemented("exception")
        if w & 0xFFC00000 == 0xD5000000:
            return TInsn(self.system(w, pc))
        raise Unimplemented("branch/sys")

    def system(self, w, pc):
        if w & 0xFFFFF01F == 0xD503201F:  # hints (NOP, YIELD, WFE, ...)
            crm_op2 = bits(w, 11, 5)
            if crm_op2 == 1:
                return "hw_yield();"
            return ""
        if w & 0xFFFFF0FF == 0xD503305F:
            return "c->excl_addr = ~(u64)0;"  # CLREX
        if w & 0xFFFFF09F == 0xD503309F:  # DSB / DMB / ISB
            if bits(w, 6, 5) in (0, 1):  # DSB/DMB
                return "HW_FENCE();"
            return ""
        if w & 0xFFF00000 == 0xD5300000 or w & 0xFFF00000 == 0xD5100000:
            l = bit(w, 21)
            sysreg = bits(w, 20, 5)
            rt = bits(w, 4, 0)
            name = {
                0xDE82: "tpidr_el0", 0xDE83: "tpidrro_el0",
                0xDA20: "fpcr", 0xDA21: "fpsr",
                0xDF02: "cntvct", 0xDF00: "cntfrq", 0xDF01: "cntpct",
                0xC006: "midr", 0xC005: "mpidr", 0xD807: "dczid",
                0xDA10: "nzcv", 0xD801: "ctr",
            }.get(sysreg)
            if name is None:
                raise Unimplemented(f"sysreg {sysreg:#x}")
            if name == "nzcv":
                if l:
                    return wr(rt, 1, "((u64)c->nf << 31) | ((u64)c->zf << 30) | ((u64)c->cf << 29) | ((u64)c->vf << 28)")
                return f"SETNZCV(c, (u32)({rd(rt, 1)} >> 28));"
            if name == "ctr" and l:
                return wr(rt, 1, "0x8444C004")  # Cortex-A57 cache type: 64-byte lines
            if l:
                if name in ("cntvct", "cntpct"):
                    return wr(rt, 1, "hw_cntvct()")
                if name == "cntfrq":
                    return wr(rt, 1, "19200000")
                if name == "dczid":
                    return wr(rt, 1, "4")  # 64-byte DC ZVA blocks
                if name in ("midr", "mpidr"):
                    return wr(rt, 1, "0x410FD071" if name == "midr" else "0x80000000")
                return wr(rt, 1, f"c->{name}")
            if name in ("tpidr_el0", "fpcr", "fpsr"):
                return f"c->{name} = {rd(rt, 1)};"
            raise Unimplemented(f"msr {name}")
        if w & 0xFFF80000 == 0xD5080000:  # SYS (DC/IC cache maintenance)
            op1, crn, crm, op2 = bits(w, 18, 16), bits(w, 15, 12), bits(w, 11, 8), bits(w, 7, 5)
            rt = bits(w, 4, 0)
            if crn == 7 and crm == 4 and op1 == 3 and op2 == 1:  # DC ZVA
                return f"memset((void*)(uintptr_t)({rd(rt, 1)} & ~(u64)63), 0, 64);"
            return ""  # other cache maintenance ops are no-ops on the host
        if w & 0xFFF8F01F == 0xD500401F:  # MSR (immediate) - PSTATE fields
            return ""
        raise Unimplemented("system")

    # ---------------------------------------------------------- loads / stores
    def ldst(self, w, pc):
        if w & 0x3F000000 == 0x08000000:
            return self.ld_excl(w)
        if w & 0x3B000000 == 0x18000000:
            return self.ld_literal(w, pc)
        if w & 0x3A000000 == 0x28000000:
            return self.ld_pair(w)
        if w & 0x3B000000 == 0x39000000 or w & 0x3B200000 == 0x38000000 or w & 0x3B200C00 == 0x38200800:
            return self.ld_reg(w)
        if w & 0xBF000000 == 0x0C000000 or w & 0xBF000000 == 0x0D000000:
            return self.simd.ld_struct(w)
        if w & 0x3B200C00 == 0x38200000:
            return self.ld_atomic(w)
        raise Unimplemented("ldst")

    def ld_excl(self, w):
        size = bits(w, 31, 30)
        o2, l, o1, o0 = bit(w, 23), bit(w, 22), bit(w, 21), bit(w, 15)
        rs, rt2, rn, rt = bits(w, 20, 16), bits(w, 14, 10), bits(w, 9, 5), bits(w, 4, 0)
        nb = 1 << size
        sf = size == 3
        addr = rd(rn, 1, sp=True)
        if o2 == 1 and o1 == 0:  # LDAR / STLR / LDLAR / STLLR
            if l:
                return wr(rt, sf, f"ATOMIC_LOAD{nb * 8}({addr})")
            return f"ATOMIC_STORE{nb * 8}({addr}, {rd(rt, sf)});"
        if o2 == 0 and o1 == 0:  # LDXR/LDAXR, STXR/STLXR
            if l:
                return (f"{{ u64 a_ = {addr}; u64 v_ = ATOMIC_LOAD{nb * 8}(a_); "
                        f"c->excl_addr = a_; c->excl_val = v_; {wr(rt, sf, 'v_')} }}")
            return (f"{{ u64 a_ = {addr}; "
                    f"{wr(rs, 0, f'!EXCL_STORE{nb * 8}(c, a_, {rd(rt, sf)})')} }}")
        if o2 == 0 and o1 == 1:  # LDXP / STXP
            if size not in (2, 3):
                raise Unimplemented("excl pair size")
            ew = 32 if size == 2 else 64
            if l:
                if rt == 31 or rt2 == 31:
                    raise Unimplemented("ldxp zr")
                return f"{{ u64 a_ = {addr}; EXCL_LOADP{ew}(c, a_, &c->x[{rt}], &c->x[{rt2}]); }}"
            return (f"{{ u64 a_ = {addr}; "
                    f"{wr(rs, 0, f'!EXCL_STOREP{ew}(c, a_, {rd(rt, size == 3)}, {rd(rt2, size == 3)})')} }}")
        raise Unimplemented("ld_excl")

    def ld_atomic(self, w):
        """ARMv8.1 LSE atomics (LDADD, SWP, ...) - rare in Switch titles."""
        raise Unimplemented("lse atomic")

    def ld_literal(self, w, pc):
        opc, v = bits(w, 31, 30), bit(w, 26)
        rt = bits(w, 4, 0)
        addr = hx((pc + (sext(bits(w, 23, 5), 19) << 2)) & mask(64))
        if v:
            size = {0: 4, 1: 8, 2: 16}[opc]
            return fp_load(rt, size, addr)
        if opc == 0:
            return wr(rt, 0, f"LD32({addr})")
        if opc == 1:
            return wr(rt, 1, f"LD64({addr})")
        if opc == 2:
            return wr(rt, 1, f"(s64)LDS32({addr})")
        return ""  # PRFM literal

    def ld_pair(self, w):
        opc, v, mode, l = bits(w, 31, 30), bit(w, 26), bits(w, 24, 23), bit(w, 22)
        imm7 = sext(bits(w, 21, 15), 7)
        rt2, rn, rt = bits(w, 14, 10), bits(w, 9, 5), bits(w, 4, 0)
        if v:
            size = {0: 4, 1: 8, 2: 16}[opc]
        else:
            size = 8 if opc == 2 else 4
        off = imm7 * size
        base = rd(rn, 1, sp=True)
        if mode == 1:  # post-index
            ea = "a_"
            pre = f"u64 a_ = {base};"
            post = wr(rn, 1, f"a_ + {off}", sp=True)
        elif mode == 3:  # pre-index
            ea = "a_"
            pre = f"u64 a_ = {base} + {off};"
            post = wr(rn, 1, "a_", sp=True)
        else:  # signed offset / non-temporal
            ea = "a_"
            pre = f"u64 a_ = {base}" + (f" + {off};" if off else ";")
            post = ""
        a1, a2 = ea, f"{ea} + {size}"
        if v:
            if l:
                body = fp_load(rt, size, a1) + " " + fp_load(rt2, size, a2)
            else:
                body = fp_store(rt, size, a1) + " " + fp_store(rt2, size, a2)
        else:
            sf = size == 8
            if l:
                if opc == 1:  # LDPSW
                    body = wr(rt, 1, f"(s64)LDS32({a1})") + " " + wr(rt2, 1, f"(s64)LDS32({a2})")
                else:
                    fn = LD_FN[size]
                    # Load both before writing back so Rt == Rn behaves.
                    body = (f"{T(sf)} v1_ = {fn}({a1}), v2_ = {fn}({a2}); "
                            + wr(rt, sf, "v1_") + " " + wr(rt2, sf, "v2_"))
            else:
                fn = ST_FN[size]
                body = f"{fn}({a1}, {rd(rt, sf)}); {fn}({a2}, {rd(rt2, sf)});"
        return f"{{ {pre} {body} {post} }}"

    def ld_reg(self, w):
        size, v, opc = bits(w, 31, 30), bit(w, 26), bits(w, 23, 22)
        rn, rt = bits(w, 9, 5), bits(w, 4, 0)
        base = rd(rn, 1, sp=True)
        # Determine access size and kind
        if v:
            nbytes = 16 if (size == 0 and opc & 2) else (1 << size)
            is_load = bool(opc & 1)
        else:
            nbytes = 1 << size
            is_load = opc != 0
        # Addressing
        pre, post, ea = "", "", "a_"
        if bits(w, 25, 24) == 1:  # unsigned immediate
            off = bits(w, 21, 10) * nbytes
            pre = f"u64 a_ = {base}" + (f" + {off};" if off else ";")
        elif bit(w, 21) == 0:
            imm9 = sext(bits(w, 20, 12), 9)
            idx = bits(w, 11, 10)
            if idx == 0 or idx == 2:  # unscaled / unprivileged
                pre = f"u64 a_ = {base}" + (f" + ({imm9});" if imm9 else ";")
            elif idx == 1:  # post
                pre = f"u64 a_ = {base};"
                post = wr(rn, 1, f"a_ + ({imm9})", sp=True)
            else:  # pre
                pre = f"u64 a_ = {base} + ({imm9});"
                post = wr(rn, 1, "a_", sp=True)
        else:  # register offset
            rm = bits(w, 20, 16)
            option = bits(w, 15, 13)
            s = bit(w, 12)
            amt = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[nbytes] if s else 0
            pre = f"u64 a_ = {base} + {extend_reg(rm, 1, option, amt)};"
        if v:
            body = fp_load(rt, nbytes, ea) if is_load else fp_store(rt, nbytes, ea)
        else:
            if opc == 0:
                body = f"{ST_FN[nbytes]}({ea}, {rd(rt, nbytes == 8)});"
            elif opc == 1:
                body = wr(rt, nbytes == 8, f"{LD_FN[nbytes]}({ea})")
            elif size == 3:  # PRFM / PRFUM
                return ""
            elif opc == 2:  # sign-extend to 64
                body = wr(rt, 1, f"(s64){LDS_FN[nbytes]}({ea})")
            else:  # sign-extend to 32
                body = wr(rt, 0, f"(s32){LDS_FN[nbytes]}({ea})")
        return f"{{ {pre} {body} {post} }}".replace("  ", " ")

    # ---------------------------------------------------------- data processing register
    def dp_reg(self, w, pc):
        sf = bit(w, 31)
        rm, rn, rdn = bits(w, 20, 16), bits(w, 9, 5), bits(w, 4, 0)
        t = T(sf)
        m = 64 if sf else 32
        if w & 0x1F000000 == 0x0A000000:  # logical (shifted register)
            opc = bits(w, 30, 29)
            stype, n, amt = bits(w, 23, 22), bit(w, 21), bits(w, 15, 10)
            if not sf and amt >= 32:
                raise Unimplemented("logical shift amt")
            b = shift_reg(rm, sf, stype, amt)
            if n:
                b = f"({t})~{b}"
            a = rd(rn, sf)
            if opc == 1 and rn == 31 and not n and amt == 0:  # MOV
                return wr(rdn, sf, rd(rm, sf))
            e = {0: f"{a} & {b}", 1: f"{a} | {b}", 2: f"{a} ^ {b}", 3: f"{a} & {b}"}[opc]
            if opc == 3:
                return wr(rdn, sf, f"LOGICFLAGS{m}(c, {e})")
            return wr(rdn, sf, e)
        if w & 0x1F200000 == 0x0B000000:  # add/sub (shifted register)
            op, s = bit(w, 30), bit(w, 29)
            stype, amt = bits(w, 23, 22), bits(w, 15, 10)
            if stype == 3:
                raise Unimplemented("add shift ror")
            b = shift_reg(rm, sf, stype, amt)
            a = rd(rn, sf)
            return self._addsub(sf, op, s, rdn, a, b, sp_dest=False)
        if w & 0x1F200000 == 0x0B200000:  # add/sub (extended register)
            op, s = bit(w, 30), bit(w, 29)
            option, amt = bits(w, 15, 13), bits(w, 12, 10)
            if amt > 4:
                raise Unimplemented("ext amt")
            b = extend_reg(rm, sf, option, amt)
            a = rd(rn, sf, sp=True)
            return self._addsub(sf, op, s, rdn, a, b, sp_dest=not s)
        if w & 0x1FE00000 == 0x1A000000:  # ADC / SBC
            op, s = bit(w, 30), bit(w, 29)
            a, b = rd(rn, sf), rd(rm, sf)
            if op:
                b = f"({t})~{b}"
            if s:
                return wr(rdn, sf, f"ADC{m}(c, {a}, {b}, c->cf)")
            return wr(rdn, sf, f"({t})({a} + {b} + ({t})c->cf)")
        if w & 0x1FE00000 == 0x1A400000:  # CCMN / CCMP
            op = bit(w, 30)
            cond = bits(w, 15, 12)
            nzcv = bits(w, 3, 0)
            a = rd(rn, sf)
            b = f"({t}){bits(w, 20, 16)}" if bit(w, 11) else rd(rm, sf)
            fn = ("SUBS" if op else "ADDS") + str(m)
            return (f"if ({cond_expr(cond)}) {{ (void){fn}(c, {a}, {b}); }} "
                    f"else {{ SETNZCV(c, {nzcv}); }}")
        if w & 0x1FE00000 == 0x1A800000:  # conditional select
            op, op2 = bit(w, 30), bits(w, 11, 10)
            cond = bits(w, 15, 12)
            a, b = rd(rn, sf), rd(rm, sf)
            if op == 0 and op2 == 0:
                e = b
            elif op == 0 and op2 == 1:
                e = f"({t})({b} + 1)"
            elif op == 1 and op2 == 0:
                e = f"({t})~{b}"
            elif op == 1 and op2 == 1:
                e = f"({t})(0 - {b})"
            else:
                raise Unimplemented("csel op2")
            return wr(rdn, sf, f"({cond_expr(cond)}) ? ({t}){a} : ({t}){e}")
        if w & 0x5FE00000 == 0x1AC00000:  # 2-source
            opcode = bits(w, 15, 10)
            a, b = rd(rn, sf), rd(rm, sf)
            s = ST(sf)
            if opcode == 2:
                return wr(rdn, sf, f"UDIV{m}({a}, {b})")
            if opcode == 3:
                return wr(rdn, sf, f"SDIV{m}({a}, {b})")
            if opcode == 8:
                return wr(rdn, sf, f"({t}){a} << ({b} & {m - 1})")
            if opcode == 9:
                return wr(rdn, sf, f"({t}){a} >> ({b} & {m - 1})")
            if opcode == 10:
                return wr(rdn, sf, f"({t})(({s}){a} >> ({b} & {m - 1}))")
            if opcode == 11:
                return wr(rdn, sf, f"ROR{m}({a}, {b} & {m - 1})")
            if 16 <= opcode <= 23:  # CRC32{B,H,W,X} / CRC32C{B,H,W,X}: Wd = crc(Wn, Rm)
                nbytes = 1 << (opcode & 3)
                fn = "CRC32C" if opcode >= 20 else "CRC32"
                data = rd(rm, nbytes == 8)
                return wr(rdn, 0, f"{fn}({rd(rn, 0)}, {data}, {nbytes})")
            raise Unimplemented(f"dp2 {opcode}")
        if w & 0x5FE00000 == 0x5AC00000:  # 1-source
            opcode = bits(w, 15, 10)
            a = rd(rn, sf)
            if opcode == 0:
                return wr(rdn, sf, f"RBIT{m}({a})")
            if opcode == 1:
                return wr(rdn, sf, f"REV16_{m}({a})")
            if opcode == 2:
                return wr(rdn, sf, f"REV32_{m}({a})" if sf else f"BSWAP32({a})")
            if opcode == 3 and sf:
                return wr(rdn, sf, f"BSWAP64({a})")
            if opcode == 4:
                return wr(rdn, sf, f"CLZ{m}({a})")
            if opcode == 5:
                return wr(rdn, sf, f"CLS{m}({a})")
            raise Unimplemented(f"dp1 {opcode}")
        if w & 0x1F000000 == 0x1B000000:  # 3-source
            op31, o0 = bits(w, 23, 21), bit(w, 15)
            ra = bits(w, 14, 10)
            if op31 == 0:
                a, b, acc = rd(rn, sf), rd(rm, sf), rd(ra, sf)
                if ra == 31:
                    return wr(rdn, sf, f"({t})({'0 - ' if o0 else ''}({t}){a} * ({t}){b})")
                return wr(rdn, sf, f"({t})({acc} {'-' if o0 else '+'} ({t}){a} * ({t}){b})")
            if op31 in (1, 5) and sf:
                if op31 == 1:
                    prod = f"(u64)((s64)(s32){rd(rn, 1)} * (s64)(s32){rd(rm, 1)})"
                else:
                    prod = f"((u64)(u32){rd(rn, 1)} * (u64)(u32){rd(rm, 1)})"
                acc = rd(ra, 1)
                return wr(rdn, 1, f"{acc} {'-' if o0 else '+'} {prod}")
            if op31 == 2 and sf:
                return wr(rdn, 1, f"SMULH({rd(rn, 1)}, {rd(rm, 1)})")
            if op31 == 6 and sf:
                return wr(rdn, 1, f"UMULH({rd(rn, 1)}, {rd(rm, 1)})")
            raise Unimplemented("dp3")
        raise Unimplemented("dp_reg")

    def _addsub(self, sf, op, s, rdn, a, b, sp_dest):
        t = T(sf)
        m = 64 if sf else 32
        if s:
            fn = ("SUBS" if op else "ADDS") + str(m)
            return wr(rdn, sf, f"{fn}(c, {a}, {b})")
        if op and a in ("(u64)0", "(u32)0"):  # NEG
            return wr(rdn, sf, f"({t})(0 - {b})", sp=sp_dest)
        return wr(rdn, sf, f"({t})({a} {'-' if op else '+'} {b})", sp=sp_dest)
