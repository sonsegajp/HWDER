"""Differential testing of the AArch64->C translator against Unicorn.

For a sample of real instruction encodings taken from the game's main module:
  1. translate each to C and compile all of them into a test DLL,
  2. map the game image and a scratch buffer at identical addresses in this process
     and in Unicorn,
  3. run every instruction from several random register states in both, and compare
     all general/vector registers, NZCV, the next PC and the scratch memory.

Usage: python -m recomp.difftest [--per-mnemonic N] [--states N] [--filter MNEMONIC]
"""
import argparse
import collections
import ctypes
import os
import random
import struct
import subprocess
import sys

import capstone
from unicorn import Uc, UC_ARCH_ARM64, UC_MODE_ARM, UcError
from unicorn import arm64_const as A

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nso import Process  # noqa: E402
from recomp.a64 import Translator, Unimplemented, hx  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "build", "difftest")
VCVARS = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

SCRATCH = 0x10000000
SCRATCH_SIZE = 0x100000
MID = SCRATCH + SCRATCH_SIZE // 2

SKIP = {"bl", "b", "blr", "br", "ret", "svc", "brk", "ldaxr", "stlxr", "ldxr", "stxr", "ldaxrh", "stlxrh",
        "ldaxrb", "stlxrb", "ldaxp", "stlxp", "clrex", "dmb", "dsb", "isb", "msr", "mrs", "hint", "nop",
        "yield", "dc", "ic", "sys", "prfm", "prfum", "udf", "frecpe", "frsqrte"}


class V128(ctypes.Union):
    _fields_ = [("b", ctypes.c_uint8 * 16), ("d", ctypes.c_uint64 * 2)]


class Ctx(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint64 * 32), ("sp", ctypes.c_uint64), ("v", V128 * 32),
                ("nf", ctypes.c_uint8), ("zf", ctypes.c_uint8), ("cf", ctypes.c_uint8), ("vf", ctypes.c_uint8),
                ("fpcr", ctypes.c_uint32), ("fpsr", ctypes.c_uint32),
                ("tpidr_el0", ctypes.c_uint64), ("tpidrro_el0", ctypes.c_uint64),
                ("excl_addr", ctypes.c_uint64), ("excl_val", ctypes.c_uint64), ("excl_val2", ctypes.c_uint64),
                ("thread", ctypes.c_void_p)]


def host_map(addr, size, data=b""):
    k32 = ctypes.windll.kernel32
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    p = k32.VirtualAlloc(addr, size, 0x3000, 0x04)
    if p != addr:
        raise RuntimeError(f"could not map host memory at {addr:#x} (got {p})")
    if data:
        ctypes.memmove(addr, bytes(data), len(data))


def is_ldst(w):
    return (w >> 25) & 0b0101 == 0b0100


FP_SPECIALS32 = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000,
                 0x00000001, 0x4F000000, 0xCF000000, 0x4F800000, 0x3F000000, 0x3FC00000, 0x40200000]
INT_SPECIALS = [0, 1, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFF, 0x7FFFFFFF,
                0x80000000, 0x100000000, 0xFF, 0xFFFF]


def rand_float32(rng):
    r = rng.random()
    if r < 0.15:
        return rng.choice(FP_SPECIALS32)
    if r < 0.6:
        return struct.unpack("<I", struct.pack("<f", rng.uniform(-1000, 1000)))[0]
    if r < 0.8:
        return struct.unpack("<I", struct.pack("<f", rng.uniform(-2, 2)))[0]
    return rng.getrandbits(32)


def rand_float64(rng):
    r = rng.random()
    if r < 0.6:
        return struct.unpack("<Q", struct.pack("<d", rng.uniform(-1e6, 1e6)))[0]
    if r < 0.7:
        return rng.choice([0, 1 << 63, 0x3FF0000000000000, 0x7FF0000000000000, 0x7FF8000000000000])
    return rng.getrandbits(64)


def rand_vreg(rng):
    r = rng.random()
    if r < 0.5:
        return sum(rand_float32(rng) << (32 * i) for i in range(4))
    if r < 0.7:
        return rand_float64(rng) | (rand_float64(rng) << 64)
    return rng.getrandbits(128)


def rand_state(rng, w):
    st = {"x": [0] * 31, "v": [0] * 32}
    mem = is_ldst(w)
    for i in range(31):
        if mem:
            st["x"][i] = rng.randrange(0, 64) * 8
        else:
            r = rng.random()
            st["x"][i] = rng.choice(INT_SPECIALS) if r < 0.2 else (rng.getrandbits(64) if r < 0.7 else rng.getrandbits(rng.choice([8, 16, 32])))
    if not mem and rng.random() < 0.4:  # make some registers equal to exercise ==/carry edges
        v = st["x"][rng.randrange(31)]
        for i in rng.sample(range(31), 12):
            st["x"][i] = v
    if rng.random() < 0.3:
        v = st["v"][rng.randrange(32)]
        for i in rng.sample(range(32), 12):
            st["v"][i] = v
    if mem:
        rn = (w >> 5) & 31
        if rn != 31:
            st["x"][rn] = MID + rng.randrange(-64, 64) * 16
    for i in range(32):
        st["v"][i] = rand_vreg(rng)
    st["sp"] = MID + rng.randrange(-64, 64) * 16
    st["nzcv"] = rng.getrandbits(4)
    return st


def collect_samples(m, per_mnemonic, flt):
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    seen = set()
    buckets = collections.defaultdict(list)
    rng = random.Random(1234)
    words = []
    for off in range(m.text_off, m.text_off + m.text_size, 4):
        w = struct.unpack_from("<I", m.image, off)[0]
        if w in seen:
            continue
        seen.add(w)
        words.append((m.base + off, w))
    rng.shuffle(words)
    for pc, w in words:
        r = list(md.disasm_lite(struct.pack("<I", w), pc))
        if not r:
            continue
        mn = r[0][2].split(".")[0] if not r[0][2].startswith("b.") else "b.cond"
        if mn in SKIP or (flt and mn not in flt):
            continue
        if len(buckets[mn]) < per_mnemonic:
            buckets[mn].append((pc, w, f"{r[0][2]} {r[0][3]}"))
    return buckets


def build_dll(tests):
    os.makedirs(BUILD, exist_ok=True)
    src = os.path.join(BUILD, "difftest.c")
    tr = Translator()
    lines = ['#include "hwder/cpu.h"',
             "u64 hw_cntvct(void) { return 0; } void hw_yield(void) {}",
             "void hw_svc(Ctx* c, u32 i) {} void hw_brk(Ctx* c, u64 pc, u32 i) {} void hw_unimpl(Ctx* c, u64 pc, u32 w) {}"]
    ok = []
    for i, (pc, w, dis) in enumerate(tests):
        try:
            t = tr.translate(w, pc)
        except Unimplemented:
            continue
        if t.kind == "bcond":
            body = f"return ({t.cond}) ? {hx(t.target)} : {hx(pc + 4)};"
        elif t.kind in (None,):
            body = f"{t.c} return {hx(pc + 4)};"
        else:
            continue
        lines.append(f"/* {dis} */\n__declspec(dllexport) u64 t_{len(ok)}(Ctx* restrict c) {{ {body} }}")
        ok.append((pc, w, dis))
    with open(src, "w") as f:
        f.write("\n".join(lines) + "\n")
    inc = os.path.join(ROOT, "runtime", "include")
    dll = os.path.join(BUILD, "difftest.dll")
    import msvc
    r = msvc.run(["clang-cl", "/nologo", "/LD", "/O2", "/arch:AVX2", "-mfma", "-maes", "-mpclmul", "-msse4.2", "-Wno-everything",
                  f"/I{inc}", src, f"/Fe{dll}", f"/Fo{BUILD}\\", "/link", "/NOIMPLIB", "/NOEXP"], cwd=BUILD)
    if r.returncode != 0:
        print(r.stdout[-6000:], r.stderr[-3000:])
        raise SystemExit("compile failed")
    return ctypes.CDLL(dll), ok


def run_unicorn(uc, pc, st):
    for i in range(31):
        uc.reg_write(A.UC_ARM64_REG_X0 + i if i < 29 else (A.UC_ARM64_REG_X29 if i == 29 else A.UC_ARM64_REG_X30), st["x"][i])
    uc.reg_write(A.UC_ARM64_REG_SP, st["sp"])
    for i in range(32):
        uc.reg_write(A.UC_ARM64_REG_Q0 + i, st["v"][i])
    uc.reg_write(A.UC_ARM64_REG_NZCV, st["nzcv"] << 28)
    uc.emu_start(pc, pc + 4, count=1)
    out = {"x": [uc.reg_read(A.UC_ARM64_REG_X0 + i if i < 29 else (A.UC_ARM64_REG_X29 if i == 29 else A.UC_ARM64_REG_X30))
                 for i in range(31)],
           "sp": uc.reg_read(A.UC_ARM64_REG_SP),
           "v": [uc.reg_read(A.UC_ARM64_REG_Q0 + i) for i in range(32)],
           "nzcv": (uc.reg_read(A.UC_ARM64_REG_NZCV) >> 28) & 0xF,
           "pc": uc.reg_read(A.UC_ARM64_REG_PC)}
    return out


def run_host(fn, st):
    c = Ctx()
    for i in range(31):
        c.x[i] = st["x"][i]
    c.sp = st["sp"]
    for i in range(32):
        c.v[i].d[0] = st["v"][i] & (2 ** 64 - 1)
        c.v[i].d[1] = st["v"][i] >> 64
    n = st["nzcv"]
    c.nf, c.zf, c.cf, c.vf = (n >> 3) & 1, (n >> 2) & 1, (n >> 1) & 1, n & 1
    fn.restype = ctypes.c_uint64
    pc = fn(ctypes.byref(c))
    return {"x": list(c.x[:31]), "sp": c.sp,
            "v": [c.v[i].d[0] | (c.v[i].d[1] << 64) for i in range(32)],
            "nzcv": (c.nf << 3) | (c.zf << 2) | (c.cf << 1) | c.vf, "pc": pc}


def is_nan32(v):
    return (v >> 23) & 0xFF == 0xFF and v & 0x7FFFFF


def is_nan64(v):
    return (v >> 52) & 0x7FF == 0x7FF and v & 0xFFFFFFFFFFFFF


def vec_equal(a, b):
    if a == b:
        return True
    # Treat NaN==NaN per lane (payload propagation differs between ARM and x86).
    ok32 = all(((a >> s) & 0xFFFFFFFF) == ((b >> s) & 0xFFFFFFFF) or
               (is_nan32((a >> s) & 0xFFFFFFFF) and is_nan32((b >> s) & 0xFFFFFFFF)) for s in range(0, 128, 32))
    ok64 = all(((a >> s) & (2 ** 64 - 1)) == ((b >> s) & (2 ** 64 - 1)) or
               (is_nan64((a >> s) & (2 ** 64 - 1)) and is_nan64((b >> s) & (2 ** 64 - 1))) for s in range(0, 128, 64))
    return ok32 or ok64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-mnemonic", type=int, default=40)
    ap.add_argument("--states", type=int, default=12)
    ap.add_argument("--filter", nargs="*")
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--module", default="main")
    a = ap.parse_args()

    p = Process(os.path.join(ROOT, "data", "exefs"))
    m = [x for x in p.modules if x.name == a.module][0]
    buckets = collect_samples(m, a.per_mnemonic, set(a.filter or []))
    tests = [t for b in buckets.values() for t in b]
    print(f"{len(tests)} sample encodings across {len(buckets)} mnemonics")
    dll, ok = build_dll(tests)
    print(f"compiled {len(ok)} tests")

    host_map(m.base, (m.image_size + 0xFFFF) & ~0xFFFF, m.image)
    host_map(SCRATCH, SCRATCH_SIZE)
    uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
    uc.reg_write(A.UC_ARM64_REG_CPACR_EL1, 0x300000)
    img_size = (m.image_size + 0xFFF) & ~0xFFF
    uc.mem_map(m.base, img_size)
    uc.mem_write(m.base, bytes(m.image))
    uc.mem_map(SCRATCH, SCRATCH_SIZE)

    rng = random.Random(42)
    fails = collections.defaultdict(list)
    passed = 0
    compared = 0
    uc_errors = collections.Counter()
    for idx, (pc, w, dis) in enumerate(ok):
        fn = getattr(dll, f"t_{idx}")
        for _ in range(a.states):
            st = rand_state(rng, w)
            init_mem = bytes(rng.getrandbits(8) for _ in range(4096))
            for base in (MID - 2048,):
                uc.mem_write(base, init_mem)
                ctypes.memmove(base, init_mem, len(init_mem))
            try:
                ue = run_unicorn(uc, pc, st)
            except UcError as e:
                uc_errors[str(e)] += 1
                continue
            compared += 1  # instruction faulted in the reference (e.g. out-of-range access)
            umem = bytes(uc.mem_read(MID - 2048, 4096))
            he = run_host(fn, st)
            hmem = ctypes.string_at(MID - 2048, 4096)
            diffs = []
            for i in range(31):
                if ue["x"][i] != he["x"][i]:
                    diffs.append(f"x{i}: uc={ue['x'][i]:#x} host={he['x'][i]:#x}")
            if ue["sp"] != he["sp"]:
                diffs.append(f"sp: uc={ue['sp']:#x} host={he['sp']:#x}")
            for i in range(32):
                if not vec_equal(ue["v"][i], he["v"][i]):
                    diffs.append(f"v{i}: uc={ue['v'][i]:032x} host={he['v'][i]:032x}")
            if ue["nzcv"] != he["nzcv"]:
                diffs.append(f"nzcv: uc={ue['nzcv']:04b} host={he['nzcv']:04b}")
            if ue["pc"] != he["pc"]:
                diffs.append(f"pc: uc={ue['pc']:#x} host={he['pc']:#x}")
            if umem != hmem:
                diffs.append("memory differs")
            if diffs:
                fails[dis.split()[0]].append((dis, w, st, diffs))
                break
        else:
            passed += 1
    print(f"passed {passed}/{len(ok)}  (state comparisons: {compared}, unicorn faults: {dict(uc_errors)})")
    for mn, fl in sorted(fails.items(), key=lambda kv: -len(kv[1])):
        print(f"--- {mn}: {len(fl)} failing encodings")
        for dis, w, st, diffs in fl[:a.show]:
            src = {f"x{i}": hex(st['x'][i]) for i in range(31)}
            print(f"   {w:08x} {dis}: {'; '.join(diffs[:4])}")


if __name__ == "__main__":
    main()
