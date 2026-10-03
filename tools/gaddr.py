"""Guest address inspector: disassemble around addresses and name them.

    python tools/gaddr.py ADDR[:N] ...     disassemble N instructions before ADDR (default 16), with symbols
    python tools/gaddr.py -s ADDR ...      only print the symbol containing ADDR
"""
import bisect
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(os.path.join(HERE, ".."))
from nso import Process  # noqa: E402
import capstone  # noqa: E402

p = Process("data/exefs")
syms = p.symbolize()
keys = sorted(syms)
md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)


def mod_of(a):
    for m in p.modules:
        if m.base <= a < m.base + len(m.image):
            return m


def u64(a):
    m = mod_of(a)
    return struct.unpack_from("<Q", m.image, a - m.base)[0] if m else 0


def name(a):
    i = bisect.bisect_right(keys, a) - 1
    if i < 0:
        return "?"
    base = keys[i]
    m = mod_of(a)
    return f"{m.name if m else '?'}!{syms[base]}+{a - base:#x}"


def plt_target(a):
    """If `a` is a PLT stub (adrp x16; ldr x17, [x16, #off]; add; br x17), return the resolved target."""
    m = mod_of(a)
    if not m:
        return None
    w = struct.unpack_from("<4I", m.image, a - m.base)
    if (w[0] & 0x9F00001F) != 0x90000010 or (w[3] & 0xFFFFFC1F) != 0xD61F0000:
        return None
    immlo, immhi = (w[0] >> 29) & 3, (w[0] >> 5) & 0x7FFFF
    page = (a & ~0xFFF) + ((((immhi << 2) | immlo) ^ 0x100000) - 0x100000 << 12)
    off = ((w[1] >> 10) & 0xFFF) * 8
    return u64(page + off)


def describe_target(t):
    tgt = plt_target(t)
    return f"{name(t)}" + (f" -> {name(tgt)}" if tgt else "")


if sys.argv[1:2] == ["-f"]:  # -f SUBSTRING[:N]: disassemble N instructions from each matching symbol
    pat, n = (sys.argv[2].split(":") + ["60"])[:2]
    for a in keys:
        if pat in syms[a]:
            m = mod_of(a)
            print(f"--- {a:#x} {name(a)}")
            for i in md.disasm(bytes(m.image[a - m.base:a - m.base + 4 * int(n)]), a):
                extra = ""
                if i.mnemonic in ("bl", "b") and i.op_str.startswith("#"):
                    extra = "   ; " + describe_target(int(i.op_str[1:], 16))
                print(f"{i.address:#x}: {i.mnemonic} {i.op_str}{extra}")
    sys.exit(0)

only_sym = "-s" in sys.argv
for arg in [x for x in sys.argv[1:] if x != "-s"]:
    a, n = (arg.split(":") + ["16"])[:2]
    a, n = int(a, 16), int(n)
    if only_sym:
        print(f"{a:#x}: {name(a)}")
        continue
    m = mod_of(a)
    off = a - m.base
    print(f"--- {a:#x} {name(a)}")
    for i in md.disasm(bytes(m.image[off - 4 * n:off + 16]), a - 4 * n):
        extra = ""
        if i.mnemonic in ("bl", "b") and i.op_str.startswith("#"):
            extra = "   ; " + describe_target(int(i.op_str[1:], 16))
        print(f"{i.address:#x}: {i.mnemonic} {i.op_str}{extra}")
