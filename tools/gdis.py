"""Disassemble guest code with import names: python gdis.py ADDR [LEN] [ADDR LEN ...]"""
import sys, capstone, subprocess, functools
sys.argv, args = [sys.argv[0]], sys.argv[1:]
from recomp.gen import *

@functools.lru_cache(None)
def load():
    proc = Process('../data/exefs'); mod = Module(proc, 'main')
    plt, _ = find_plt_stubs(mod)
    names = list(plt.values())
    dem = dict(zip(names, subprocess.run(['C:/devkitPro/devkitA64/bin/aarch64-none-elf-c++filt.exe'], input='\n'.join(names), capture_output=True, text=True).stdout.split('\n')))
    return mod, plt, dem

def dis(a, n):
    mod, plt, dem = load()
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    for x, s, mn, op in md.disasm_lite(bytes(mod.m.image[a - mod.base:a + n - mod.base]), a):
        extra = ''
        if mn in ('bl', 'b') and op.startswith('#'):
            t = int(op[1:], 16)
            extra = dem.get(plt.get(t, ''), '')
        print(f"  {x:#x} {mn} {op} {extra}")

for i in range(0, len(args), 2):
    a = int(args[i], 16); n = int(args[i + 1], 16) if i + 1 < len(args) else 0x80
    print(f"--- {a:#x}"); dis(a, n)
