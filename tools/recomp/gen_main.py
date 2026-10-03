"""Driver: recompile every ExeFS module into one program.

    python -m recomp.gen_main [--modules rtld,main,subsdk0,subsdk1,sdk]

Outputs (in generated/):
  rc_NNNN.c       recompiled functions (~12k lines per file)
  funcs.h         prototypes
  functable.c     guest address -> host function table (all modules)
  module_info.c   module layout (the runtime loads the NSOs itself from the user's ExeFS)
  sources.cmake   file list for CMake
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nso import Process  # noqa: E402
from recomp.gen import Emitter, Module, discover, fn_name  # noqa: E402
from recomp.a64 import hx  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="rtld,main,subsdk0,subsdk1,sdk")
    ap.add_argument("--exefs", default=os.path.join(HERE, "..", "..", "data", "exefs"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "generated"))
    ap.add_argument("--lines-per-file", type=int, default=12000)
    ap.add_argument("--extra", default=os.path.join(HERE, "..", "..", "data", "extra_entries.txt"),
                    help="extra entry points (hex, one per line) discovered at runtime")
    a = ap.parse_args()
    t0 = time.time()

    def log(s):
        print(f"[{time.time() - t0:6.1f}s] {s}", flush=True)

    proc = Process(a.exefs)
    extra = set()
    if os.path.exists(a.extra):
        extra = {int(x, 16) for x in open(a.extra).read().split() if x.strip()}
        log(f"{len(extra)} extra entry points from {a.extra}")
    os.makedirs(a.out, exist_ok=True)

    all_starts, files, mod_infos, resumable = [], [], [], []
    for name in a.modules.split(","):
        if not any(m.name == name for m in proc.modules):
            continue
        mod = Module(proc, name)
        log(f"module {name}: base={mod.base:#x} text={mod.text_lo:#x}-{mod.text_hi:#x}")
        funcs, _ = discover(mod, {}, log, {x for x in extra if mod.in_text(x)})
        em = Emitter(mod, funcs, {}, log)
        starts = sorted(funcs)
        # One file per 64 KiB of guest code: adding an entry point only changes (and rebuilds) one file.
        buckets = {}
        for st in starts:
            buckets.setdefault((st - mod.text_lo) >> 16, []).append(st)
        for bucket, sts in sorted(buckets.items()):
            fname = f"rc_{name}_{bucket:03x}.c"
            body = "".join(em.emit_function(st, funcs[st]) for st in sts)
            # Per-file prototypes (not a global header) so unrelated changes don't rebuild everything.
            protos = sorted(set(re.findall(r"\bf_[0-9a-f]+\b", body)))
            text = ('#include "hwder/cpu.h"\n#include "hwder/dispatch.h"\n'
                    + "".join(f"void {p}(Ctx* restrict c);\n" for p in protos) + "\n" + body)
            path = os.path.join(a.out, fname)
            old = open(path).read() if os.path.exists(path) else None
            if old != text:  # keep timestamps of unchanged files for incremental builds
                with open(path, "w") as f:
                    f.write(text)
            files.append(fname)
        log(f"  {len(starts)} functions; stats {dict(em.stats)}")
        all_starts += starts
        resumable += getattr(em, "resumable", [])
        mod_infos.append((name, mod.base, mod.m.image_size, mod.text_lo, mod.text_hi))

    for old in os.listdir(a.out):  # drop files from an older layout
        if old.startswith("rc_") and old.endswith(".c") and old not in files:
            os.remove(os.path.join(a.out, old))

    def write_if_changed(fname, text):
        path = os.path.join(a.out, fname)
        if not os.path.exists(path) or open(path).read() != text:
            with open(path, "w") as f:
                f.write(text)

    with open(os.path.join(a.out, "sources.cmake"), "w") as f:
        f.write("set(RECOMP_SOURCES\n" + "".join(f"  ${{GEN_DIR}}/{x}\n" for x in files) + ")\n")
    with open(os.path.join(a.out, "funcs.h"), "w") as f:
        f.write('#pragma once\n#include "hwder/cpu.h"\n#include "hwder/dispatch.h"\n')
        for st in all_starts:
            f.write(f"void {fn_name(st)}(Ctx* restrict c);\n")
    with open(os.path.join(a.out, "functable.c"), "w") as f:
        f.write('#include "funcs.h"\n\nconst HwFuncEntry hw_func_table[] = {\n')
        for st in all_starts:
            f.write(f"  {{ {hx(st)}, {fn_name(st)} }},\n")
        f.write("};\nconst unsigned hw_func_count = sizeof(hw_func_table) / sizeof(hw_func_table[0]);\n")
        # Functions that accept entry at any instruction (unresolved indirect branches), sorted.
        f.write("const unsigned long long hw_resumable[] = {" + ",".join(hx(x) for x in sorted(resumable)) + ",0};\n")
        f.write(f"const unsigned hw_resumable_count = {len(resumable)};\n")
    with open(os.path.join(a.out, "module_info.c"), "w") as f:
        f.write('#include "hwder/module.h"\n\nconst HwModuleInfo hw_modules[] = {\n')
        for name, base, size, tlo, thi in mod_infos:
            f.write(f'  {{ "{name}", {hx(base)}, {hx(size)}, {hx(tlo)}, {hx(thi)} }},\n')
        f.write("};\nconst unsigned hw_module_count = sizeof(hw_modules) / sizeof(hw_modules[0]);\n")
    log(f"done: {len(all_starts)} functions in {len(files)} files")


if __name__ == "__main__":
    main()
