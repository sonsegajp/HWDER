"""Build tools/shadertest (offline translator harness) and optionally run + validate the corpus.

    python tools/shadertest/build.py [--hist] [--run] [--corpus build/shaders_romfs]

Every program is translated and checked with tools/spirv-tools/spirv-val.exe (built from
KhronosGroup/SPIRV-Tools source).
"""
import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import msvc  # noqa: E402

EXE = os.path.join(ROOT, "build", "shadertest", "shadertest.exe")
VAL = os.path.join(ROOT, "tools", "spirv-tools", "spirv-val.exe")


def sources():
    txt = open(os.path.join(ROOT, "runtime", "src", "shader", "sources.cmake")).read()
    return [os.path.join(ROOT, l.strip()) for l in txt.splitlines() if l.strip().startswith("runtime/")]


def build():
    obj = os.path.join(ROOT, "build", "shadertest")
    os.makedirs(obj, exist_ok=True)
    tp = os.path.join(ROOT, "third_party")
    args = ["clang-cl", "/nologo", "/std:c++20", "/EHsc", "/O2", "/Z7", "/MT", "-Wno-everything", "/DNOMINMAX",
            f"/I{os.path.join(ROOT, 'runtime', 'include')}", f"/I{os.path.join(ROOT, 'runtime', 'src')}",
            f"/I{os.path.join(tp, 'spirv-headers', 'include')}", f"/Fo{obj}\\", f"/Fe{EXE}",
            os.path.join(HERE, "shadertest.cpp")] + sources() + ["/link", "/DEBUG"]
    r = msvc.run(args, cwd=obj)
    if r.returncode:
        print(r.stdout[-8000:], r.stderr[-3000:])
        sys.exit(1)


def validate(spv):
    r = subprocess.run([VAL, "--target-env", "vulkan1.1", spv], capture_output=True, text=True)
    return spv, r.returncode, (r.stdout + r.stderr).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--hist", action="store_true")
    ap.add_argument("--nobuild", action="store_true")
    ap.add_argument("--corpus", default=os.path.join(ROOT, "build", "shaders_romfs"))
    a = ap.parse_args()
    if not a.nobuild:
        build()
    if a.hist:
        subprocess.run([EXE, "hist", os.path.relpath(a.corpus, ROOT)], cwd=ROOT)
    if a.run:
        out = os.path.join(ROOT, "build", "shadertest", "spv")
        os.makedirs(out, exist_ok=True)
        for f in [os.path.join(out, x) for x in os.listdir(out) if x.endswith(".spv")]:
            os.remove(f)
        r = subprocess.run([EXE, "run", os.path.relpath(a.corpus, ROOT), os.path.relpath(out, ROOT)], capture_output=True, text=True, cwd=ROOT)
        print(r.stdout.strip(), "exit", r.returncode)
        uniq = sorted(set(r.stderr.strip().splitlines()))
        if uniq:
            print("translator log (unique):")
            for l in uniq[:80]:
                print("  ", l)
        spvs = sorted(os.path.join(out, x) for x in os.listdir(out) if x.endswith(".spv"))
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(validate, spvs))
        bad = [x for x in res if x[1]]
        print(f"validated {len(res)}: {len(res) - len(bad)} ok, {len(bad)} failed")
        msgs = {}
        for spv, _, m in bad:
            key = m.splitlines()[0][:200] if m else "?"
            msgs.setdefault(key, []).append(os.path.basename(spv))
        for k, v in sorted(msgs.items(), key=lambda kv: -len(kv[1]))[:30]:
            print(f"{len(v):5d} {k}  e.g. {v[0]}")


if __name__ == "__main__":
    main()
