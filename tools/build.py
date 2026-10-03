"""Configure + build HWDER with clang-cl and Ninja inside the VS developer environment.

    python tools/build.py [--jobs N] [--target hwder] [--only-file generated/rc_000.c]
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msvc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSCM = r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake"
CMAKE = os.path.join(VSCM, "CMake", "bin", "cmake.exe")
NINJA = os.path.join(VSCM, "Ninja", "ninja.exe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--only-file")
    ap.add_argument("--target", default="hwder")
    a = ap.parse_args()
    env = msvc.vs_env()
    if a.only_file:
        inc = os.path.join(ROOT, "runtime", "include")
        tp = os.path.join(ROOT, "third_party")
        cxx = a.only_file.endswith(".cpp")
        extra = (["/std:c++20", "/EHsc", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN", f"/I{os.path.join(ROOT, 'runtime', 'src')}",
                  f"/I{os.path.join(tp, 'vulkan-headers', 'include')}", f"/I{os.path.join(tp, 'spirv-headers', 'include')}"]
                 if cxx else [])
        obj = os.path.join(ROOT, "build", "probe_" + os.path.basename(a.only_file) + ".obj")
        r = msvc.run(["clang-cl", "/nologo", "/c", "/O2", "/arch:AVX2", "-mfma", "-Wno-everything",
                      f"/I{inc}", f"/I{os.path.join(ROOT, 'generated')}"] + extra + [a.only_file,
                      f"/Fo{obj}"], cwd=ROOT)
        print(r.stdout[-8000:], r.stderr[-4000:])
        sys.exit(r.returncode)
    bdir = os.path.join(ROOT, "build", "release")
    os.makedirs(bdir, exist_ok=True)
    # Several agents may build at once: serialize on a lock file.
    import msvcrt
    import time
    lockf = open(os.path.join(ROOT, "build", "build.lock"), "a+")
    while True:
        try:
            msvcrt.locking(lockf.fileno(), msvcrt.LK_NBLCK, 1)
            break
        except OSError:
            time.sleep(1)
    if not os.path.exists(os.path.join(bdir, "build.ninja")):
        r = subprocess.run([CMAKE, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={NINJA}", "-DCMAKE_BUILD_TYPE=Release",
                            "-DCMAKE_C_COMPILER=clang-cl", "-DCMAKE_CXX_COMPILER=clang-cl", ROOT],
                           cwd=bdir, env=env)
        if r.returncode:
            sys.exit(r.returncode)
    r = subprocess.run([NINJA, "-j", str(a.jobs), a.target], cwd=bdir, env=env)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
