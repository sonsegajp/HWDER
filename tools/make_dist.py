#!/usr/bin/env python3
"""Package a distributable HWDER build.

    python tools/make_dist.py [--build build/release] [--out dist] [--name HWDER-<date>]

The zip holds only our own work: hwder.exe (+ pdb unless --no-pdb), a README and a default
hwder_settings.ini. No game data and no keys are included: at first launch the exe reads the
user's own .nsp dump and prod.keys placed next to it and decrypts them in-process.
"""
import argparse
import datetime
import os
import shutil
import struct
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

README = """HWDER - Hyrule Warriors: Definitive Edition, native Windows build
=================================================================

This is a static recompilation of the Switch game: the game's own code was translated to
native x86-64 and linked with our runtime (HOS services, Maxwell -> Vulkan renderer, audio,
video). It is not an emulator and ships with no game data.

Setup
-----
1. Put these two files from your own console next to hwder.exe:
     <anything>.nsp     your dump of Hyrule Warriors: Definitive Edition (0100AE00096EA000, base game)
     prod.keys          your console's keys (also found in %USERPROFILE%\\.switch, or yuzu/Ryujinx key folders)
2. Run hwder.exe. Code and assets are read straight out of the NSP while you play; nothing
   is extracted to disk.

   Alternatively an already extracted dump works too: exefs\\ and romfs\\ folders next to the
   exe, or HWDER_EXEFS / HWDER_ROMFS environment variables pointing at them.

Requirements
------------
- Windows 10/11 x64, a Vulkan 1.2 GPU driver (NVIDIA / AMD / Intel), AVX2 CPU.
- Media Foundation (part of Windows; on "N" editions install the Media Feature Pack) for FMVs.

Controls
--------
- Any XInput controller (Xbox pads, or others through Steam Input / DS4Windows).
- Keyboard: WASD move, arrows camera, K/Enter = A, J/Backspace = B, I = X, L = Y,
  Q/E = L/R, U/O = ZL/ZR, Space = +, Tab = -, 1-4 = D-pad, Z/C = stick clicks.
- F1   video options (resolution scale, unlocked/ultrawide resolution, vsync, borderless,
       anisotropic filtering, FPS counter) and the save-file cheats tab.
- F11  toggle borderless fullscreen.

Files created next to the exe
-----------------------------
  hwder_settings.ini     options (edit with F1 or by hand)
  hwder.log              last run's log - attach it to bug reports
  save\\                  save data (same layout as the console's)
  cache\\, pipeline_cache.bin   shader/texture caches; safe to delete

Troubleshooting
---------------
- "No game data found": the .nsp or prod.keys is not next to hwder.exe.
- "not an NCA3 (wrong header_key?)": prod.keys is incomplete or from another console firmware;
  re-dump with Lockpick_RCM.
- "no title key for rights id": the NSP uses ticket crypto and its .tik is missing; add a
  title.keys file (rightsid = key) next to prod.keys.
"""

DEFAULT_SETTINGS = """refresh=60
vsync=1
aniso=16
window_w=0
window_h=0
borderless=1
res_scale=1.000
unlocked_res=0
show_fps=0
display_60hz=0
interp_hz=0
"""


def pe_imports(path):
    """DLLs named in the import table (to catch anything that is not a Windows system DLL)."""
    with open(path, "rb") as f:
        data = f.read()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (0x70 if magic == 0x20B else 0x60)
    imp_rva, imp_size = struct.unpack_from("<II", data, dd + 8)
    secs = []
    sh = opt + struct.unpack_from("<H", data, pe + 20)[0]
    for i in range(nsec):
        vsz, va, rsz, rp = struct.unpack_from("<IIII", data, sh + 40 * i + 8)
        secs.append((va, vsz, rp))

    def rva2off(r):
        for va, vsz, rp in secs:
            if va <= r < va + max(vsz, 1):
                return rp + (r - va)
        return None

    names = []
    off = rva2off(imp_rva)
    while off is not None:
        name_rva = struct.unpack_from("<I", data, off + 12)[0]
        if not name_rva:
            break
        no = rva2off(name_rva)
        names.append(data[no:data.index(b"\0", no)].decode())
        off += 20
    return names


SYSTEM_DLLS = {"kernel32.dll", "user32.dll", "gdi32.dll", "advapi32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll",
               "dbghelp.dll", "winmm.dll", "psapi.dll", "bcrypt.dll", "ws2_32.dll", "shlwapi.dll", "mf.dll", "mfplat.dll",
               "mfreadwrite.dll", "mfuuid.dll", "imm32.dll", "dwmapi.dll", "synchronization.dll", "ntdll.dll", "xinput1_4.dll",
               "xinput9_1_0.dll", "mmdevapi.dll", "avrt.dll", "ksuser.dll", "d3d11.dll", "dxgi.dll", "setupapi.dll", "hid.dll",
               "api-ms-win-core-synch-l1-2-0.dll"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(ROOT, "build", "release"))
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--name", default="HWDER-" + datetime.date.today().isoformat())
    ap.add_argument("--no-pdb", action="store_true")
    a = ap.parse_args()

    exe = os.path.join(a.build, "hwder.exe")
    if not os.path.exists(exe):
        sys.exit("missing " + exe)
    imports = pe_imports(exe)
    extra = [d for d in imports if d.lower() not in SYSTEM_DLLS and not d.lower().startswith("api-ms-win")]
    print("imports:", ", ".join(imports))
    if extra:
        print("WARNING: non-system DLLs imported, they must ship in the zip:", extra)

    stage = os.path.join(a.out, a.name)
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    shutil.copy2(exe, stage)
    pdb = os.path.join(a.build, "hwder.pdb")
    if not a.no_pdb and os.path.exists(pdb):
        shutil.copy2(pdb, stage)
    for d in extra:
        src = os.path.join(a.build, d)
        if os.path.exists(src):
            shutil.copy2(src, stage)
        else:
            print("WARNING: cannot find", d, "next to the exe")
    with open(os.path.join(stage, "README.txt"), "w", encoding="utf-8") as f:
        f.write(README)
    with open(os.path.join(stage, "hwder_settings.ini"), "w", encoding="utf-8") as f:
        f.write(DEFAULT_SETTINGS)
    open(os.path.join(stage, "PUT_YOUR_NSP_AND_prod.keys_HERE"), "w").close()

    zpath = os.path.join(a.out, a.name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for fn in sorted(os.listdir(stage)):
            z.write(os.path.join(stage, fn), os.path.join(a.name, fn))
    print("dist:", zpath, f"({os.path.getsize(zpath) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
