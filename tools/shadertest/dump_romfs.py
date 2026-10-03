"""Extract every precompiled NVN (GLSLC) Maxwell shader program found in the RomFS.

    python tools/shadertest/dump_romfs.py [romfs_dir] [out_dir]

GLSLC program sections start with a 0x30-byte NVN header (magic 0x12345678) followed by the 0x50-byte
Shader Program Header and the Maxwell code (scheduling word every 4th slot). Each unique program is
written to out_dir/<hash>.<stage>.bin as SPH + code (the code is cut after the terminating BRA-to-self).
Used as the corpus for the HWDER_SHADER_TEST mode of the shader translator.
"""
import hashlib
import mmap
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP_DIRS = {"sound", "voice", "se", "movie"}
STAGES = {1: "vs", 2: "tcs", 3: "tes", 4: "gs", 5: "fs"}
MAGIC = struct.pack("<I", 0x12345678)
BRA_SELF = 0xE2400FFFFF87000F


def program_at(d, o):
    """o = offset of the SPH. Returns (stage, bytes) or None."""
    if o + 0x60 > len(d):
        return None
    w0 = struct.unpack_from("<I", d, o)[0]
    sph_type, version, st = w0 & 31, (w0 >> 5) & 31, (w0 >> 10) & 15
    if version != 3 or st not in STAGES or (sph_type == 2) != (st == 5) or sph_type not in (1, 2):
        return None
    code = o + 0x50
    n = 0
    limit = min(len(d), code + 0x40000)
    while code + n * 8 + 8 <= limit:
        w = struct.unpack_from("<Q", d, code + n * 8)[0]
        # Program end: unconditional BRA with a tiny negative offset (BRA-to-self after EXIT).
        if n % 4 != 0 and (w >> 52) == 0xE24 and ((w >> 20) & 0xFFFFFF) >= 0xFFFFE0 and (w & 0x1F) == 0xF:
            n += 1
            n = (n + 3) & ~3
            return STAGES[st], bytes(d[o:code + min(n * 8, limit - code)])
        n += 1
    return None


def main():
    romfs = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "romfs")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "build", "shaders_romfs")
    os.makedirs(out, exist_ok=True)
    seen = set()
    counts = {}
    files_with = 0
    for dirpath, dirnames, filenames in os.walk(romfs):
        dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if os.path.getsize(p) < 0x100:
                continue
            with open(p, "rb") as f:
                d = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                pos = d.find(MAGIC)
                found = 0
                while pos >= 0:
                    r = program_at(d, pos + 0x30)
                    if r:
                        stage, blob = r
                        h = hashlib.sha1(blob).hexdigest()[:16]
                        found += 1
                        if h not in seen:
                            seen.add(h)
                            counts[stage] = counts.get(stage, 0) + 1
                            with open(os.path.join(out, f"{h}.{stage}.bin"), "wb") as g:
                                g.write(blob)
                    pos = d.find(MAGIC, pos + 4)
                d.close()
            if found:
                files_with += 1
                print(f"{found:5d} {os.path.relpath(p, romfs)}", flush=True)
    print("files with shaders:", files_with, "unique programs:", len(seen), counts)


if __name__ == "__main__":
    main()
