#!/usr/bin/env python3
"""Extract ExeFS / RomFS / control data from a Switch NSP using the user's own keys.

Usage: python nx_extract.py <game.nsp> <prod.keys> <out_dir> [--no-romfs]

Nothing here contains keys or game data; everything comes from the user's dump.
"""
import os
import struct
import sys
import argparse

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

CHUNK = 16 * 1024 * 1024
CONTENT_TYPES = {0: "program", 1: "meta", 2: "control", 3: "manual", 4: "data", 5: "publicdata"}


def load_keys(path):
    keys = {}
    with open(path, "r") as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            keys[k.strip()] = bytes.fromhex(v.strip())
    return keys


def aes_ecb_dec(key, data):
    d = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
    return d.update(data) + d.finalize()


def xts_dec_sectors(key, data, first_sector, sector_size=0x200):
    out = bytearray()
    for i in range(0, len(data), sector_size):
        tweak = (first_sector + i // sector_size).to_bytes(16, "big")  # Nintendo tweak
        d = Cipher(algorithms.AES(key), modes.XTS(tweak)).decryptor()
        out += d.update(data[i:i + sector_size]) + d.finalize()
    return bytes(out)


# ---------------------------------------------------------------- PFS0

def parse_pfs0(read, base):
    hdr = read(base, 16)
    magic, n, stsz, _ = struct.unpack("<4sIII", hdr)
    if magic != b"PFS0":
        raise ValueError(f"bad PFS0 magic {magic!r} at {base:#x}")
    ents = read(base + 16, 24 * n)
    strtab = read(base + 16 + 24 * n, stsz)
    data_base = base + 16 + 24 * n + stsz
    files = []
    for i in range(n):
        off, sz, name_off, _ = struct.unpack_from("<QQII", ents, 24 * i)
        name = strtab[name_off:strtab.index(b"\0", name_off)].decode()
        files.append((name, data_base + off, sz))
    return files


# ---------------------------------------------------------------- NCA

class NcaSection:
    def __init__(self, nca, idx, start, end, fsh):
        self.nca, self.idx, self.start, self.end, self.fsh = nca, idx, start, end, fsh
        self.fs_type = fsh[2]
        self.hash_type = fsh[3]
        self.enc_type = fsh[4]
        self.ctr_hi = fsh[0x140:0x148][::-1]  # stored LE, used BE

    def read(self, off, size):
        """Read decrypted bytes at section-relative offset."""
        abs_off = self.start + off
        aligned = abs_off & ~0xF
        skip = abs_off - aligned
        raw = self.nca.read_raw(aligned, size + skip)
        if self.enc_type == 1:
            return raw[skip:skip + size]
        if self.enc_type != 3:
            raise NotImplementedError(f"encryption type {self.enc_type}")
        ctr = self.ctr_hi + (aligned >> 4).to_bytes(8, "big")
        d = Cipher(algorithms.AES(self.nca.ctr_key), modes.CTR(ctr)).decryptor()
        return d.update(raw)[skip:skip + size]

    def data_region(self):
        h = self.fsh[8:8 + 0xF8]
        if self.hash_type == 2:  # HierarchicalSha256 (PFS0)
            layer_count = struct.unpack_from("<I", h, 0x24)[0]
            off, size = struct.unpack_from("<QQ", h, 0x28 + 16 * (layer_count - 1))
            return off, size
        if self.hash_type == 3:  # HierarchicalIntegrity (IVFC)
            if h[:4] != b"IVFC":
                raise ValueError("bad IVFC")
            off, size = struct.unpack_from("<QQ", h, 0x10 + 0x18 * 5)
            return off, size
        raise NotImplementedError(f"hash type {self.hash_type}")


class Nca:
    def __init__(self, f, offset, size, keys, title_keys):
        self.f, self.offset, self.size = f, offset, size
        f.seek(offset)
        hdr = xts_dec_sectors(keys["header_key"], f.read(0xC00), 0)
        if hdr[0x200:0x204] != b"NCA3":
            raise ValueError("not NCA3: " + repr(hdr[0x200:0x204]))
        self.hdr = hdr
        self.content_type = hdr[0x205]
        self.kaek_idx = hdr[0x207]
        self.title_id = struct.unpack_from("<Q", hdr, 0x210)[0]
        self.sdk_version = struct.unpack_from("<I", hdr, 0x21C)[0]
        keygen = max(hdr[0x206], hdr[0x220])
        self.mkrev = keygen - 1 if keygen > 0 else 0
        rights_id = hdr[0x230:0x240]
        if any(rights_id):
            enc_tk = title_keys.get(rights_id)
            if enc_tk is None:
                raise KeyError("missing title key for rights id " + rights_id.hex())
            self.ctr_key = aes_ecb_dec(keys[f"titlekek_{self.mkrev:02x}"], enc_tk)
        else:
            kak_name = ["key_area_key_application", "key_area_key_ocean", "key_area_key_system"][self.kaek_idx]
            kak = keys[f"{kak_name}_{self.mkrev:02x}"]
            key_area = aes_ecb_dec(kak, hdr[0x300:0x340])
            self.ctr_key = key_area[0x20:0x30]
        self.sections = []
        for i in range(4):
            start_blk, end_blk = struct.unpack_from("<II", hdr, 0x240 + 16 * i)
            if end_blk == 0:
                continue
            fsh = hdr[0x400 + 0x200 * i:0x600 + 0x200 * i]
            self.sections.append(NcaSection(self, i, start_blk * 0x200, end_blk * 0x200, fsh))

    def read_raw(self, off, size):
        self.f.seek(self.offset + off)
        return self.f.read(size)


# ---------------------------------------------------------------- RomFS

def extract_romfs(sec, out_dir):
    data_off, data_size = sec.data_region()
    read = lambda o, s: sec.read(data_off + o, s)
    hdr = read(0, 0x50)
    vals = struct.unpack_from("<10Q", hdr, 0)
    dir_meta_off, dir_meta_size = vals[3], vals[4]
    file_meta_off, file_meta_size = vals[7], vals[8]
    file_data_off = vals[9]
    dir_meta = read(dir_meta_off, dir_meta_size)
    file_meta = read(file_meta_off, file_meta_size)
    NONE = 0xFFFFFFFF
    manifest = []

    def walk_files(first, path):
        fo = first
        while fo != NONE:
            parent, sibling, doff, dsize, _hs, nlen = struct.unpack_from("<IIQQII", file_meta, fo)
            name = file_meta[fo + 0x20:fo + 0x20 + nlen].decode("utf-8")
            fpath = os.path.join(path, name)
            manifest.append((fpath, file_data_off + doff, dsize))
            fo = sibling

    def walk_dirs(do, path):
        parent, sibling, child_dir, child_file, _hs, nlen = struct.unpack_from("<IIIIII", dir_meta, do)
        name = dir_meta[do + 0x18:do + 0x18 + nlen].decode("utf-8")
        p = os.path.join(path, name) if name else path
        os.makedirs(p, exist_ok=True)
        walk_files(child_file, p)
        cd = child_dir
        while cd != NONE:
            walk_dirs(cd, p)
            cd = struct.unpack_from("<I", dir_meta, cd + 4)[0]

    walk_dirs(0, out_dir)
    total = sum(s for _, _, s in manifest)
    done = 0
    for path, off, size in manifest:
        with open(path, "wb") as o:
            pos = 0
            while pos < size:
                n = min(CHUNK, size - pos)
                o.write(read(off + pos, n))
                pos += n
        done += size
        print(f"\r  romfs {done / total * 100:5.1f}%  {len(manifest)} files", end="", flush=True)
    print()
    return manifest


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nsp")
    ap.add_argument("prod_keys")
    ap.add_argument("out")
    ap.add_argument("--no-romfs", action="store_true")
    a = ap.parse_args()

    keys = load_keys(a.prod_keys)
    f = open(a.nsp, "rb")
    raw_read = lambda o, s: (f.seek(o), f.read(s))[1]
    files = parse_pfs0(raw_read, 0)

    title_keys = {}
    for name, off, size in files:
        if name.endswith(".tik"):
            tik = raw_read(off, size)
            sig_type = struct.unpack_from("<I", tik, 0)[0]
            body = {0x10000: 0x240, 0x10001: 0x140, 0x10002: 0x80,
                    0x10003: 0x240, 0x10004: 0x140, 0x10005: 0x80}[sig_type]
            title_keys[tik[body + 0x160:body + 0x170]] = tik[body + 0x40:body + 0x50]

    os.makedirs(a.out, exist_ok=True)
    for name, off, size in files:
        if not name.endswith(".nca") or name.endswith(".cnmt.nca"):
            continue
        nca = Nca(f, off, size, keys, title_keys)
        ctype = CONTENT_TYPES.get(nca.content_type, str(nca.content_type))
        print(f"{name}: {ctype} tid={nca.title_id:016x} sdk={nca.sdk_version:#x} mkrev={nca.mkrev} "
              f"sections={[(s.idx, s.fs_type, s.enc_type, hex(s.end - s.start)) for s in nca.sections]}")
        for sec in nca.sections:
            if sec.fs_type == 1:  # PFS0
                doff, _ = sec.data_region()
                pfs = parse_pfs0(lambda o, s: sec.read(doff + o, s), 0)
                sub = "exefs" if ctype == "program" else f"{ctype}_pfs{sec.idx}"
                d = os.path.join(a.out, sub)
                os.makedirs(d, exist_ok=True)
                for fn, foff, fsz in pfs:
                    with open(os.path.join(d, fn), "wb") as o:
                        o.write(sec.read(doff + foff, fsz))
                    print(f"  {sub}/{fn} {fsz:,}")
            elif sec.fs_type == 0:  # RomFS
                if ctype == "program" and a.no_romfs:
                    print("  (skipping program romfs)")
                    continue
                sub = "romfs" if ctype == "program" else f"{ctype}_romfs_{name[:8]}"
                print(f"  extracting {sub}")
                extract_romfs(sec, os.path.join(a.out, sub))


if __name__ == "__main__":
    main()
