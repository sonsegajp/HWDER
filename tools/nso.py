"""NSO loading, dynamic linking and symbol resolution for Horizon modules.

Mirrors what Horizon's rtld does at boot: map each module, apply its RELA/JMPREL
relocations, and bind imported symbols against the global module list.
"""
import struct
import lz4.block

DT_NULL, DT_NEEDED, DT_PLTRELSZ, DT_PLTGOT, DT_HASH, DT_STRTAB, DT_SYMTAB, DT_RELA, DT_RELASZ, \
    DT_RELAENT, DT_STRSZ, DT_SYMENT = range(12)
DT_JMPREL = 23
DT_INIT, DT_FINI = 12, 13
DT_INIT_ARRAY, DT_FINI_ARRAY, DT_INIT_ARRAYSZ, DT_FINI_ARRAYSZ = 25, 26, 27, 28
DT_GNU_HASH = 0x6FFFFEF5
DT_RELACOUNT = 0x6FFFFFF9

R_AARCH64_ABS64 = 257
R_AARCH64_GLOB_DAT = 1025
R_AARCH64_JUMP_SLOT = 1026
R_AARCH64_RELATIVE = 1027

STT_FUNC, STT_OBJECT = 2, 1
SHN_UNDEF = 0

# Load order used by the Horizon loader for application ExeFS.
LOAD_ORDER = ["rtld", "main"] + [f"subsdk{i}" for i in range(10)] + ["sdk"]


class Symbol:
    __slots__ = ("name", "value", "size", "type", "bind", "shndx", "module")

    def __init__(self, name, value, size, info, shndx, module):
        self.name, self.value, self.size = name, value, size
        self.type, self.bind = info & 0xF, info >> 4
        self.shndx, self.module = shndx, module

    @property
    def defined(self):
        return self.shndx != SHN_UNDEF


class Module:
    def __init__(self, name, data):
        self.name = name
        magic, _ver, _r, flags = struct.unpack_from("<4sIII", data, 0)
        if magic != b"NSO0":
            raise ValueError(f"{name}: not an NSO")
        segs = []
        for i in range(3):
            foff, moff, size, extra = struct.unpack_from("<IIII", data, 0x10 + 16 * i)
            csize = struct.unpack_from("<I", data, 0x60 + 4 * i)[0]
            raw = data[foff:foff + csize]
            if flags & (1 << i):
                raw = lz4.block.decompress(raw, uncompressed_size=size)
            assert len(raw) == size, (name, i, len(raw), size)
            segs.append((moff, size, raw, extra))
        self.build_id = data[0x40:0x60]
        (self.text_off, self.text_size, text, _), (self.ro_off, self.ro_size, ro, _), \
            (self.data_off, self.data_size, dat, self.bss_size) = segs
        self.image_size = (self.data_off + self.data_size + self.bss_size + 0xFFF) & ~0xFFF
        img = bytearray(self.image_size)
        img[self.text_off:self.text_off + self.text_size] = text
        img[self.ro_off:self.ro_off + self.ro_size] = ro
        img[self.data_off:self.data_off + self.data_size] = dat
        self.image = img
        self.base = 0
        self._parse_mod0()

    # -- helpers on module-relative offsets
    def u32(self, off):
        return struct.unpack_from("<I", self.image, off)[0]

    def u64(self, off):
        return struct.unpack_from("<Q", self.image, off)[0]

    def cstr(self, off):
        end = self.image.index(b"\0", off)
        return self.image[off:end].decode("utf-8", "replace")

    def _parse_mod0(self):
        mod0 = self.u32(4)
        if self.image[mod0:mod0 + 4] != b"MOD0":
            raise ValueError(f"{self.name}: missing MOD0")
        rel = lambda i: mod0 + struct.unpack_from("<i", self.image, mod0 + 4 + 4 * i)[0]
        self.mod0 = mod0
        self.dynamic_off = rel(0)
        self.bss_start, self.bss_end = rel(1), rel(2)
        self.eh_frame_hdr = (rel(3), rel(4))
        self.module_object = rel(5)
        dyn = {}
        dyn_multi = {}
        off = self.dynamic_off
        while True:
            tag, val = struct.unpack_from("<qQ", self.image, off)
            off += 16
            if tag == DT_NULL:
                break
            dyn[tag] = val
            dyn_multi.setdefault(tag, []).append(val)
        self.dyn = dyn
        self.needed = dyn_multi.get(DT_NEEDED, [])
        strtab = dyn.get(DT_STRTAB, 0)
        symtab = dyn.get(DT_SYMTAB, 0)
        strsz = dyn.get(DT_STRSZ, 0)
        self.strtab = strtab
        # Symbol count: from DT_HASH nchain, else assume symtab runs up to strtab.
        if DT_HASH in dyn:
            nsyms = self.u32(dyn[DT_HASH] + 4)
        elif DT_GNU_HASH in dyn:
            nsyms = self._gnu_hash_count(dyn[DT_GNU_HASH])
        else:
            nsyms = (strtab - symtab) // 24
        self.symbols = []
        for i in range(nsyms):
            so = symtab + 24 * i
            st_name, st_info, _o, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", self.image, so)
            name = self.cstr(strtab + st_name) if st_name < strsz else ""
            self.symbols.append(Symbol(name, st_value, st_size, st_info, st_shndx, self))
        self.relas = []
        for tag_off, tag_sz in ((DT_RELA, DT_RELASZ), (DT_JMPREL, DT_PLTRELSZ)):
            if tag_off in dyn:
                for o in range(dyn[tag_off], dyn[tag_off] + dyn[tag_sz], 24):
                    r_off, r_info, r_add = struct.unpack_from("<QQq", self.image, o)
                    self.relas.append((r_off, r_info & 0xFFFFFFFF, r_info >> 32, r_add))
        self.init_array = self._ptr_array(DT_INIT_ARRAY, DT_INIT_ARRAYSZ)
        self.fini_array = self._ptr_array(DT_FINI_ARRAY, DT_FINI_ARRAYSZ)

    def _ptr_array(self, t_off, t_sz):
        if t_off not in self.dyn:
            return []
        return [self.dyn[t_off] + 8 * i for i in range(self.dyn[t_sz] // 8)]

    def _gnu_hash_count(self, off):
        nbuckets, symoffset, bloom_size, _shift = struct.unpack_from("<IIII", self.image, off)
        buckets = off + 16 + 8 * bloom_size
        chains = buckets + 4 * nbuckets
        last = 0
        for i in range(nbuckets):
            last = max(last, self.u32(buckets + 4 * i))
        if last < symoffset:
            return symoffset
        while not (self.u32(chains + 4 * (last - symoffset)) & 1):
            last += 1
        return last + 1

    @property
    def module_name(self):
        """Module path string embedded at the start of .rodata (e.g. 'nnSdk.nss')."""
        ro = self.ro_off
        length = self.u32(ro + 4)
        if 0 < length < 0x200 and self.u32(ro) == 0:
            return self.image[ro + 8:ro + 8 + length].split(b"\0")[0].decode("utf-8", "replace")
        return ""

    def defined_exports(self):
        return [s for s in self.symbols if s.defined and s.name]


class Process:
    """All modules mapped at their final addresses, relocated and linked."""

    def __init__(self, exefs_dir, base=0xFFFE00000):
        import os
        self.modules = []
        addr = base
        for name in LOAD_ORDER:
            p = os.path.join(exefs_dir, name)
            if not os.path.exists(p):
                continue
            m = Module(name, open(p, "rb").read())
            m.base = addr
            addr += (m.image_size + 0x1FFFFF) & ~0x1FFFFF  # 2MiB alignment, like the real loader
            self.modules.append(m)
        self.end = addr
        self.global_syms = {}
        for m in self.modules:
            for s in m.defined_exports():
                if s.bind in (1, 2) and s.name not in self.global_syms:  # GLOBAL / WEAK
                    self.global_syms[s.name] = s
        self.unresolved = []
        self.imports = {}  # (module, symbol name) -> resolved Symbol or None
        for m in self.modules:
            self._relocate(m)

    def addr_of(self, sym):
        return sym.module.base + sym.value

    def _relocate(self, m):
        for r_off, r_type, r_sym, r_add in m.relas:
            if r_type == R_AARCH64_RELATIVE:
                val = m.base + r_add
            elif r_type in (R_AARCH64_GLOB_DAT, R_AARCH64_JUMP_SLOT, R_AARCH64_ABS64):
                sym = m.symbols[r_sym]
                if sym.defined and sym.bind == 0:  # local symbol in this module
                    target = m.base + sym.value
                else:
                    tgt = self.global_syms.get(sym.name)
                    if tgt is None and sym.defined:
                        tgt = sym
                    self.imports[(m.name, sym.name)] = tgt
                    if tgt is None:
                        self.unresolved.append((m.name, sym.name))
                        target = 0
                    else:
                        target = self.addr_of(tgt)
                val = (target + (r_add if r_type != R_AARCH64_JUMP_SLOT else 0)) & 0xFFFFFFFFFFFFFFFF
            else:
                raise NotImplementedError(f"{m.name}: reloc type {r_type}")
            struct.pack_into("<Q", m.image, r_off, val)

    def module_at(self, addr):
        for m in self.modules:
            if m.base <= addr < m.base + m.image_size:
                return m
        return None

    def symbolize(self):
        """Map absolute address -> best name, for every defined function symbol in every module."""
        out = {}
        for m in self.modules:
            for s in m.symbols:
                if s.defined and s.name and s.type == STT_FUNC:
                    out.setdefault(m.base + s.value, s.name)
        return out
