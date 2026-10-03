// In-process NSP reader: PFS0 container -> program NCA3 -> ExeFS (PFS0 section) + RomFS (IVFC
// section), decrypted on demand. Nothing is written to disk; file data comes out of the user's
// own dump with the user's own keys.
//
//   NSP            PFS0: [name].nca files (+ .tik/.cert for titlekey crypto, .cnmt.nca metadata)
//   NCA3 header    0xC00 bytes, AES-128-XTS with header_key, 0x200-byte sectors, tweak = sector (BE)
//   key area       0x300..0x340, AES-128-ECB with key_area_key_<kaek>_<keygen>; key 2 is the CTR key
//   titlekey       rights id set: titlekey from the ticket (or title.keys), AES-ECB with titlekek_<keygen>
//   sections       AES-128-CTR, counter = ctr_hi (8 bytes from the FS header, byte-reversed) ++ (offset >> 4)
//   section data   hash type 2 (HierarchicalSha256 / PFS0): last layer; hash type 3 (IVFC): level 5
#include <windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>

#include "fs_internal.h"
#include "nsp.h"

namespace hwfs::nsp {

namespace {

// ---------------------------------------------------------------- AES (CNG, AES-NI backed)
class Aes {
public:
    Aes() = default;
    Aes(const Aes&) = delete;
    Aes& operator=(const Aes&) = delete;
    Aes(Aes&& o) noexcept : key_(o.key_) { o.key_ = nullptr; }
    Aes& operator=(Aes&& o) noexcept {
        if (this != &o) {
            if (key_) BCryptDestroyKey(key_);
            key_ = o.key_;
            o.key_ = nullptr;
        }
        return *this;
    }
    ~Aes() {
        if (key_) BCryptDestroyKey(key_);
    }
    bool init(const u8 key[16]) {
        static BCRYPT_ALG_HANDLE alg = [] {
            BCRYPT_ALG_HANDLE a = nullptr;
            if (BCryptOpenAlgorithmProvider(&a, BCRYPT_AES_ALGORITHM, nullptr, 0) < 0) return (BCRYPT_ALG_HANDLE) nullptr;
            BCryptSetProperty(a, BCRYPT_CHAINING_MODE, (PUCHAR)BCRYPT_CHAIN_MODE_ECB, sizeof(BCRYPT_CHAIN_MODE_ECB), 0);
            return a;
        }();
        if (!alg) return false;
        return BCryptGenerateSymmetricKey(alg, &key_, nullptr, 0, (PUCHAR)key, 16, 0) >= 0;
    }
    // In place, whole 16-byte blocks.
    bool encrypt(u8* data, size_t n) const {
        ULONG out = 0;
        return BCryptEncrypt(key_, data, (ULONG)n, nullptr, nullptr, 0, data, (ULONG)n, &out, 0) >= 0;
    }
    bool decrypt(u8* data, size_t n) const {
        ULONG out = 0;
        return BCryptDecrypt(key_, data, (ULONG)n, nullptr, nullptr, 0, data, (ULONG)n, &out, 0) >= 0;
    }

private:
    BCRYPT_KEY_HANDLE key_ = nullptr;
};

// XTS sector decrypt (Nintendo tweak: sector number as a big-endian 128-bit value).
void xts_decrypt_sector(const Aes& k1, const Aes& k2, u8* data, size_t n, u64 sector) {
    u8 t[16] = {};
    for (int i = 0; i < 8; i++) t[15 - i] = (u8)(sector >> (8 * i));
    k2.encrypt(t, 16);
    for (size_t off = 0; off < n; off += 16) {
        u8* b = data + off;
        for (int i = 0; i < 16; i++) b[i] ^= t[i];
        k1.decrypt(b, 16);
        for (int i = 0; i < 16; i++) b[i] ^= t[i];
        // t *= alpha in GF(2^128), little-endian bit order
        u8 carry = 0;
        for (int i = 0; i < 16; i++) {
            u8 c = t[i] >> 7;
            t[i] = (u8)((t[i] << 1) | carry);
            carry = c;
        }
        if (carry) t[0] ^= 0x87;
    }
}

// ---------------------------------------------------------------- keys
struct Keys {
    std::map<std::string, std::vector<u8>> prod;      // name -> bytes
    std::map<std::string, std::vector<u8>> titles;    // rights id (hex, lower) -> encrypted titlekey
    bool has(const std::string& k) const { return prod.count(k) != 0; }
    const u8* get(const std::string& k) const {
        auto it = prod.find(k);
        return it == prod.end() ? nullptr : it->second.data();
    }
};

std::vector<u8> from_hex(std::string s) {
    std::vector<u8> out;
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return isspace(c); }), s.end());
    if (s.size() % 2) return out;
    auto nyb = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        c = (char)tolower((unsigned char)c);
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < s.size(); i += 2) {
        int a = nyb(s[i]), b = nyb(s[i + 1]);
        if (a < 0 || b < 0) return {};
        out.push_back((u8)(a * 16 + b));
    }
    return out;
}

std::string to_hex(const u8* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += d[p[i] >> 4];
        s += d[p[i] & 15];
    }
    return s;
}

bool file_exists(const std::string& p) {
    DWORD a = GetFileAttributesW(utf8_to_wide(p).c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

void load_key_file(const std::string& path, std::map<std::string, std::vector<u8>>& into, bool lower_names) {
    FILE* f = _wfopen(utf8_to_wide(path).c_str(), L"r");
    if (!f) return;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        std::string s(line);
        size_t eq = s.find('=');
        if (eq == std::string::npos) continue;
        std::string k = s.substr(0, eq), v = s.substr(eq + 1);
        k.erase(std::remove_if(k.begin(), k.end(), [](unsigned char c) { return isspace(c); }), k.end());
        if (k.empty() || k[0] == '#' || k[0] == ';') continue;
        if (lower_names)
            for (auto& c : k) c = (char)tolower((unsigned char)c);
        std::vector<u8> bytes = from_hex(v);
        if (!bytes.empty()) into[k] = std::move(bytes);
    }
    fclose(f);
}

std::string env_str(const char* name) {  // UTF-8 (read wide: ANSI cannot spell every path)
    const wchar_t* e = _wgetenv(utf8_to_wide(name).c_str());
    return e ? wide_to_utf8(e) : std::string();
}

// Candidate prod.keys locations, in priority order.
std::vector<std::string> key_candidates(const std::string& exe_dir) {
    std::vector<std::string> c;
    if (std::string e = env_str("HWDER_KEYS"); !e.empty()) c.push_back(e);
    c.push_back(exe_dir + "\\prod.keys");
    c.push_back(exe_dir + "\\keys\\prod.keys");
    if (std::string h = env_str("USERPROFILE"); !h.empty()) c.push_back(h + "\\.switch\\prod.keys");
    if (std::string a = env_str("APPDATA"); !a.empty()) {
        c.push_back(a + "\\yuzu\\keys\\prod.keys");
        c.push_back(a + "\\Ryujinx\\system\\prod.keys");
    }
    return c;
}

// ---------------------------------------------------------------- file
class File {
public:
    ~File() {
        if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
    }
    bool open(const std::string& path) {
        std::wstring w = utf8_to_wide(path);
        h_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER sz;
        GetFileSizeEx(h_, &sz);
        size_ = (u64)sz.QuadPart;
        return true;
    }
    u64 size() const { return size_; }
    bool read(u64 off, u8* dst, u64 size) const {
        while (size) {
            DWORD chunk = (DWORD)std::min<u64>(size, 0x10000000ull);
            OVERLAPPED ov{};
            ov.Offset = (DWORD)off;
            ov.OffsetHigh = (DWORD)(off >> 32);
            DWORD got = 0;
            if (!ReadFile(h_, dst, chunk, &got, &ov) || got == 0) return false;
            dst += got;
            off += got;
            size -= got;
        }
        return true;
    }

private:
    HANDLE h_ = INVALID_HANDLE_VALUE;
    u64 size_ = 0;
};

struct PfsEntry {
    std::string name;
    u64 offset, size;  // absolute within the reader's space
};

// Parses a PFS0 through `read(off, dst, size)`; entries get absolute offsets (base + data start).
template <class Reader>
bool parse_pfs0(const Reader& read, u64 base, std::vector<PfsEntry>& out) {
    u8 hdr[16];
    if (!read(base, hdr, 16) || memcmp(hdr, "PFS0", 4) != 0) return false;
    u32 n, stsz;
    memcpy(&n, hdr + 4, 4);
    memcpy(&stsz, hdr + 8, 4);
    if (n > 4096 || stsz > (1u << 24)) return false;
    std::vector<u8> ents(24ull * n), strtab(stsz);
    if (n && !read(base + 16, ents.data(), ents.size())) return false;
    if (stsz && !read(base + 16 + ents.size(), strtab.data(), stsz)) return false;
    u64 data_base = base + 16 + ents.size() + stsz;
    for (u32 i = 0; i < n; i++) {
        u64 off, sz;
        u32 name_off;
        memcpy(&off, &ents[24 * i], 8);
        memcpy(&sz, &ents[24 * i + 8], 8);
        memcpy(&name_off, &ents[24 * i + 16], 4);
        if (name_off >= stsz) return false;
        const char* nm = (const char*)&strtab[name_off];
        out.push_back({std::string(nm, strnlen(nm, stsz - name_off)), data_base + off, sz});
    }
    return true;
}

// ---------------------------------------------------------------- NCA
struct Section {
    int idx = 0;
    u64 start = 0, end = 0;  // NCA-relative
    u8 fs_type = 0, hash_type = 0, enc_type = 0;
    u8 ctr_hi[8] = {};
    u64 data_off = 0, data_size = 0;  // section-relative data region
};

class Nca {
public:
    bool open(const File& f, u64 offset, u64 size, const Keys& keys, std::string& err);
    const std::vector<Section>& sections() const { return sections_; }
    u8 content_type() const { return content_type_; }
    u64 title_id() const { return title_id_; }
    // Decrypted section-relative read.
    bool read(const Section& s, u64 off, u8* dst, u64 size) const;
    bool read_abs(const Section& s, u64 off, u8* dst, u64 size) const { return read(s, off, dst, size); }

private:
    const File* f_ = nullptr;
    u64 offset_ = 0, size_ = 0;
    u8 content_type_ = 0;
    u64 title_id_ = 0;
    Aes ctr_;
    std::vector<Section> sections_;
};

bool Nca::open(const File& f, u64 offset, u64 size, const Keys& keys, std::string& err) {
    f_ = &f;
    offset_ = offset;
    size_ = size;
    const u8* hk = keys.get("header_key");
    if (!hk || keys.prod.at("header_key").size() != 32) return err = "header_key missing from prod.keys", false;
    Aes k1, k2;
    if (!k1.init(hk) || !k2.init(hk + 16)) return err = "AES provider unavailable", false;
    std::vector<u8> hdr(0xC00);
    if (!f.read(offset, hdr.data(), hdr.size())) return err = "cannot read NCA header", false;
    for (u64 s = 0; s < 6; s++) xts_decrypt_sector(k1, k2, &hdr[s * 0x200], 0x200, s);
    if (memcmp(&hdr[0x200], "NCA3", 4) != 0) return err = "not an NCA3 (wrong header_key?)", false;
    content_type_ = hdr[0x205];
    u8 kaek_idx = hdr[0x207];
    memcpy(&title_id_, &hdr[0x210], 8);
    u8 keygen = std::max(hdr[0x206], hdr[0x220]);
    int mkrev = keygen > 0 ? keygen - 1 : 0;
    char suffix[8];
    snprintf(suffix, sizeof(suffix), "_%02x", mkrev);
    const u8* rights = &hdr[0x230];
    bool has_rights = false;
    for (int i = 0; i < 16; i++) has_rights |= rights[i] != 0;
    u8 ctr_key[16];
    if (has_rights) {
        std::string rid = to_hex(rights, 16);
        auto it = keys.titles.find(rid);
        if (it == keys.titles.end() || it->second.size() != 16)
            return err = "no title key for rights id " + rid + " (ticket missing from the NSP and not in title.keys)", false;
        const u8* kek = keys.get("titlekek" + std::string(suffix));
        if (!kek) return err = "titlekek" + std::string(suffix) + " missing from prod.keys", false;
        Aes a;
        if (!a.init(kek)) return err = "AES provider unavailable", false;
        memcpy(ctr_key, it->second.data(), 16);
        a.decrypt(ctr_key, 16);
    } else {
        static const char* names[] = {"key_area_key_application", "key_area_key_ocean", "key_area_key_system"};
        if (kaek_idx > 2) return err = "bad key area index", false;
        std::string kn = names[kaek_idx] + std::string(suffix);
        const u8* kak = keys.get(kn);
        if (!kak) return err = kn + " missing from prod.keys", false;
        Aes a;
        if (!a.init(kak)) return err = "AES provider unavailable", false;
        u8 area[0x40];
        memcpy(area, &hdr[0x300], 0x40);
        a.decrypt(area, 0x40);
        memcpy(ctr_key, area + 0x20, 16);
    }
    if (!ctr_.init(ctr_key)) return err = "AES provider unavailable", false;

    for (int i = 0; i < 4; i++) {
        u32 start_blk, end_blk;
        memcpy(&start_blk, &hdr[0x240 + 16 * i], 4);
        memcpy(&end_blk, &hdr[0x244 + 16 * i], 4);
        if (end_blk == 0) continue;
        const u8* fsh = &hdr[0x400 + 0x200 * i];
        Section s;
        s.idx = i;
        s.start = (u64)start_blk * 0x200;
        s.end = (u64)end_blk * 0x200;
        s.fs_type = fsh[2];
        s.hash_type = fsh[3];
        s.enc_type = fsh[4];
        for (int k = 0; k < 8; k++) s.ctr_hi[k] = fsh[0x147 - k];
        const u8* h = fsh + 8;
        if (s.hash_type == 2) {
            u32 layers;
            memcpy(&layers, h + 0x24, 4);
            if (layers == 0 || layers > 8) return err = "bad PFS0 hash layer count", false;
            memcpy(&s.data_off, h + 0x28 + 16 * (layers - 1), 8);
            memcpy(&s.data_size, h + 0x30 + 16 * (layers - 1), 8);
        } else if (s.hash_type == 3) {
            if (memcmp(h, "IVFC", 4) != 0) return err = "bad IVFC header", false;
            memcpy(&s.data_off, h + 0x10 + 0x18 * 5, 8);
            memcpy(&s.data_size, h + 0x18 + 0x18 * 5, 8);
        } else {
            s.data_off = 0;
            s.data_size = s.end - s.start;
        }
        if (s.enc_type != 1 && s.enc_type != 3) {
            hw_log("nsp: section %d uses encryption type %u (unsupported, skipped)", i, s.enc_type);
            continue;
        }
        sections_.push_back(s);
    }
    return true;
}

bool Nca::read(const Section& s, u64 off, u8* dst, u64 size) const {
    if (off > s.end - s.start) return false;
    size = std::min(size, s.end - s.start - off);
    u64 abs = s.start + off;  // NCA-relative
    if (s.enc_type == 1) return f_->read(offset_ + abs, dst, size);
    constexpr u64 kChunk = 1 << 20;
    thread_local std::vector<u8> raw, ctr;
    while (size) {
        u64 aligned = abs & ~0xFull;
        u64 skip = abs - aligned;
        u64 want = std::min(size, kChunk - skip);
        u64 n = (skip + want + 15) & ~0xFull;
        raw.resize(n);
        ctr.resize(n);
        if (!f_->read(offset_ + aligned, raw.data(), n)) return false;
        u64 blk = aligned >> 4;
        for (u64 i = 0; i < n; i += 16) {
            u8* c = &ctr[i];
            memcpy(c, s.ctr_hi, 8);
            u64 v = blk + (i >> 4);
            for (int k = 0; k < 8; k++) c[15 - k] = (u8)(v >> (8 * k));
        }
        if (!ctr_.encrypt(ctr.data(), n)) return false;
        const u8* src = raw.data() + skip;
        const u8* pad = ctr.data() + skip;
        for (u64 i = 0; i < want; i++) dst[i] = src[i] ^ pad[i];
        dst += want;
        abs += want;
        size -= want;
    }
    return true;
}

// ---------------------------------------------------------------- state
struct State {
    std::string path;
    File file;
    Keys keys;
    Nca program;
    const Section* exefs = nullptr;
    const Section* romfs = nullptr;
    std::vector<PfsEntry> exefs_files;  // offsets relative to the exefs section
};
State* g_state = nullptr;

bool find_nsp(const std::string& exe_dir, std::string& out) {
    if (std::string e = env_str("HWDER_NSP"); !e.empty()) {
        out = e;
        return true;
    }
    for (const std::string& dir : {exe_dir, exe_dir + "\\game"}) {
        WIN32_FIND_DATAW fd;
        std::wstring pat = utf8_to_wide(dir + "\\*.nsp");
        HANDLE h = FindFirstFileW(pat.c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) continue;
        std::string best;
        do {
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            std::string name = wide_to_utf8(fd.cFileName);
            if (best.empty() || name < best) best = name;
        } while (FindNextFileW(h, &fd));
        FindClose(h);
        if (!best.empty()) {
            out = dir + "\\" + best;
            return true;
        }
    }
    return false;
}

}  // namespace

bool init(const std::string& exe_dir, std::string& error) {
    if (g_state) return true;
    std::string nsp;
    bool forced = !env_str("HWDER_NSP").empty();
    if (!forced && (!env_str("HWDER_EXEFS").empty() || !env_str("HWDER_ROMFS").empty())) return false;  // dev: directory mode
    if (!find_nsp(exe_dir, nsp)) return false;

    auto st = std::make_unique<State>();
    st->path = nsp;
    if (!st->file.open(nsp)) {
        error = "cannot open " + nsp;
        hw_log("nsp: %s", error.c_str());
        return false;
    }
    std::string keys_path;
    for (const std::string& c : key_candidates(exe_dir))
        if (file_exists(c)) {
            keys_path = c;
            break;
        }
    if (keys_path.empty()) {
        error = "prod.keys not found. Put your prod.keys next to hwder.exe (or in %USERPROFILE%\\.switch).";
        hw_log("nsp: %s", error.c_str());
        return false;
    }
    load_key_file(keys_path, st->keys.prod, false);
    std::string tk = keys_path.substr(0, keys_path.find_last_of("\\/") + 1) + "title.keys";
    load_key_file(tk, st->keys.titles, true);
    hw_log("nsp: %s", nsp.c_str());
    hw_log("nsp: keys from %s (%zu keys, %zu title keys)", keys_path.c_str(), st->keys.prod.size(), st->keys.titles.size());

    std::vector<PfsEntry> files;
    auto raw = [&](u64 off, u8* dst, u64 size) { return st->file.read(off, dst, size); };
    if (!parse_pfs0(raw, 0, files)) {
        error = "not a PFS0 NSP: " + nsp;
        hw_log("nsp: %s", error.c_str());
        return false;
    }
    // Tickets: rights id at body+0x160, encrypted title key at body+0x40.
    for (const PfsEntry& e : files) {
        if (e.name.size() < 4 || e.name.compare(e.name.size() - 4, 4, ".tik") != 0 || e.size < 0x2C0) continue;
        std::vector<u8> tik(e.size);
        if (!st->file.read(e.offset, tik.data(), tik.size())) continue;
        u32 sig;
        memcpy(&sig, tik.data(), 4);
        u32 body = sig == 0x10000 || sig == 0x10003 ? 0x240 : sig == 0x10001 || sig == 0x10004 ? 0x140 : 0x80;
        if (body + 0x170 > tik.size()) continue;
        st->keys.titles[to_hex(&tik[body + 0x160], 16)] = std::vector<u8>(&tik[body + 0x40], &tik[body + 0x50]);
    }
    // Program NCA: content type 0, not the .cnmt.nca metadata.
    bool found = false;
    std::string first_err;
    for (const PfsEntry& e : files) {
        if (e.name.size() < 4 || e.name.compare(e.name.size() - 4, 4, ".nca") != 0) continue;
        if (e.name.size() >= 9 && e.name.compare(e.name.size() - 9, 9, ".cnmt.nca") == 0) continue;
        Nca nca;
        std::string err;
        if (!nca.open(st->file, e.offset, e.size, st->keys, err)) {
            if (first_err.empty()) first_err = e.name + ": " + err;
            hw_log("nsp: %s: %s", e.name.c_str(), err.c_str());
            continue;
        }
        if (nca.content_type() != 0) continue;
        st->program = std::move(nca);
        found = true;
        hw_log("nsp: program NCA %s (title %016llx, %zu sections)", e.name.c_str(), (unsigned long long)st->program.title_id(),
               st->program.sections().size());
        break;
    }
    if (!found) {
        error = first_err.empty() ? "no program NCA in " + nsp : first_err;
        hw_log("nsp: %s", error.c_str());
        return false;
    }
    for (const Section& s : st->program.sections()) {
        if (s.fs_type == 0 && !st->romfs) st->romfs = &s;
        if (s.fs_type == 1 && !st->exefs) {
            std::vector<PfsEntry> pfs;
            auto rd = [&](u64 off, u8* dst, u64 size) { return st->program.read(s, s.data_off + off, dst, size); };
            if (!parse_pfs0(rd, 0, pfs)) continue;
            bool has_main = false;
            for (auto& p : pfs) has_main |= p.name == "main";
            if (!has_main) continue;  // the logo partition
            st->exefs = &s;
            for (auto& p : pfs) p.offset += s.data_off;
            st->exefs_files = std::move(pfs);
        }
    }
    if (!st->exefs) {
        error = "program NCA has no ExeFS";
        hw_log("nsp: %s", error.c_str());
        return false;
    }
    if (!st->romfs) hw_log("nsp: program NCA has no RomFS section");
    else
        hw_log("nsp: romfs section %d: 0x%llx bytes", st->romfs->idx, (unsigned long long)st->romfs->data_size);
    std::string names;
    for (auto& p : st->exefs_files) names += (names.empty() ? "" : " ") + p.name;
    hw_log("nsp: exefs: %s", names.c_str());
    g_state = st.release();
    return true;
}

bool active() { return g_state != nullptr; }

const std::string& path() {
    static std::string none;
    return g_state ? g_state->path : none;
}

bool read_exefs(const char* name, std::vector<u8>& out) {
    if (!g_state) return false;
    for (const PfsEntry& e : g_state->exefs_files)
        if (e.name == name) {
            out.resize(e.size);
            return g_state->program.read(*g_state->exefs, e.offset, out.data(), e.size);
        }
    return false;
}

std::vector<std::string> exefs_files() {
    std::vector<std::string> v;
    if (g_state)
        for (auto& e : g_state->exefs_files) v.push_back(e.name);
    return v;
}

u64 romfs_size() { return g_state && g_state->romfs ? g_state->romfs->data_size : 0; }

bool romfs_read(u64 off, u8* dst, u64 size) {
    if (!g_state || !g_state->romfs) return false;
    const Section& s = *g_state->romfs;
    if (off > s.data_size) return false;
    if (size > s.data_size - off) {
        memset(dst + (s.data_size - off), 0, (size_t)(size - (s.data_size - off)));
        size = s.data_size - off;
    }
    return size == 0 || g_state->program.read(s, s.data_off + off, dst, size);
}

}  // namespace hwfs::nsp
