// Virtual RomFS image.
//
// The SDK's RomFS driver reads the raw (level 3) RomFS image through the IStorage returned by
// fsp-srv OpenDataStorageByCurrentProcess and parses it itself. We synthesize that image from the
// extracted romfs directory: the header and metadata tables live in memory, file data is served
// straight from the host files (opened lazily, positional reads, no locks on the hot path).
//
// Image layout:  [header 0x50][pad to 0x200][file data, each file 0x10-aligned]
//                [dir hash table][dir entries][file hash table][file entries]
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <vector>

#include "fs_internal.h"
#include "nsp.h"

namespace hwfs {

namespace {

constexpr u32 kEmpty = 0xFFFFFFFFu;
constexpr u64 kDataStart = 0x200;
constexpr u64 kFileAlign = 0x10;

struct RomFsHeader {
    u64 header_size;
    u64 dir_hash_offset, dir_hash_size;
    u64 dir_meta_offset, dir_meta_size;
    u64 file_hash_offset, file_hash_size;
    u64 file_meta_offset, file_meta_size;
    u64 file_data_offset;
};
static_assert(sizeof(RomFsHeader) == 0x50);

u32 romfs_hash(u32 parent, const std::string& name) {
    u32 h = parent ^ 123456789u;
    for (unsigned char c : name) {
        h = (h >> 5) | (h << 27);
        h ^= c;
    }
    return h;
}

// Bucket count used by Nintendo's builder: small counts are odd, larger ones avoid small factors.
u32 hash_table_count(u32 n) {
    if (n < 3) return 3;
    if (n < 19) return n | 1;
    u32 c = n;
    auto bad = [](u32 v) {
        for (u32 p : {2u, 3u, 5u, 7u, 11u, 13u, 17u})
            if (v % p == 0) return true;
        return false;
    };
    while (bad(c)) c++;
    return c;
}

u32 align4(u32 v) { return (v + 3) & ~3u; }

struct BDir {
    std::string name;
    std::wstring host;
    u32 parent = 0;
    std::vector<u32> dirs, files;
    u32 off = 0;
};
struct BFile {
    std::string name;
    std::wstring host;
    u32 parent = 0;
    u64 size = 0;
    u64 data = 0;  // relative to file data start
    u32 off = 0;
};

class RomFsImage {
public:
    bool build(const std::wstring& root);
    Result read(u64 off, u8* dst, u64 size);
    u64 size() const { return total_; }
    size_t file_count() const { return files_.size(); }
    std::string describe(u64 off) const;  // what an image offset maps to (tracing)
    void self_test();                     // HWDER_FS_SELFTEST: parse the image like the SDK does

private:
    void scan(u32 dir_idx);
    HANDLE handle(u32 file);
    bool read_host(u32 file, u64 off, u8* dst, u64 size);

    std::vector<BDir> dirs_;
    std::vector<BFile> files_;
    std::vector<u8> head_;  // [0, kDataStart)
    std::vector<u8> meta_;  // [meta_start_, total_)
    u64 meta_start_ = 0, total_ = 0;
    struct Span {
        u64 start, size;  // absolute image offsets
        u32 file;
    };
    std::vector<Span> spans_;  // non-empty files in data order
    std::unique_ptr<std::atomic<HANDLE>[]> handles_;
};

void RomFsImage::scan(u32 di) {
    std::wstring pattern = dirs_[di].host + L"\\*";
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileExW(pattern.c_str(), FindExInfoBasic, &fd, FindExSearchNameMatch, nullptr,
                                FIND_FIRST_EX_LARGE_FETCH);
    if (h == INVALID_HANDLE_VALUE) return;
    std::vector<std::pair<std::string, WIN32_FIND_DATAW>> sub_dirs, sub_files;
    do {
        if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
        std::string name = wide_to_utf8(fd.cFileName);
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) sub_dirs.emplace_back(std::move(name), fd);
        else sub_files.emplace_back(std::move(name), fd);
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    auto by_name = [](const auto& a, const auto& b) { return a.first < b.first; };
    std::sort(sub_dirs.begin(), sub_dirs.end(), by_name);
    std::sort(sub_files.begin(), sub_files.end(), by_name);
    for (auto& [name, f] : sub_files) {
        BFile bf;
        bf.name = name;
        bf.host = dirs_[di].host + L"\\" + f.cFileName;
        bf.parent = di;
        bf.size = ((u64)f.nFileSizeHigh << 32) | f.nFileSizeLow;
        dirs_[di].files.push_back((u32)files_.size());
        files_.push_back(std::move(bf));
    }
    std::vector<u32> children;
    for (auto& [name, f] : sub_dirs) {
        BDir bd;
        bd.name = name;
        bd.host = dirs_[di].host + L"\\" + f.cFileName;
        bd.parent = di;
        children.push_back((u32)dirs_.size());
        dirs_.push_back(std::move(bd));
    }
    dirs_[di].dirs = children;
    for (u32 c : children) scan(c);
}

bool RomFsImage::build(const std::wstring& root) {
    DWORD attr = GetFileAttributesW(root.c_str());
    bool exists = attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
    dirs_.push_back(BDir{});
    dirs_[0].host = root;
    if (exists) scan(0);

    // Entry offsets: directories breadth-first-ish (creation order), files grouped per directory.
    u32 dir_meta = 0;
    for (auto& d : dirs_) {
        d.off = dir_meta;
        dir_meta += 0x18 + align4((u32)d.name.size());
    }
    // Files: order of their directories, so siblings are contiguous.
    u32 file_meta = 0;
    u64 data = 0;
    std::vector<u32> data_order;
    data_order.reserve(files_.size());
    for (auto& d : dirs_)
        for (u32 fi : d.files) {
            BFile& f = files_[fi];
            f.off = file_meta;
            file_meta += 0x20 + align4((u32)f.name.size());
            data = (data + kFileAlign - 1) & ~(kFileAlign - 1);
            f.data = data;
            data += f.size;
            data_order.push_back(fi);
        }

    u32 dir_buckets = hash_table_count((u32)dirs_.size());
    u32 file_buckets = hash_table_count((u32)files_.size());
    RomFsHeader hdr{};
    hdr.header_size = sizeof(RomFsHeader);
    hdr.file_data_offset = kDataStart;
    meta_start_ = (kDataStart + data + 0xF) & ~0xFull;
    hdr.dir_hash_offset = meta_start_;
    hdr.dir_hash_size = (u64)dir_buckets * 4;
    hdr.dir_meta_offset = hdr.dir_hash_offset + hdr.dir_hash_size;
    hdr.dir_meta_size = dir_meta;
    hdr.file_hash_offset = hdr.dir_meta_offset + hdr.dir_meta_size;
    hdr.file_hash_size = (u64)file_buckets * 4;
    hdr.file_meta_offset = hdr.file_hash_offset + hdr.file_hash_size;
    hdr.file_meta_size = file_meta;
    total_ = hdr.file_meta_offset + hdr.file_meta_size;

    head_.assign(kDataStart, 0);
    memcpy(head_.data(), &hdr, sizeof(hdr));
    meta_.assign(total_ - meta_start_, 0);
    u32* dir_hash = (u32*)(meta_.data() + (hdr.dir_hash_offset - meta_start_));
    u8* dir_tab = meta_.data() + (hdr.dir_meta_offset - meta_start_);
    u32* file_hash = (u32*)(meta_.data() + (hdr.file_hash_offset - meta_start_));
    u8* file_tab = meta_.data() + (hdr.file_meta_offset - meta_start_);
    std::fill(dir_hash, dir_hash + dir_buckets, kEmpty);
    std::fill(file_hash, file_hash + file_buckets, kEmpty);

    for (size_t i = 0; i < dirs_.size(); i++) {
        BDir& d = dirs_[i];
        u32* e = (u32*)(dir_tab + d.off);
        u32 parent_off = dirs_[d.parent].off;
        e[0] = parent_off;
        // Sibling: next directory in the parent's child list.
        e[1] = kEmpty;
        if (i) {
            auto& sib = dirs_[d.parent].dirs;
            auto it = std::find(sib.begin(), sib.end(), (u32)i);
            if (it + 1 != sib.end()) e[1] = dirs_[*(it + 1)].off;
        }
        e[2] = d.dirs.empty() ? kEmpty : dirs_[d.dirs[0]].off;
        e[3] = d.files.empty() ? kEmpty : files_[d.files[0]].off;
        u32 b = romfs_hash(parent_off, d.name) % dir_buckets;
        e[4] = dir_hash[b];
        dir_hash[b] = d.off;
        e[5] = (u32)d.name.size();
        memcpy(e + 6, d.name.data(), d.name.size());
    }
    for (auto& d : dirs_)
        for (size_t k = 0; k < d.files.size(); k++) {
            BFile& f = files_[d.files[k]];
            u8* e = file_tab + f.off;
            u32 parent_off = d.off;
            u32 sibling = k + 1 < d.files.size() ? files_[d.files[k + 1]].off : kEmpty;
            u64 offset = f.data, size = f.size;
            u32 b = romfs_hash(parent_off, f.name) % file_buckets;
            u32 next = file_hash[b];
            file_hash[b] = f.off;
            u32 name_size = (u32)f.name.size();
            memcpy(e + 0x00, &parent_off, 4);
            memcpy(e + 0x04, &sibling, 4);
            memcpy(e + 0x08, &offset, 8);
            memcpy(e + 0x10, &size, 8);
            memcpy(e + 0x18, &next, 4);
            memcpy(e + 0x1C, &name_size, 4);
            memcpy(e + 0x20, f.name.data(), name_size);
        }

    for (u32 fi : data_order)
        if (files_[fi].size) spans_.push_back({kDataStart + files_[fi].data, files_[fi].size, fi});
    handles_ = std::make_unique<std::atomic<HANDLE>[]>(files_.size());
    for (size_t i = 0; i < files_.size(); i++) handles_[i].store(nullptr, std::memory_order_relaxed);
    return exists;
}

HANDLE RomFsImage::handle(u32 fi) {
    HANDLE h = handles_[fi].load(std::memory_order_acquire);
    if (h) return h;
    HANDLE nh = CreateFileW(files_[fi].host.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (nh == INVALID_HANDLE_VALUE) {
        hw_log("romfs: cannot open host file '%s' (error %lu)", wide_to_utf8(files_[fi].host).c_str(), GetLastError());
        return INVALID_HANDLE_VALUE;
    }
    HANDLE expected = nullptr;
    if (!handles_[fi].compare_exchange_strong(expected, nh, std::memory_order_acq_rel)) {
        CloseHandle(nh);
        return expected;
    }
    return nh;
}

bool RomFsImage::read_host(u32 fi, u64 off, u8* dst, u64 size) {
    HANDLE h = handle(fi);
    if (h == INVALID_HANDLE_VALUE) return false;
    while (size) {
        DWORD chunk = (DWORD)std::min<u64>(size, 0x40000000ull);
        OVERLAPPED ov{};
        ov.Offset = (DWORD)off;
        ov.OffsetHigh = (DWORD)(off >> 32);
        DWORD got = 0;
        if (!ReadFile(h, dst, chunk, &got, &ov) || got == 0) return false;
        dst += got;
        off += got;
        size -= got;
    }
    return true;
}

Result RomFsImage::read(u64 off, u8* dst, u64 size) {
    if (off > total_) return ResultOutOfRange;
    if (size > total_ - off) {
        memset(dst + (total_ - off), 0, (size_t)(size - (total_ - off)));
        size = total_ - off;
    }
    while (size) {
        u64 n;
        if (off < kDataStart) {
            n = std::min<u64>(size, kDataStart - off);
            memcpy(dst, head_.data() + off, (size_t)n);
        } else if (off >= meta_start_) {
            n = size;
            memcpy(dst, meta_.data() + (off - meta_start_), (size_t)n);
        } else {
            // Last span starting at or before off.
            auto it = std::upper_bound(spans_.begin(), spans_.end(), off,
                                       [](u64 v, const Span& s) { return v < s.start; });
            const Span* s = it == spans_.begin() ? nullptr : &*(it - 1);
            if (s && off < s->start + s->size) {
                n = std::min<u64>(size, s->start + s->size - off);
                if (!read_host(s->file, off - s->start, dst, n)) {
                    static std::atomic<int> warned{0};
                    if (warned++ < 16)
                        hw_log("romfs: host read failed: %s +0x%llx size 0x%llx", files_[s->file].name.c_str(),
                               (unsigned long long)(off - s->start), (unsigned long long)n);
                    memset(dst, 0, (size_t)n);
                }
            } else {
                u64 next = it == spans_.end() ? meta_start_ : it->start;
                n = std::min<u64>(size, next - off);
                memset(dst, 0, (size_t)n);  // alignment padding
            }
        }
        dst += n;
        off += n;
        size -= n;
    }
    return 0;
}

std::string RomFsImage::describe(u64 off) const {
    if (off < kDataStart) return "[header]";
    if (off >= meta_start_) return "[metadata]";
    auto it = std::upper_bound(spans_.begin(), spans_.end(), off, [](u64 v, const Span& s) { return v < s.start; });
    if (it == spans_.begin()) return "[pad]";
    const Span& s = *(it - 1);
    if (off >= s.start + s.size) return "[pad]";
    std::string path = files_[s.file].name;
    for (u32 d = files_[s.file].parent; d; d = dirs_[d].parent) path = dirs_[d].name + "/" + path;
    char buf[32];
    snprintf(buf, sizeof(buf), " @0x%llx", (unsigned long long)(off - s.start));
    return "/" + path + buf;
}

// Resolves every file through the image's own hash tables (as the SDK's RomFsFileSystem does, using
// only Read) and compares its size and head/tail bytes against the host file.
void RomFsImage::self_test() {
    auto t0 = std::chrono::steady_clock::now();
    RomFsHeader h;
    read(0, (u8*)&h, sizeof(h));
    auto rd32 = [&](u64 off) {
        u32 v = 0;
        read(off, (u8*)&v, 4);
        return v;
    };
    auto entry_name = [&](u64 off, u32 len) {
        std::string n(len, ' ');
        if (len) read(off, (u8*)n.data(), len);
        return n;
    };
    auto find_dir = [&](u32 parent, const std::string& name) -> u32 {
        u32 b = romfs_hash(parent, name) % (u32)(h.dir_hash_size / 4);
        for (u32 cur = rd32(h.dir_hash_offset + b * 4); cur != kEmpty;) {
            u64 e = h.dir_meta_offset + cur;
            if (rd32(e) == parent && entry_name(e + 0x18, rd32(e + 0x14)) == name) return cur;
            cur = rd32(e + 0x10);
        }
        return kEmpty;
    };
    size_t ok = 0, bad = 0;
    std::vector<u8> a(4096), b(4096);
    for (size_t fi = 0; fi < files_.size(); fi++) {
        std::vector<std::string> parts{files_[fi].name};
        for (u32 d = files_[fi].parent; d; d = dirs_[d].parent) parts.insert(parts.begin(), dirs_[d].name);
        u32 dir = 0;
        for (size_t k = 0; k + 1 < parts.size() && dir != kEmpty; k++) dir = find_dir(dir, parts[k]);
        u32 found = kEmpty;
        if (dir != kEmpty) {
            u32 bk = romfs_hash(dir, parts.back()) % (u32)(h.file_hash_size / 4);
            for (u32 cur = rd32(h.file_hash_offset + bk * 4); cur != kEmpty;) {
                u64 e = h.file_meta_offset + cur;
                if (rd32(e) == dir && entry_name(e + 0x20, rd32(e + 0x1C)) == parts.back()) {
                    found = cur;
                    break;
                }
                cur = rd32(e + 0x18);
            }
        }
        bool good = found != kEmpty;
        if (good) {
            u64 e = h.file_meta_offset + found, off = 0, size = 0;
            read(e + 8, (u8*)&off, 8);
            read(e + 16, (u8*)&size, 8);
            good = size == files_[fi].size;
            for (int tail = 0; good && tail < 2; tail++) {
                u64 n = std::min<u64>(size, a.size());
                u64 pos = tail ? size - n : 0;
                read(h.file_data_offset + off + pos, a.data(), n);
                good = read_host((u32)fi, pos, b.data(), n) && !memcmp(a.data(), b.data(), (size_t)n);
            }
        }
        if (good) ok++;
        else if (bad++ < 10) hw_log("romfs: selftest FAILED for %s", files_[fi].name.c_str());
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    hw_log("romfs: selftest: %zu ok, %zu failed (%.0f ms)", ok, bad, ms);
}

RomFsImage* g_image = nullptr;
std::once_flag g_image_once;

RomFsImage* image() {
    std::call_once(g_image_once, [] {
        auto* img = new RomFsImage;
        auto t0 = std::chrono::steady_clock::now();
        const std::wstring& root = romfs_host_dir();
        if (!img->build(root))
            hw_log("romfs: directory '%s' not found - RomFS will be empty", wide_to_utf8(root).c_str());
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        hw_log("romfs: built virtual image: %zu files, %.2f GiB, in %.1f ms", img->file_count(),
               img->size() / (1024.0 * 1024.0 * 1024.0), ms);
        g_image = img;
        if (const char* e = getenv("HWDER_FS_SELFTEST"); e && *e && *e != '0') img->self_test();
    });
    return g_image;
}

class IStorage : public FsService {
public:
    IStorage() : FsService("romfs:IStorage") {
        reg(0, [](Request& rq, Response& rs) {  // Read(s64 offset, s64 size, B buffer)
            u64 off = rq.pop<u64>();
            u64 size = rq.pop<u64>();
            ipc::Buffer b = rq.out_buffer();
            if (b.size < size) size = b.size;
            if ((s64)off < 0) return void(rs.result = ResultInvalidOffset);
            if (!b.addr || !size) return;
            // Slow-read diagnostics: a single RomFS read over 20 ms stalls the game thread that issued it
            // (movie start, scene loads); logs the size so the decrypt/IO throughput can be judged.
            LARGE_INTEGER rt0, rt1, rf;
            QueryPerformanceFrequency(&rf);
            QueryPerformanceCounter(&rt0);
            struct ReadTimer {
                LARGE_INTEGER *t0, *f;
                u64 off, size;
                ~ReadTimer() {
                    LARGE_INTEGER t1;
                    QueryPerformanceCounter(&t1);
                    double ms = (t1.QuadPart - t0->QuadPart) * 1000.0 / f->QuadPart;
                    if (ms > 20.0)
                        hw_log("fs: slow: romfs read of %llu KiB at 0x%llx took %.0f ms (%.0f MB/s)", (unsigned long long)(size >> 10),
                               (unsigned long long)off, ms, size / 1048576.0 / (ms / 1000.0));
                }
            } read_timer{&rt0, &rf, off, size};
            (void)rt1;
            if (nsp::active()) {  // served straight out of the NSP's program NCA
                if (!nsp::romfs_read(off, (u8*)b.addr, size)) {
                    static std::atomic<int> warned{0};
                    if (warned++ < 16) hw_log("fs: nsp romfs read 0x%llx+0x%llx FAILED", (unsigned long long)off, (unsigned long long)size);
                    memset((void*)(uintptr_t)b.addr, 0, (size_t)size);
                }
                if (trace_enabled()) hw_log("fs: nsp romfs read 0x%llx+0x%llx", (unsigned long long)off, (unsigned long long)size);
                return;
            }
            RomFsImage* img = image();
            rs.result = img->read(off, (u8*)b.addr, size);
            if (trace_enabled())
                hw_log("fs: romfs read 0x%llx+0x%llx %s -> 0x%x", (unsigned long long)off, (unsigned long long)size,
                       img->describe(off).c_str(), rs.result);
        });
        reg(1, [](Request&, Response& rs) { rs.result = ResultUnsupportedOperation; });  // Write
        reg(2, [](Request&, Response&) {});                                             // Flush
        reg(3, [](Request&, Response& rs) { rs.result = ResultUnsupportedOperation; });  // SetSize
        reg(4, [](Request&, Response& rs) { rs.push<u64>(nsp::active() ? nsp::romfs_size() : image()->size()); });  // GetSize
        reg(5, [](Request&, Response& rs) {                                             // OperateRange
            u8 info[0x40] = {};
            rs.data.insert(rs.data.end(), info, info + sizeof(info));
        });
    }
};

}  // namespace

const std::wstring& romfs_host_dir() {
    static std::wstring dir = [] {
        std::wstring p;
        if (const wchar_t* e = _wgetenv(L"HWDER_ROMFS")) p = e;
        else {
            p = utf8_to_wide(hw_romfs_path());
        }
        wchar_t full[32768];
        DWORD n = GetFullPathNameW(p.c_str(), 32768, full, nullptr);
        if (n && n < 32768) p = full;
        while (!p.empty() && (p.back() == L'\\' || p.back() == L'/')) p.pop_back();
        if (p.rfind(L"\\\\", 0) != 0) p = L"\\\\?\\" + p;  // long-path form
        return p;
    }();
    return dir;
}

ServicePtr open_romfs_storage() {
    if (!nsp::active()) image();
    return std::make_shared<IStorage>();
}

}  // namespace hwfs
