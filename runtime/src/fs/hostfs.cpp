// IFileSystem / IFile / IDirectory backed by a host directory (save data, SD card).
#include <windows.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "fs_internal.h"

namespace hwfs {

namespace {

constexpr u32 kEntryDir = 0, kEntryFile = 1;
constexpr u32 kOpenRead = 1, kOpenWrite = 2, kOpenAppend = 4;
constexpr u32 kDirFilterDirs = 1, kDirFilterFiles = 2, kDirNoFileSize = 0x80000000u;

// Guest path -> normalized component list ("." and ".." resolved, clamped at the root).
bool normalize(const std::string& in, std::string* out) {
    std::string p = in;
    size_t colon = p.find(':');
    if (colon != std::string::npos && p.find('/') > colon) p = p.substr(colon + 1);  // strip "mount:"
    std::vector<std::string> parts;
    size_t i = 0;
    while (i <= p.size()) {
        size_t j = p.find_first_of("/\\", i);
        if (j == std::string::npos) j = p.size();
        std::string c = p.substr(i, j - i);
        if (c == "..") {
            if (!parts.empty()) parts.pop_back();
        } else if (!c.empty() && c != ".") {
            parts.push_back(std::move(c));
        }
        i = j + 1;
    }
    out->clear();
    for (auto& c : parts) *out += "/" + c;
    if (out->empty()) *out = "/";
    return true;
}

u64 filetime_to_posix(const FILETIME& ft) {
    u64 t = ((u64)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    return t < 116444736000000000ull ? 0 : (t - 116444736000000000ull) / 10000000ull;
}

bool is_dir(const std::wstring& p) {
    DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}
bool exists(const std::wstring& p) { return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES; }

void create_dirs(const std::wstring& p) {
    if (p.empty() || is_dir(p)) return;
    size_t s = p.find_last_of(L'\\');
    if (s != std::wstring::npos && s > 0) create_dirs(p.substr(0, s));
    CreateDirectoryW(p.c_str(), nullptr);
}

// Deletes everything below dir (not dir itself).
bool clean_dir(const std::wstring& dir) {
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileExW((dir + L"\\*").c_str(), FindExInfoBasic, &fd, FindExSearchNameMatch, nullptr, 0);
    if (h == INVALID_HANDLE_VALUE) return true;
    bool ok = true;
    do {
        if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
        std::wstring p = dir + L"\\" + fd.cFileName;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            ok &= clean_dir(p);
            ok &= RemoveDirectoryW(p.c_str()) != 0;
        } else {
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_READONLY) SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_NORMAL);
            ok &= DeleteFileW(p.c_str()) != 0;
        }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    return ok;
}

// Shared per-filesystem state: the root, plus how the guest named it in logs.
struct FsRoot {
    std::wstring root;  // "\\?\C:\...\save\<title>\<user>" (no trailing slash)
    std::string label;
    bool read_only = false;
    std::wstring host(const std::string& norm) const {
        if (norm == "/") return root;
        std::wstring w = utf8_to_wide(norm);
        std::replace(w.begin(), w.end(), L'/', L'\\');
        return root + w;
    }
};

class IFile : public FsService {
public:
    IFile(HANDLE h, u32 mode, std::string path) : FsService("IFile"), h_(h), mode_(mode), path_(std::move(path)) {
        reg(0, [this](Request& rq, Response& rs) {  // Read(u32 option, s64 offset, s64 size, B buffer) -> s64
            rq.pop<u32>();
            s64 off = rq.pop<s64>();
            s64 size = rq.pop<s64>();
            ipc::Buffer b = rq.out_buffer();
            if (off < 0) return void(rs.result = ResultInvalidOffset);
            if (size < 0) return void(rs.result = ResultInvalidSize);
            if (!(mode_ & kOpenRead)) return void(rs.result = ResultReadNotPermitted);
            u64 want = std::min<u64>((u64)size, b.size), done = 0;
            u8* dst = (u8*)b.addr;
            while (done < want) {
                OVERLAPPED ov{};
                u64 o = (u64)off + done;
                ov.Offset = (DWORD)o;
                ov.OffsetHigh = (DWORD)(o >> 32);
                DWORD got = 0;
                DWORD chunk = (DWORD)std::min<u64>(want - done, 0x40000000ull);
                if (!ReadFile(h_, dst + done, chunk, &got, &ov) || got == 0) break;
                done += got;
            }
            if (trace_enabled())
                hw_log("fs: file read %s 0x%llx+0x%llx -> 0x%llx", path_.c_str(), (unsigned long long)off,
                       (unsigned long long)want, (unsigned long long)done);
            rs.push<u64>(done);
        });
        reg(1, [this](Request& rq, Response& rs) {  // Write(u32 option, s64 offset, s64 size, A buffer)
            u32 option = rq.pop<u32>();
            s64 off = rq.pop<s64>();
            s64 size = rq.pop<s64>();
            ipc::Buffer b = rq.in_buffer();
            if (off < 0) return void(rs.result = ResultInvalidOffset);
            if (size < 0) return void(rs.result = ResultInvalidSize);
            if (!(mode_ & kOpenWrite)) return void(rs.result = ResultWriteNotPermitted);
            u64 want = std::min<u64>((u64)size, b.size), done = 0;
            const u8* src = (const u8*)b.addr;
            while (done < want) {
                OVERLAPPED ov{};
                u64 o = (u64)off + done;
                ov.Offset = (DWORD)o;
                ov.OffsetHigh = (DWORD)(o >> 32);
                DWORD put = 0;
                DWORD chunk = (DWORD)std::min<u64>(want - done, 0x40000000ull);
                if (!WriteFile(h_, src + done, chunk, &put, &ov) || put == 0) {
                    hw_log("fs: write to %s failed (error %lu)", path_.c_str(), GetLastError());
                    return void(rs.result = ResultUsableSpaceNotEnough);
                }
                done += put;
            }
            if (option & 1) FlushFileBuffers(h_);
            if (trace_enabled())
                hw_log("fs: file write %s 0x%llx+0x%llx", path_.c_str(), (unsigned long long)off, (unsigned long long)want);
        });
        reg(2, [this](Request&, Response&) {  // Flush
            if (mode_ & kOpenWrite) FlushFileBuffers(h_);
        });
        reg(3, [this](Request& rq, Response& rs) {  // SetSize(s64)
            s64 size = rq.pop<s64>();
            if (size < 0) return void(rs.result = ResultInvalidSize);
            if (!(mode_ & kOpenWrite)) return void(rs.result = ResultWriteNotPermitted);
            FILE_END_OF_FILE_INFO eof;
            eof.EndOfFile.QuadPart = size;
            if (!SetFileInformationByHandle(h_, FileEndOfFileInfo, &eof, sizeof(eof)))
                rs.result = ResultUsableSpaceNotEnough;
        });
        reg(4, [this](Request&, Response& rs) {  // GetSize -> s64
            LARGE_INTEGER sz{};
            GetFileSizeEx(h_, &sz);
            rs.push<s64>(sz.QuadPart);
        });
        reg(5, [](Request&, Response& rs) {  // OperateRange -> QueryRangeInfo (0x40)
            u8 info[0x40] = {};
            rs.data.insert(rs.data.end(), info, info + sizeof(info));
        });
    }
    ~IFile() override { CloseHandle(h_); }

private:
    HANDLE h_;
    u32 mode_;
    std::string path_;
};

#pragma pack(push, 1)
struct DirectoryEntry {
    char name[0x301];
    u8 pad0[3];
    u8 type;
    u8 pad1[3];
    s64 size;
};
#pragma pack(pop)
static_assert(sizeof(DirectoryEntry) == 0x310);

class IDirectory : public FsService {
public:
    IDirectory(std::vector<DirectoryEntry> entries) : FsService("IDirectory"), entries_(std::move(entries)) {
        reg(0, [this](Request& rq, Response& rs) {  // Read(B buffer) -> s64 count
            ipc::Buffer b = rq.out_buffer();
            std::lock_guard<std::mutex> l(m_);
            u64 n = std::min<u64>(b.size / sizeof(DirectoryEntry), entries_.size() - pos_);
            if (n) memcpy((void*)b.addr, &entries_[pos_], (size_t)n * sizeof(DirectoryEntry));
            pos_ += (size_t)n;
            rs.push<s64>((s64)n);
        });
        reg(1, [this](Request&, Response& rs) { rs.push<s64>((s64)entries_.size()); });  // GetEntryCount
    }

private:
    std::mutex m_;
    std::vector<DirectoryEntry> entries_;
    size_t pos_ = 0;
};

class IFileSystem : public FsService {
public:
    explicit IFileSystem(std::shared_ptr<FsRoot> fs) : FsService("IFileSystem"), fs_(std::move(fs)) {
        // All path-taking commands: path in the first X (type 0x19) buffer.
        reg(0, [this](Request& rq, Response& rs) {  // CreateFile(u32 option, s64 size, path)
            rq.pop<u32>();
            s64 size = rq.pop<s64>();
            std::string p;
            std::wstring h = path(rq, 0, &p);
            if (ro(rs)) return;
            if (exists(h)) return fail(rs, ResultPathAlreadyExists, "CreateFile", p);
            if (!parent_is_dir(h)) return fail(rs, ResultPathNotFound, "CreateFile", p);
            HANDLE f = CreateFileW(h.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                   nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (f == INVALID_HANDLE_VALUE) return fail(rs, ResultPathNotFound, "CreateFile", p);
            if (size > 0) {
                FILE_END_OF_FILE_INFO eof;
                eof.EndOfFile.QuadPart = size;
                if (!SetFileInformationByHandle(f, FileEndOfFileInfo, &eof, sizeof(eof)))
                    rs.result = ResultUsableSpaceNotEnough;
            }
            CloseHandle(f);
            log_op("CreateFile", p, rs.result, size);
        });
        reg(1, [this](Request& rq, Response& rs) {  // DeleteFile(path)
            std::string p;
            std::wstring h = path(rq, 0, &p);
            if (ro(rs)) return;
            DWORD a = GetFileAttributesW(h.c_str());
            if (a == INVALID_FILE_ATTRIBUTES || (a & FILE_ATTRIBUTE_DIRECTORY))
                return fail(rs, ResultPathNotFound, "DeleteFile", p);
            if (!DeleteFileW(h.c_str())) rs.result = ResultTargetLocked;
            log_op("DeleteFile", p, rs.result);
        });
        reg(2, [this](Request& rq, Response& rs) {  // CreateDirectory(path)
            std::string p;
            std::wstring h = path(rq, 0, &p);
            if (ro(rs)) return;
            if (exists(h)) return fail(rs, ResultPathAlreadyExists, "CreateDirectory", p);
            if (!parent_is_dir(h)) return fail(rs, ResultPathNotFound, "CreateDirectory", p);
            if (!CreateDirectoryW(h.c_str(), nullptr)) rs.result = ResultPathNotFound;
            log_op("CreateDirectory", p, rs.result);
        });
        reg(3, [this](Request& rq, Response& rs) {  // DeleteDirectory(path)
            std::string p;
            std::wstring h = path(rq, 0, &p);
            if (ro(rs)) return;
            if (!is_dir(h)) return fail(rs, ResultPathNotFound, "DeleteDirectory", p);
            if (!RemoveDirectoryW(h.c_str()))
                rs.result = GetLastError() == ERROR_DIR_NOT_EMPTY ? ResultDirectoryNotEmpty : ResultTargetLocked;
            log_op("DeleteDirectory", p, rs.result);
        });
        reg(4, [this](Request& rq, Response& rs) {  // DeleteDirectoryRecursively(path)
            std::string p;
            std::wstring h = path(rq, 0, &p);
            if (ro(rs)) return;
            if (!is_dir(h)) return fail(rs, ResultPathNotFound, "DeleteDirectoryRecursively", p);
            if (p == "/") rs.result = clean_dir(h) ? 0 : ResultTargetLocked;
            else if (!clean_dir(h) || !RemoveDirectoryW(h.c_str())) rs.result = ResultTargetLocked;
            log_op("DeleteDirectoryRecursively", p, rs.result);
        });
        reg(5, [this](Request& rq, Response& rs) { rename(rq, rs, false); });  // RenameFile(old, new)
        reg(6, [this](Request& rq, Response& rs) { rename(rq, rs, true); });   // RenameDirectory(old, new)
        reg(7, [this](Request& rq, Response& rs) {  // GetEntryType(path) -> u32
            std::string p;
            std::wstring h = path(rq, 0, &p);
            DWORD a = GetFileAttributesW(h.c_str());
            if (a == INVALID_FILE_ATTRIBUTES) {
                if (trace_enabled()) hw_log("fs: [%s] GetEntryType %s -> not found", fs_->label.c_str(), p.c_str());
                return void(rs.result = ResultPathNotFound);
            }
            rs.push<u32>((a & FILE_ATTRIBUTE_DIRECTORY) ? kEntryDir : kEntryFile);
        });
        reg(8, [this](Request& rq, Response& rs) {  // OpenFile(u32 mode, path) -> IFile
            u32 mode = rq.pop<u32>();
            std::string p;
            std::wstring h = path(rq, 0, &p);
            if (fs_->read_only && (mode & (kOpenWrite | kOpenAppend))) return void(rs.result = ResultPermissionDenied);
            DWORD access = 0;
            if (mode & kOpenRead) access |= GENERIC_READ;
            if (mode & (kOpenWrite | kOpenAppend)) access |= GENERIC_WRITE;
            if (!access) access = GENERIC_READ;
            if (is_dir(h)) return fail(rs, ResultPathNotFound, "OpenFile", p);
            HANDLE f = CreateFileW(h.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (f == INVALID_HANDLE_VALUE) {
                DWORD e = GetLastError();
                return fail(rs, e == ERROR_SHARING_VIOLATION ? ResultTargetLocked : ResultPathNotFound, "OpenFile", p);
            }
            log_op("OpenFile", p, 0, mode);
            rs.push_object(std::make_shared<IFile>(f, mode, fs_->label + ":" + p));
        });
        reg(9, [this](Request& rq, Response& rs) {  // OpenDirectory(u32 filter, path) -> IDirectory
            u32 filter = rq.pop<u32>();
            std::string p;
            std::wstring h = path(rq, 0, &p);
            if (!is_dir(h)) return fail(rs, ResultPathNotFound, "OpenDirectory", p);
            std::vector<DirectoryEntry> entries;
            WIN32_FIND_DATAW fd;
            HANDLE fh = FindFirstFileExW((h + L"\\*").c_str(), FindExInfoBasic, &fd, FindExSearchNameMatch, nullptr, 0);
            if (fh != INVALID_HANDLE_VALUE) {
                do {
                    if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
                    bool dir = fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY;
                    if (dir && !(filter & kDirFilterDirs)) continue;
                    if (!dir && !(filter & kDirFilterFiles)) continue;
                    DirectoryEntry e{};
                    std::string n = wide_to_utf8(fd.cFileName);
                    memcpy(e.name, n.data(), std::min<size_t>(n.size(), 0x300));
                    e.type = dir ? kEntryDir : kEntryFile;
                    if (!dir && !(filter & kDirNoFileSize))
                        e.size = (s64)(((u64)fd.nFileSizeHigh << 32) | fd.nFileSizeLow);
                    entries.push_back(e);
                } while (FindNextFileW(fh, &fd));
                FindClose(fh);
            }
            log_op("OpenDirectory", p, 0, entries.size());
            rs.push_object(std::make_shared<IDirectory>(std::move(entries)));
        });
        reg(10, [](Request&, Response&) {});  // Commit: writes go straight to the host files
        reg(11, [this](Request&, Response& rs) { rs.push<s64>(free_space()); });       // GetFreeSpaceSize
        reg(12, [this](Request&, Response& rs) { rs.push<s64>(total_space()); });      // GetTotalSpaceSize
        reg(13, [this](Request& rq, Response& rs) {  // CleanDirectoryRecursively(path)
            std::string p;
            std::wstring h = path(rq, 0, &p);
            if (ro(rs)) return;
            if (!is_dir(h)) return fail(rs, ResultPathNotFound, "CleanDirectoryRecursively", p);
            if (!clean_dir(h)) rs.result = ResultTargetLocked;
            log_op("CleanDirectoryRecursively", p, rs.result);
        });
        reg(14, [this](Request& rq, Response& rs) {  // GetFileTimeStampRaw(path) -> 0x20 bytes
            std::string p;
            std::wstring h = path(rq, 0, &p);
            WIN32_FILE_ATTRIBUTE_DATA d;
            if (!GetFileAttributesExW(h.c_str(), GetFileExInfoStandard, &d)) return void(rs.result = ResultPathNotFound);
            rs.push<u64>(filetime_to_posix(d.ftCreationTime));
            rs.push<u64>(filetime_to_posix(d.ftLastAccessTime));
            rs.push<u64>(filetime_to_posix(d.ftLastWriteTime));
            rs.push<u64>(1);  // is_valid
        });
        reg(15, [](Request&, Response&) {});  // QueryEntry: nothing to report
    }

private:
    std::wstring path(Request& rq, size_t i, std::string* norm) {
        normalize(rq.in_string(i), norm);
        return fs_->host(*norm);
    }
    bool ro(Response& rs) {
        if (!fs_->read_only) return false;
        rs.result = ResultPermissionDenied;
        return true;
    }
    static bool parent_is_dir(const std::wstring& h) {
        size_t s = h.find_last_of(L'\\');
        return s != std::wstring::npos && is_dir(h.substr(0, s));
    }
    void fail(Response& rs, Result r, const char* op, const std::string& p) {
        rs.result = r;
        log_op(op, p, r);
    }
    void log_op(const char* op, const std::string& p, Result r, u64 arg = 0) {
        hw_log("fs: [%s] %s %s (0x%llx) -> 0x%x", fs_->label.c_str(), op, p.c_str(), (unsigned long long)arg, r);
    }
    void rename(Request& rq, Response& rs, bool dir) {
        const char* op = dir ? "RenameDirectory" : "RenameFile";
        std::string a, b;
        std::wstring ha = path(rq, 0, &a), hb = path(rq, 1, &b);
        if (ro(rs)) return;
        DWORD attr = GetFileAttributesW(ha.c_str());
        if (attr == INVALID_FILE_ATTRIBUTES || (bool(attr & FILE_ATTRIBUTE_DIRECTORY) != dir))
            return fail(rs, ResultPathNotFound, op, a);
        if (exists(hb)) return fail(rs, ResultPathAlreadyExists, op, b);
        if (!parent_is_dir(hb)) return fail(rs, ResultPathNotFound, op, b);
        if (!MoveFileExW(ha.c_str(), hb.c_str(), 0)) rs.result = ResultTargetLocked;
        log_op(op, a + " -> " + b, rs.result);
    }
    s64 free_space() {
        ULARGE_INTEGER avail{};
        if (GetDiskFreeSpaceExW(fs_->root.c_str(), &avail, nullptr, nullptr))
            return (s64)std::min<u64>(avail.QuadPart, 0x100000000ull);  // report at most 4 GiB
        return 0x100000000ll;
    }
    s64 total_space() { return 0x100000000ll; }

    std::shared_ptr<FsRoot> fs_;
};

}  // namespace

ServicePtr make_host_filesystem(const std::wstring& root, bool read_only, const std::string& label) {
    auto fs = std::make_shared<FsRoot>();
    fs->root = root;
    while (!fs->root.empty() && (fs->root.back() == L'\\' || fs->root.back() == L'/')) fs->root.pop_back();
    fs->read_only = read_only;
    fs->label = label;
    if (!read_only) create_dirs(fs->root);
    return std::make_shared<IFileSystem>(fs);
}

}  // namespace hwfs
