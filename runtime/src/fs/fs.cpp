// fsp-srv: the filesystem service (SDK 3.5.1 command set).
//  - RomFS: OpenDataStorageByCurrentProcess serves a virtual RomFS image (romfs.cpp).
//  - Save data: host directories <exe dir>\save\<title id>\<user id | device | ...>\ (hostfs.cpp).
//  - SD card: <exe dir>\sdmc\.
#include <windows.h>

#include <cstdio>
#include <set>

#include "fs_internal.h"

namespace hwfs {

std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}

std::string wide_to_utf8(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

const std::wstring& exe_dir() {
    static std::wstring dir = [] {
        std::wstring p(32768, L'\0');
        DWORD n = GetModuleFileNameW(nullptr, p.data(), (DWORD)p.size());
        p.resize(n);
        p = p.substr(0, p.find_last_of(L"\\/"));
        if (p.rfind(L"\\\\", 0) != 0) p = L"\\\\?\\" + p;
        return p;
    }();
    return dir;
}

bool trace_enabled() {
    static bool on = [] {
        const char* e = getenv("HWDER_FS_TRACE");
        return e && *e && *e != '0';
    }();
    return on;
}

void FsService::dispatch(Request& rq, Response& rs) {
    auto it = commands_.find(rq.command);
    if (it != commands_.end()) return it->second(rq, rs);
    static std::mutex m;
    static std::set<std::pair<std::string, u32>> seen;
    std::lock_guard<std::mutex> l(m);
    if (seen.insert({name_, rq.command}).second)
        hw_log("fs: %s command %u not implemented (returning success)", name_.c_str(), rq.command);
}

namespace {

constexpr u8 kSaveSystem = 0, kSaveAccount = 1, kSaveBcat = 2, kSaveDevice = 3, kSaveTemporary = 4, kSaveCache = 5;

struct SaveDataAttribute {
    u64 program_id;
    u64 user_id[2];
    u64 system_save_data_id;
    u8 type;
    u8 rank;
    u16 index;
    u32 pad;
    u64 reserved[3];
};
static_assert(sizeof(SaveDataAttribute) == 0x40);

std::wstring save_dir(const SaveDataAttribute& a) {
    char buf[96];
    if (a.type == kSaveSystem) {
        snprintf(buf, sizeof(buf), "\\save\\system\\%016llX", (unsigned long long)a.system_save_data_id);
    } else {
        u64 tid = a.program_id ? a.program_id : kern::g_process.title_id;
        char who[40];
        bool has_user = a.user_id[0] || a.user_id[1];
        switch (a.type) {
        case kSaveAccount:
            if (has_user) snprintf(who, sizeof(who), "%016llX%016llX", (unsigned long long)a.user_id[1],
                                   (unsigned long long)a.user_id[0]);
            else snprintf(who, sizeof(who), "device");
            break;
        case kSaveBcat: snprintf(who, sizeof(who), "bcat"); break;
        case kSaveTemporary: snprintf(who, sizeof(who), "temporary"); break;
        case kSaveCache: snprintf(who, sizeof(who), "cache_%u", a.index); break;
        default: snprintf(who, sizeof(who), "device"); break;
        }
        snprintf(buf, sizeof(buf), "\\save\\%016llX\\%s", (unsigned long long)tid, who);
    }
    return exe_dir() + utf8_to_wide(buf);
}

ServicePtr open_save(const SaveDataAttribute& a, bool read_only, const char* how) {
    std::wstring dir = save_dir(a);
    std::string label = wide_to_utf8(dir.substr(exe_dir().size() + 1));
    hw_log("fs: %s type %u title %016llx user %016llx%016llx -> %s", how, a.type, (unsigned long long)a.program_id,
           (unsigned long long)a.user_id[1], (unsigned long long)a.user_id[0], label.c_str());
    // Save data always "exists": the directory is created on first open.
    return make_host_filesystem(dir, read_only, label);
}

class ISaveDataInfoReader : public FsService {
public:
    ISaveDataInfoReader() : FsService("ISaveDataInfoReader") {
        reg(0, [](Request&, Response& rs) { rs.push<s64>(0); });  // ReadSaveDataInfo: no entries
    }
};

class IEventNotifier : public FsService {
public:
    IEventNotifier() : FsService("IEventNotifier") {
        reg(0, [](Request&, Response& rs) { ipc::push_handle(rs, ipc::make_event()); });  // GetEventHandle
    }
};

class IDeviceOperator : public FsService {
public:
    IDeviceOperator() : FsService("IDeviceOperator") {
        reg(0, [](Request&, Response& rs) { rs.push<u8>(1); });      // IsSdCardInserted
        reg(1, [](Request&, Response& rs) { rs.push<u32>(1); });     // GetSdCardSpeedMode
        reg(4, [](Request&, Response& rs) { rs.push<s64>(32ll << 30); });  // GetSdCardUserAreaSize
        reg(5, [](Request&, Response& rs) { rs.push<s64>(32ll << 30); });  // GetSdCardProtectedAreaSize
        reg(200, [](Request&, Response& rs) { rs.push<u8>(1); });    // IsGameCardInserted
        reg(202, [](Request&, Response& rs) { rs.push<u32>(1); });   // GetGameCardHandle
    }
};

class FspSrv : public FsService {
public:
    FspSrv() : FsService("fsp-srv") {
        reg(1, [](Request&, Response&) {});  // SetCurrentProcess(pid)
        reg(2, [](Request&, Response& rs) {  // OpenDataFileSystemByCurrentProcess (old API): read-only romfs view
            hw_log("fs: OpenDataFileSystemByCurrentProcess");
            rs.push_object(make_host_filesystem(romfs_host_dir(), true, "rom"));
        });
        reg(18, [](Request&, Response& rs) {  // OpenSdCardFileSystem
            hw_log("fs: OpenSdCardFileSystem");
            rs.push_object(make_host_filesystem(exe_dir() + L"\\sdmc", false, "sdmc"));
        });
        reg(21, [](Request& rq, Response&) { hw_log("fs: DeleteSaveDataFileSystem(0x%llx)", (unsigned long long)rq.pop<u64>()); });
        reg(22, [](Request& rq, Response&) {  // CreateSaveDataFileSystem(attr, creation info, meta info)
            auto a = rq.pop<SaveDataAttribute>();
            open_save(a, false, "CreateSaveDataFileSystem");
        });
        reg(23, [](Request& rq, Response&) {  // CreateSaveDataFileSystemBySystemSaveDataId
            auto a = rq.pop<SaveDataAttribute>();
            open_save(a, false, "CreateSaveDataFileSystemBySystemSaveDataId");
        });
        reg(24, [](Request&, Response&) {});  // RegisterSaveDataFileSystemAtomicDeletion
        reg(25, [](Request&, Response&) {});  // DeleteSaveDataFileSystemBySaveDataSpaceId
        reg(26, [](Request&, Response&) {});  // FormatSdCardDryRun
        reg(27, [](Request&, Response& rs) { rs.push<u8>(1); });  // IsExFatSupported
        reg(51, [](Request& rq, Response& rs) {  // OpenSaveDataFileSystem(u8 space, attr)
            rq.pop<u8>();
            auto a = rq.pop<SaveDataAttribute>();
            rs.push_object(open_save(a, false, "OpenSaveDataFileSystem"));
        });
        reg(52, [](Request& rq, Response& rs) {  // OpenSaveDataFileSystemBySystemSaveDataId(u8 space, attr)
            rq.pop<u8>();
            auto a = rq.pop<SaveDataAttribute>();
            a.type = kSaveSystem;
            rs.push_object(open_save(a, false, "OpenSaveDataFileSystemBySystemSaveDataId"));
        });
        reg(53, [](Request& rq, Response& rs) {  // OpenReadOnlySaveDataFileSystem(u8 space, attr)
            rq.pop<u8>();
            auto a = rq.pop<SaveDataAttribute>();
            rs.push_object(open_save(a, true, "OpenReadOnlySaveDataFileSystem"));
        });
        reg(60, [](Request&, Response& rs) { rs.push_object(std::make_shared<ISaveDataInfoReader>()); });
        reg(61, [](Request&, Response& rs) { rs.push_object(std::make_shared<ISaveDataInfoReader>()); });
        reg(200, [](Request&, Response& rs) {  // OpenDataStorageByCurrentProcess
            hw_log("fs: OpenDataStorageByCurrentProcess");
            rs.push_object(open_romfs_storage());
        });
        reg(202, [](Request& rq, Response& rs) {  // OpenDataStorageByDataId(u8 storage, u64 data id)
            u8 storage = rq.pop<u8>();
            u64 id = rq.pop<u64>();
            hw_log("fs: OpenDataStorageByDataId(storage %u, data id %016llx) -> not found", storage,
                   (unsigned long long)id);
            rs.result = ResultTargetNotFound;
        });
        reg(203, [](Request&, Response& rs) {  // OpenPatchDataStorageByCurrentProcess: no update installed
            hw_log("fs: OpenPatchDataStorageByCurrentProcess -> no patch");
            rs.result = ResultTargetNotFound;
        });
        reg(400, [](Request&, Response& rs) { rs.push_object(std::make_shared<IDeviceOperator>()); });
        reg(500, [](Request&, Response& rs) { rs.push_object(std::make_shared<IEventNotifier>()); });
        reg(501, [](Request&, Response& rs) { rs.push_object(std::make_shared<IEventNotifier>()); });
        reg(601, [](Request&, Response& rs) { rs.push<s64>(0); });  // QuerySaveDataTotalSize
        reg(620, [](Request&, Response&) {});                       // SetSdCardEncryptionSeed
        reg(1003, [](Request&, Response&) {});                      // DisableAutoSaveDataCreation
        reg(1004, [](Request& rq, Response&) { rq.pop<u32>(); });   // SetGlobalAccessLogMode
        reg(1005, [](Request&, Response& rs) { rs.push<u32>(0); }); // GetGlobalAccessLogMode
        reg(1006, [](Request& rq, Response&) {                      // OutputAccessLogToSdCard(X string)
            if (trace_enabled()) hw_log("fs: access log: %s", rq.in_string().c_str());
        });
    }
};

}  // namespace

}  // namespace hwfs

namespace ipc {

void register_fs_services() { register_as<hwfs::FspSrv>({"fsp-srv"}); }

}  // namespace ipc
