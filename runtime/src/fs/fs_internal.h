// Internal interfaces of the fs subsystem (fsp-srv, RomFS image, host-backed filesystems).
#pragma once
#include <mutex>
#include <set>
#include <string>

#include "service/common.h"

namespace hwfs {

using ipc::Request;
using ipc::Response;
using ipc::Result;
using ipc::ServicePtr;

// nn::fs result codes (module 2).
constexpr Result FsResult(u32 desc) { return 2u | (desc << 9); }
constexpr Result ResultPathNotFound = FsResult(1);
constexpr Result ResultPathAlreadyExists = FsResult(2);
constexpr Result ResultTargetLocked = FsResult(7);
constexpr Result ResultDirectoryNotEmpty = FsResult(8);
constexpr Result ResultUsableSpaceNotEnough = FsResult(30);
constexpr Result ResultTargetNotFound = FsResult(1002);
constexpr Result ResultNotImplemented = FsResult(3001);
constexpr Result ResultOutOfRange = FsResult(3005);
constexpr Result ResultInvalidPath = FsResult(6001);
constexpr Result ResultInvalidOffset = FsResult(6061);
constexpr Result ResultInvalidSize = FsResult(6062);
constexpr Result ResultFileExtensionWithoutOpenModeAllowAppend = FsResult(6201);
constexpr Result ResultReadNotPermitted = FsResult(6202);
constexpr Result ResultWriteNotPermitted = FsResult(6203);
constexpr Result ResultUnsupportedOperation = FsResult(6300);
constexpr Result ResultPermissionDenied = FsResult(6400);

std::wstring utf8_to_wide(const std::string& s);
std::string wide_to_utf8(const std::wstring& s);
const std::wstring& exe_dir();  // directory containing hwder.exe (no trailing slash)
bool trace_enabled();           // HWDER_FS_TRACE set

// Base for fs interfaces: unknown commands are logged once and succeed with no output.
class FsService : public ipc::Service {
public:
    using ipc::Service::Service;
    void dispatch(Request& rq, Response& rs) override;
};

// RomFS: a virtual level-3 RomFS image synthesized from the extracted romfs directory.
ServicePtr open_romfs_storage();
const std::wstring& romfs_host_dir();

// IFileSystem backed by a host directory (save data, SD card, read-only romfs view).
ServicePtr make_host_filesystem(const std::wstring& root, bool read_only, const std::string& label);

}  // namespace hwfs
