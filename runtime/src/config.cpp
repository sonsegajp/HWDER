// Runtime configuration (paths to the user's game data).
#include <windows.h>

#include <cstdlib>
#include <string>

#include "hwder/runtime.h"

static std::string g_romfs;

void hw_config_load(const std::string& dir) {
    // UTF-8 like `dir`; HWDER_ROMFS is read as wide so non-ANSI paths survive.
    const wchar_t* env = _wgetenv(L"HWDER_ROMFS");
    if (env) {
        int n = WideCharToMultiByte(CP_UTF8, 0, env, -1, nullptr, 0, nullptr, nullptr);
        g_romfs.assign(n > 0 ? n - 1 : 0, '\0');
        WideCharToMultiByte(CP_UTF8, 0, env, -1, g_romfs.data(), n, nullptr, nullptr);
    } else {
        g_romfs = dir + "\\romfs";
    }
    hw_log("romfs: %s", g_romfs.c_str());
}

const std::string& hw_romfs_path() { return g_romfs; }
