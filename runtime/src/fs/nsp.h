// In-process NSP loading: the game's ExeFS modules and RomFS are read straight out of the user's
// own .nsp dump (decrypted on the fly with their prod.keys), so a distribution needs no extraction
// step. Directory mode (extracted exefs/ + romfs/) keeps working when no NSP is present or when
// HWDER_EXEFS / HWDER_ROMFS are set.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace hwfs::nsp {

// Looks for an NSP (HWDER_NSP, else *.nsp beside the exe) and keys (HWDER_KEYS, prod.keys beside
// the exe, ~/.switch, yuzu/Ryujinx key folders). Returns true when the game will be served from
// the NSP. On failure the reason is logged and `error` describes it for the user.
bool init(const std::string& exe_dir, std::string& error);
bool active();
const std::string& path();

// ExeFS file (main, rtld, sdk, subsdk0, main.npdm ...) fully decrypted. False if absent.
bool read_exefs(const char* name, std::vector<uint8_t>& out);
std::vector<std::string> exefs_files();

// Raw (level 3) RomFS image of the program NCA, as the SDK's IStorage sees it.
uint64_t romfs_size();
bool romfs_read(uint64_t off, uint8_t* dst, uint64_t size);

}  // namespace hwfs::nsp
