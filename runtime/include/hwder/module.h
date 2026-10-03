// Layout of the recompiled modules, emitted by tools/recomp/gen_main.py (module_info.c).
#pragma once
#include "hwder/cpu.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct HwModuleInfo {
    const char* name;  // ExeFS file name (rtld, main, subsdk0, subsdk1, sdk)
    u64 base, image_size, text_lo, text_hi;
} HwModuleInfo;
extern const HwModuleInfo hw_modules[];
extern const unsigned hw_module_count;
#ifdef __cplusplus
}
#endif
