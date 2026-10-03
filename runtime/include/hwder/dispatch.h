// HWDER - interface between recompiled code and the runtime.
#pragma once
#include "hwder/cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*HwFn)(Ctx* c);

typedef struct HwFuncEntry {
    u64 addr;
    HwFn fn;
} HwFuncEntry;

extern const HwFuncEntry hw_func_table[];
extern const unsigned hw_func_count;

// Resolve a guest code address to its recompiled host function (NULL if none).
HwFn hw_lookup(u64 addr);
// Indirect call: BLR. Return address is already in x30.
void hw_call(Ctx* c, u64 addr);
// Indirect jump: BR / RET Xn. Target in c->pc.
void hw_dispatch(Ctx* c);

#ifdef __cplusplus
}
#endif
