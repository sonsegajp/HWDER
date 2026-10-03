// Maxwell instruction decoding.
#pragma once
#include "hwder/runtime.h"

namespace shader {

enum class Op : u16 {
#define X(n, p) n,
#include "shader/opcodes.inc"
#undef X
    INVALID
};

Op decode(u64 inst);
const char* op_name(Op op);

// Bit-field helpers on a 64-bit instruction word.
struct Inst {
    u64 raw;
    u32 bits(int lo, int n) const { return u32((raw >> lo) & ((1ull << n) - 1)); }
    bool bit(int b) const { return (raw >> b) & 1; }
    s32 sbits(int lo, int n) const { return s32(u32(bits(lo, n)) << (32 - n)) >> (32 - n); }
};

}  // namespace shader
