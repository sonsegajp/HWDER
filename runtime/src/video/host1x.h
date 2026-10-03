// host1x command stream processing for the multimedia engines (NVDEC, VIC, NVJPG).
#pragma once
#include "hwder/runtime.h"

namespace host1x {

// Per-channel engine state: THI registers and the engine's method registers (u64 slots, indexed
// by method number as written to THI METHOD0).
struct ChannelState {
    u32 cur_class = 0;
    u32 thi[0x20] = {};
    u64 regs[0x400] = {};
    u64 executes = 0;
};

// Run `words` command words for a channel whose engine class is `engine_class`.
void process(u32 engine_class, const u32* words, u32 count, ChannelState* st);

}  // namespace host1x

namespace video {
// Map a pinned 32-bit device address back to guest memory.
bool iova_to_guest(u32 iova, u64* addr, u64* size);
// Engine executes (decoder.cpp / vic.cpp)
void nvdec_execute(host1x::ChannelState& st);
void vic_execute(host1x::ChannelState& st);
}  // namespace video
