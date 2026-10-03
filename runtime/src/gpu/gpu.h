// GPU core: channels, command processing (GPFIFO / pushbuffers) and engines.
#pragma once
#include <vector>

#include "renderer.h"

namespace gpu {

struct Fence {
    u32 id, value;
};

// One GPFIFO entry as submitted by the guest (u64: address bits 2-39, size in words bits 42-62).
struct GpEntry {
    u64 raw;
    u64 address() const { return raw & 0xFFFFFFFFFCull; }
    u32 words() const { return (u32)((raw >> 42) & 0x1FFFFF); }
    bool no_prefetch() const { return (raw >> 63) & 1; }
};

struct Submission {
    std::vector<GpEntry> entries;
    bool wait = false, increment = false;
    Fence wait_fence{};
    u32 syncpoint = 0;
    u32 increment_count = 0;
};

struct Channel;

// Address space shared by channels bound to it.
MemoryManager* create_address_space();

Channel* create_channel(MemoryManager* mm);
void channel_submit(Channel* ch, Submission&& s);  // queued; executed in order on the GPU thread

Renderer* renderer();  // created on first use
double gpu_busy_ms_reset();  // GPU thread time spent executing submissions since the last call

}  // namespace gpu
