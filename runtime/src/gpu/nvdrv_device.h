// nvdrv device model. The nvdrv service (owned by the GPU core, gpu/nvdrv*.cpp) routes Open/Ioctl/
// Close to Device objects registered by path. Other subsystems (video: nvhost-nvdec, nvhost-vic)
// register their own devices here.
#pragma once
#include <functional>
#include <memory>
#include <string>

#include "kernel/kernel.h"

namespace nv {

// NvResult values returned to the guest (u32 after the ioctl).
enum NvResult : u32 {
    Success = 0,
    NotImplemented = 1,
    NotSupported = 2,
    NotInitialized = 3,
    BadParameter = 4,
    Timeout = 5,
    InsufficientMemory = 6,
    ReadOnlyAttribute = 7,
    InvalidState = 8,
    InvalidAddress = 9,
    InvalidSize = 0xA,
    BadValue = 0xB,
    AlreadyAllocated = 0xD,
    Busy = 0xE,
    ResourceError = 0xF,
    CountMismatch = 0x10,
    TryAgain = 0x14,
};

struct IoctlBuffers {
    const u8* in = nullptr;  u64 in_size = 0;    // primary input (A/X buffer)
    u8* out = nullptr;       u64 out_size = 0;   // primary output (B/C buffer) - may alias `in` data layout
    const u8* in2 = nullptr; u64 in2_size = 0;   // Ioctl2 extra input
    u8* out2 = nullptr;      u64 out2_size = 0;  // Ioctl3 extra output
};

class Device {
public:
    virtual ~Device() = default;
    // cmd: the full ioctl number (dir/size/type/nr as on Linux). For in/out ioctls the service copies
    // `in` into `out` first, so implementations may read and write `out` in place.
    virtual NvResult ioctl(u32 fd, u32 cmd, IoctlBuffers& b) = 0;
    virtual void on_open(u32 fd) {}
    virtual void on_close(u32 fd) {}
    // QueryEvent(fd, event_id): returns nullptr if the device has no such event.
    virtual std::shared_ptr<kern::ReadableEvent> query_event(u32 fd, u32 event_id) { return nullptr; }
};

// One instance per Open() of the path.
void register_device(const std::string& path, std::function<std::shared_ptr<Device>()> factory);

// ---- shared state provided by the GPU core ----
// nvmap: guest address and size of a handle (0 if unknown); handle ids and ids are both accepted.
u64 nvmap_address(u32 handle);
u64 nvmap_size(u32 handle);

// host1x syncpoints (shared by GPU channels, nvdec, vic).
u32 syncpoint_allocate();
u32 syncpoint_read(u32 id);
u32 syncpoint_increment(u32 id);            // returns new value, wakes waiters/events
u32 syncpoint_reserve(u32 id, u32 count);   // bump the "max" (expected) value; returns new max
bool syncpoint_wait(u32 id, u32 threshold, s64 timeout_ns);  // blocks the calling guest thread

}  // namespace nv
