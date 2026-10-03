// Contract between the GPU core (command processing, engines; runtime/src/gpu/*) and the Vulkan
// renderer (runtime/src/gpu/vk/*). The core owns the engine register files and calls the renderer
// whenever an engine method has an effect that needs the host GPU.
//
// Register files are raw u32 arrays indexed by method number (method address / 4), exactly as the
// guest writes them; named offsets live in maxwell_3d.h etc. (owned by the GPU core).
#pragma once
#include <functional>

#include "hwder/runtime.h"

namespace gpu {

// GPU virtual address space (nvhost-as-gpu). The guest is identity mapped, so a GPU VA translates
// to a host pointer that is also the guest CPU address.
class MemoryManager {
public:
    virtual ~MemoryManager() = default;
    virtual u8* translate(u64 gpu_va) const = 0;  // nullptr if unmapped
    // Largest contiguous mapped span starting at gpu_va (<= size).
    virtual u64 contiguous_size(u64 gpu_va, u64 size) const = 0;
    virtual void read(u64 gpu_va, void* dst, u64 size) const = 0;
    virtual void write(u64 gpu_va, const void* src, u64 size) = 0;
};

struct DrawParams {
    bool indexed = false;
    u32 topology = 0;           // Maxwell PrimitiveTopology value
    u32 first = 0, count = 0;   // vertices or indices
    u32 base_vertex = 0, base_instance = 0;
    u32 instance_count = 1;
    u32 instance = 0;           // instance index of this draw (base_instance includes it)
};

// Display surface handed to the compositor by vi / nvnflinger.
struct FramebufferInfo {
    u64 address = 0;  // guest CPU address (nvmap handle address + offset)
    u32 width = 0, height = 0, stride = 0;
    u32 format = 0;          // android pixel format (1 = RGBA8888, 4 = RGB565, ...)
    u32 block_height_log2 = 4;
    bool block_linear = true;
    u32 transform_flags = 0;
    // crop rectangle (0,0,0,0 = whole surface)
    s32 crop_left = 0, crop_top = 0, crop_right = 0, crop_bottom = 0;
    u32 swap_interval = 1;
};

class Renderer {
public:
    virtual ~Renderer() = default;

    // Maxwell 3D (class 0xB197). regs: 0xE00 u32 register file.
    virtual void draw(const u32* regs, const DrawParams& p) = 0;
    virtual void clear(const u32* regs, u32 clear_buffers_value) = 0;  // value written to CLEAR_SURFACE
    // Inline index data (INLINE_INDEX_* methods) are delivered as a buffer for the next draw.
    virtual void draw_inline_indices(const u32* regs, const DrawParams& p, const void* indices, u32 index_size) = 0;
    // CB_BIND (BindGroups[stage].cbuf): bind the ConstBuffer selector range to c[slot] of a graphics stage
    // (stage 0 VertexA/B, 1 TessControl, 2 TessEval, 3 Geometry, 4 Fragment).
    virtual void bind_const_buffer(u32 stage, u32 slot, u64 gpu_va, u32 size, bool valid) = 0;
    // Maxwell compute (class 0xB1C0): launch descriptor (QMD) at qmd_gpu_va.
    virtual void dispatch(const u32* compute_regs, u64 qmd_gpu_va) = 0;
    // Fermi 2D (class 0x902D) surface copy; regs = 2D register file.
    virtual bool blit_2d(const u32* regs2d) = 0;  // false: core falls back to a CPU copy

    // Memory coherency: guest CPU / DMA wrote this range, drop or re-upload cached host copies.
    virtual void invalidate_region(u64 cpu_addr, u64 size) = 0;
    // Guest CPU / DMA is about to read this range: write back host-rendered data.
    virtual void flush_region(u64 cpu_addr, u64 size) = 0;
    // Whether the renderer holds GPU-modified data for this range (lets DMA copy GPU-side).
    virtual bool is_gpu_modified(u64 cpu_addr, u64 size) = 0;

    // Semaphore / syncpoint ordering: run `fn` once all previously submitted host GPU work completes.
    virtual void on_gpu_idle(std::function<void()> fn) = 0;
    virtual void flush_commands() = 0;  // submit recorded work (end of GPFIFO entry batch)
    virtual void wait_idle() = 0;

    // Composite a display surface and present it (called by vi on QueueBuffer).
    virtual void present(const FramebufferInfo& fb) = 0;
};

// Set up by the renderer (gpu/vk); the core uses it. The core passes its MemoryManager.
Renderer* create_renderer(MemoryManager* mm);

}  // namespace gpu
