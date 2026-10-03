// Shared declarations of the Vulkan renderer (runtime/src/gpu/vk/*).
#pragma once
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "gpu/display.h"
#include "gpu/maxwell_3d.h"
#include "gpu/renderer.h"

namespace gpu::vk {

#define VK_LOG_ONCE(...)              \
    do {                              \
        static bool once_ = false;    \
        if (!once_) {                 \
            once_ = true;             \
            hw_log(__VA_ARGS__);      \
        }                             \
    } while (0)

#define VK_CHECK(x)                                                                   \
    do {                                                                              \
        VkResult r_ = (x);                                                            \
        if (r_ != VK_SUCCESS) {                                                       \
            hw_log("vk: error %d at %s:%d: %s", (int)r_, __FILE__, __LINE__, #x);     \
            hw_fatal("vulkan");                                                       \
        }                                                                             \
    } while (0)

u64 readable_size(u64 addr, u64 size);  // committed host bytes starting at addr
inline VkDevice dev() { return display::device().dev; }
u32 find_memory_type(u32 bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid = 0);

inline u64 hash_bytes(const void* data, u64 size, u64 seed = 0x9E3779B97F4A7C15ull) {
    const u8* p = (const u8*)data;
    u64 h = seed ^ (size * 0xff51afd7ed558ccdull);
    u64 a = 0x2545F4914F6CDD1Dull, b = 0x9E3779B97F4A7C15ull, c = 0xC2B2AE3D27D4EB4Full, d = 0x165667B19E3779F9ull;
    u64 i = 0;
    for (; i + 32 <= size; i += 32) {
        u64 w[4];
        memcpy(w, p + i, 32);
        a = (a ^ w[0]) * 0x9E3779B97F4A7C15ull; a ^= a >> 29;
        b = (b ^ w[1]) * 0xC2B2AE3D27D4EB4Full; b ^= b >> 31;
        c = (c ^ w[2]) * 0x165667B19E3779F9ull; c ^= c >> 27;
        d = (d ^ w[3]) * 0xff51afd7ed558ccdull; d ^= d >> 33;
    }
    h ^= a + (b << 1) + (c << 2) + (d << 3);
    for (; i < size; i++) h = (h ^ p[i]) * 0x100000001B3ull;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return h;
}

// ------------------------------------------------------------------ formats
struct FormatInfo {
    VkFormat vk = VK_FORMAT_UNDEFINED;  // host format of the VkImage
    u8 bpb = 4;                         // guest bytes per block
    u8 bw = 1, bh = 1;                  // guest block dimensions
    bool depth = false, stencil = false;
    bool astc = false;                  // guest ASTC, host RGBA8 (CPU decoded)
    bool compressed = false;            // host format is block compressed
    bool integer = false;
    VkComponentMapping swizzle{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                               VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    bool valid() const { return vk != VK_FORMAT_UNDEFINED; }
};
FormatInfo tic_format(const u32* tic);  // TIC words 0..7
FormatInfo rt_format(u32 rt_format);    // Maxwell RenderTargetFormat
FormatInfo zeta_format(u32 depth_format);
FormatInfo android_format(u32 pixel_format);
VkFormat vertex_format(u32 attrib, u32* size_bytes);
u32 vk_format_bytes(VkFormat f);  // host bytes per texel (uncompressed) or per block
bool vk_format_compressed(VkFormat f);
VkFormatFeatureFlags format_features(VkFormat f);  // optimal tiling, cached
void decode_astc(const u8* src, u32 width, u32 height, u32 depth, u32 bw, u32 bh, u8* dst_rgba8);

// ------------------------------------------------------------------ host buffers
struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    u8* map = nullptr;
    u64 size = 0;
};
Buffer create_buffer(u64 size, VkBufferUsageFlags usage, bool host_visible, bool prefer_device_local = false);
void destroy_buffer(Buffer& b);

// A slice of CPU-writable memory valid for the commands being recorded.
struct Slice {
    VkBuffer buf = VK_NULL_HANDLE;
    u64 offset = 0;
    u8* ptr = nullptr;
};

// Draw-path profile (GPU thread): accumulated ms per phase since the last flush; logged when large.
enum ProfPhase {
    kProfProgram, kProfFramebuffer, kProfVertex, kProfIndex, kProfDescriptors, kProfPipeline, kProfRecord,
    kProfCbuf, kProfTexGet, kProfTexView, kProfSampler, kProfDescAlloc, kProfDescUpdate, kProfFbGet, kProfFbView,
    kProfReadable, kProfCreateImage, kProfAllocBind, kProfUpload, kProfStage, kProfDecode,
    kProfGetLayout, kProfGetOverlaps, kProfGetMatch, kProfGetValidate, kProfGetMiss, kProfClear, kProfInvalidate, kProfPresent,
    kProfKeyBuild, kProfDrawTotal, kProfCount
};
static const char* const kProfNames[kProfCount] = {
    "program", "framebuffer", "vertex", "index", "descriptors", "pipeline", "record",
    "|cbuf", "tex-get", "tex-view", "sampler", "desc-alloc", "desc-update", "fb-get", "fb-view",
    "|readable", "create-image", "alloc-bind", "upload", "stage", "decode",
    "|get-layout", "get-overlaps", "get-match", "get-validate", "get-miss", "|clear", "invalidate", "present", "keybuild", "|draw-total"};
extern double g_prof[kProfCount];
extern float g_res_scale, g_res_scale_y;  // internal resolution scale (x, y) for large render targets
extern u32 g_prof_draws;
struct ProfScope {
    ProfPhase ph;
    LARGE_INTEGER a;
    explicit ProfScope(ProfPhase p) : ph(p) { QueryPerformanceCounter(&a); }
    ~ProfScope() {
        LARGE_INTEGER b, f;
        QueryPerformanceCounter(&b);
        QueryPerformanceFrequency(&f);
        g_prof[ph] += (b.QuadPart - a.QuadPart) * 1000.0 / f.QuadPart;
    }
};

// Logs scopes that exceed `limit_ms` (stall diagnostics).
struct SlowTimer {
    const char* what;
    double limit_ms;
    LARGE_INTEGER a;
    SlowTimer(const char* w, double limit) : what(w), limit_ms(limit) { QueryPerformanceCounter(&a); }
    ~SlowTimer() {
        LARGE_INTEGER b, f;
        QueryPerformanceCounter(&b);
        QueryPerformanceFrequency(&f);
        double ms = (b.QuadPart - a.QuadPart) * 1000.0 / f.QuadPart;
        if (ms > limit_ms) hw_log("vk: slow: %s took %.0f ms", what, ms);
    }
};

// ------------------------------------------------------------------ command recording
class Recorder {
public:
    VkCommandBuffer cmd();          // current command buffer (outside any rendering instance)
    VkCommandBuffer cmd_in_pass();  // current command buffer, whatever the rendering state
    void end_rendering();
    void barrier();                 // full memory barrier (also ends rendering)
    Slice stage(u64 size, u64 align = 256);
    u64 serial() const { return serial_; }
    u64 completed() const;
    void defer(std::function<void()> fn);  // run after the current serial completes (on the waiter thread)
    void flush();
    void wait_idle();
    bool rendering = false;

    // set by the renderer
    VkCommandBuffer cb_ = VK_NULL_HANDLE;
    u64 serial_ = 1;
};
Recorder& rec();
// Destroy after the GPU finished the current work and a few frames were presented.
void retire_resource(std::function<void()> fn);

// ------------------------------------------------------------------ textures
enum class ImageType : u8 { e1D, e2D, e3D };

struct ImageInfo {
    ImageType type = ImageType::e2D;
    FormatInfo fmt;
    u32 width = 1, height = 1, depth = 1;  // depth: 3D only
    u32 layers = 1, levels = 1;
    bool block_linear = true;
    u32 bh = 0, bd = 0, tws = 0;  // block height/depth log2, tile width spacing
    u32 pitch = 0;                // pitch linear only
    u32 layer_stride = 0;         // guest bytes between layers (0 = computed)
    bool cube = false;
    bool render_target = false;
};

struct ViewKey {
    VkFormat format;
    VkImageViewType type;
    u32 base_level, levels, base_layer, layers;
    u32 swizzle;  // packed VkComponentSwizzle x4
    u32 aspect;
    bool operator==(const ViewKey& o) const { return !memcmp(this, &o, sizeof(*this)); }
};
struct ViewKeyHash {
    size_t operator()(const ViewKey& k) const { return (size_t)hash_bytes(&k, sizeof(k)); }
};

struct Image {
    ImageInfo info;
    u64 addr = 0, guest_size = 0;
    u64 readable = 0;  // committed guest bytes from addr (<= guest_size)
    u64 level_offset[16] = {};
    u64 layer_stride = 0;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    std::unordered_map<ViewKey, VkImageView, ViewKeyHash> views;
    bool gpu_modified = false;   // GPU data newer than guest memory
    u64 write_seq = 0;           // order of the last GPU write (aliasing images are refreshed from the newest)
    u64 presented_seq = ~0ull;   // write_seq when this image was last presented (diagnostic: late writes)
    bool cpu_dirty = false;      // guest memory written (invalidate) since the last upload
    u64 hash = 0;
    u64 checked_epoch = 0;
    u64 last_used = 0;
    // Internal resolution scale: the VkImage of a scaled render target is scale x the guest size
    // (info keeps the guest dimensions for cache matching and layout). level_* are PHYSICAL sizes.
    float scale = 1.0f, scale_y = 1.0f;  // x / y factors (differ for "unlocked" ultrawide rendering)
    u32 phys_w() const { return scale == 1.0f ? info.width : std::max(1u, (u32)(info.width * scale + 0.5f)); }
    u32 phys_h() const { return scale_y == 1.0f ? info.height : std::max(1u, (u32)(info.height * scale_y + 0.5f)); }
    u32 level_width(u32 l) const { return std::max(1u, phys_w() >> l); }
    u32 level_height(u32 l) const { return std::max(1u, phys_h() >> l); }
    u32 level_depth(u32 l) const { return std::max(1u, info.depth >> l); }
};

struct Subresource {
    u32 level = 0, layer = 0;
};

class TextureCache {
public:
    explicit TextureCache(MemoryManager* mm) : mm_(mm) {}
    // Find or create an image for `info` at guest CPU address `addr`. sub receives the matching
    // mip level / layer when the request is a subresource of a larger image.
    Image* get(u64 addr, const ImageInfo& info, Subresource* sub);
    Image* find_exact(u64 addr);  // any image starting at addr (largest), no creation
    Image* find_newest(u64 addr);  // the image at addr with the most recent GPU write / CPU upload
    void validate_now(Image* img);  // like validate() but hashes regardless of interval and budget
    void apply_scale(float sx, float sy);  // change the internal resolution scale: scaled targets are recreated on next use
    static bool scalable(const ImageInfo& info);
    Image* make_shadow(Image* src);  // unlinked twin of src (same size/format/scale) for frame-interpolation replays
    void destroy(Image* img, bool count = true);
    void alloc_image(Image* img);    // VkImage + memory + initial layout/clear for img->info/scale
    void validate(Image* img);              // re-upload if the CPU wrote it (cheap when nothing changed)
    bool has_same_addr_alias(Image* img);   // another image starts at the same address (padded RT vs sampled)
    VkImageView view(Image* img, VkFormat fmt, VkImageViewType type, u32 base_level, u32 levels, u32 base_layer,
                     u32 layers, VkComponentMapping swz = {}, VkImageAspectFlags aspect = 0);
    VkSampler sampler(const u32* tsc);
    void mark_modified(Image* img, bool clear = false) {
        // Diagnostic: a draw into a buffer that was presented and not cleared since means the present
        // happened before the frame's rendering was complete (fence reached too early).
        if (!clear && img->presented_seq == img->write_seq) late_writes_++;
        if (clear) img->presented_seq = ~0ull;
        img->gpu_modified = true;
        img->write_seq = ++write_counter_;
    }
    u64 write_counter_ = 0;
    u64 late_writes_ = 0;  // draws into an already-presented, not yet cleared buffer (see mark_modified)
    void invalidate(u64 addr, u64 size);
    void flush(u64 addr, u64 size);
    bool is_gpu_modified(u64 addr, u64 size);
    void next_epoch() { epoch_++; }
    u32 n_uploads = 0, n_downloads = 0, n_creates = 0;  // per-frame statistics (reset by the renderer)
    u64 generation = 0;  // bumped whenever an image is created or destroyed (framebuffer cache validity)
    u64 destroy_count = 0;  // images destroyed (a recorded frame referencing one must not be replayed)
    u64 tex_gen = 0;  // bumps when a cached TIC->image mapping may change (image destroyed / alias created)
    u64 upload_bytes = 0;
    u64 hash_budget_epoch_ = 0, hash_budget_ = 0;
    void download(Image* img);  // write GPU data back to guest memory (waits for the GPU)
    void collect_garbage();
    // helpers for image info from TIC
    bool info_from_tic(const u32* tic, ImageInfo& info, u64& addr, u32& base_level, u32& levels, u32& base_layer);

    static u64 compute_layout(const ImageInfo& info, u64* level_offset, u64* layer_stride);

private:
    void upload(Image* img);
    Image* create(u64 addr, const ImageInfo& info);
    void collect_overlaps(u64 addr, u64 size, std::vector<Image*>& out);
    bool match_subresource(Image* img, u64 addr, const ImageInfo& info, Subresource* sub);
    void link(Image* img);
    void unlink(Image* img);

    MemoryManager* mm_;
    std::unordered_map<u64, std::vector<Image*>> pages_;  // 64 KiB pages -> images
    std::unordered_map<u64, VkSampler> samplers_;
    u64 epoch_ = 1;
};

}  // namespace gpu::vk
