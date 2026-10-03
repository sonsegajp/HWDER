// Texture cache: guest surfaces (block-linear / pitch) <-> VkImages keyed by guest address.
#include "gpu/settings.h"
#include <algorithm>
#include <map>

#include <unordered_set>

#include "kernel/kernel.h"
#include "vk_common.h"

namespace gpu::vk {

double g_prof[kProfCount];
float g_res_scale = 1.0f, g_res_scale_y = 1.0f;
u32 g_prof_draws;


// ------------------------------------------------------------------ device memory sub-allocation
namespace {
struct Block {
    VkDeviceMemory mem = VK_NULL_HANDLE;
    u64 size = 0;
    std::map<u64, u64> free;  // offset -> size
};
struct Alloc {
    VkDeviceMemory mem = VK_NULL_HANDLE;
    u64 offset = 0, size = 0;
    Block* block = nullptr;
};
std::mutex g_alloc_lock;
std::vector<Block*> g_blocks[VK_MAX_MEMORY_TYPES];
std::unordered_map<VkDeviceMemory, Alloc> g_image_allocs;  // keyed by image handle cast
std::unordered_map<u64, Alloc> g_allocs;

constexpr u64 kBlockSize = 128ull << 20;

Alloc allocate(const VkMemoryRequirements& req, VkMemoryPropertyFlags flags) {
    std::lock_guard<std::mutex> l(g_alloc_lock);
    u32 type = find_memory_type(req.memoryTypeBits, flags);
    Alloc a;
    if (req.size > kBlockSize / 2) {
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        VK_CHECK(vkAllocateMemory(dev(), &ai, nullptr, &a.mem));
        a.size = req.size;
        return a;
    }
    for (int attempt = 0; attempt < 2; attempt++) {
        for (Block* b : g_blocks[type]) {
            for (auto it = b->free.begin(); it != b->free.end(); ++it) {
                u64 start = (it->first + req.alignment - 1) & ~(req.alignment - 1);
                u64 end = it->first + it->second;
                if (start + req.size > end) continue;
                u64 fo = it->first;
                b->free.erase(it);
                if (start > fo) b->free[fo] = start - fo;
                if (end > start + req.size) b->free[start + req.size] = end - (start + req.size);
                a.mem = b->mem;
                a.offset = start;
                a.size = req.size;
                a.block = b;
                return a;
            }
        }
        Block* b = new Block;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = kBlockSize;
        ai.memoryTypeIndex = type;
        VK_CHECK(vkAllocateMemory(dev(), &ai, nullptr, &b->mem));
        b->size = kBlockSize;
        b->free[0] = kBlockSize;
        g_blocks[type].push_back(b);
    }
    hw_fatal("vk: allocation failed");
    return a;
}

void release(const Alloc& a) {
    std::lock_guard<std::mutex> l(g_alloc_lock);
    if (!a.block) {
        vkFreeMemory(dev(), a.mem, nullptr);
        return;
    }
    auto& fl = a.block->free;
    u64 off = a.offset, size = a.size;
    auto next = fl.lower_bound(off);
    if (next != fl.end() && next->first == off + size) {
        size += next->second;
        next = fl.erase(next);
    }
    if (next != fl.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == off) {
            prev->second += size;
            return;
        }
    }
    fl[off] = size;
}

}  // namespace

// Committed bytes from addr (VirtualQuery walk; slow when memory is committed in small pieces).
static u64 walk_committed(u64 addr, u64 size) {
    u64 done = 0;
    while (done < size) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery((void*)(addr + done), &mbi, sizeof(mbi))) break;
        if (mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))) break;
        done = (u64)mbi.BaseAddress + mbi.RegionSize - addr;
    }
    return std::min(done, size);
}
u64 readable_size(u64 addr, u64 size) {
    // The kernel's region map (whole heaps per region) answers this in a few lookups; memory that is
    // not a kernel region (host-side allocations) falls back to the OS view.
    u64 end = addr + size, pos = addr;
    while (pos < end) {
        kern::mem::Region r = kern::mem::query(pos);
        if (r.state == kern::mem::Free || r.addr + r.size <= pos) return (pos - addr) + walk_committed(pos, end - pos);
        pos = r.addr + r.size;
    }
    return size;
}

namespace {
u32 mip_shift(u32 shift, u32 unit, u32 dim) {
    while (shift > 0 && (unit << (shift - 1)) >= dim) shift--;
    return shift;
}

struct LevelParams {
    u32 w, h, d;           // texels
    u32 bx, by;            // blocks
    u32 width_bytes;
    u32 bhs, bds;          // block-linear shifts
    u32 gobs_w = 0;
    u64 size;              // guest bytes
};

LevelParams level_params(const ImageInfo& info, u32 l) {
    LevelParams p;
    p.w = std::max(1u, info.width >> l);
    p.h = std::max(1u, info.height >> l);
    p.d = info.type == ImageType::e3D ? std::max(1u, info.depth >> l) : 1;
    p.bx = (p.w + info.fmt.bw - 1) / info.fmt.bw;
    p.by = (p.h + info.fmt.bh - 1) / info.fmt.bh;
    p.width_bytes = p.bx * info.fmt.bpb;
    if (!info.block_linear) {
        u32 pitch = info.pitch ? info.pitch : p.width_bytes;
        p.bhs = p.bds = 0;
        p.size = (u64)pitch * p.by * p.d;
        return p;
    }
    if (info.levels == 1 && l == 0) {
        p.bhs = info.bh;
        p.bds = info.type == ImageType::e3D ? info.bd : 0;
    } else {
        p.bhs = mip_shift(info.bh, 8, p.by);
        p.bds = info.type == ImageType::e3D ? mip_shift(info.bd, 1, p.d) : 0;
    }
    u32 gobs_w = (p.width_bytes + 63) / 64;
    if (info.tws) {
        u32 a = 1u << info.tws;
        if (gobs_w > a) gobs_w = (gobs_w + a - 1) & ~(a - 1);
    }
    p.gobs_w = gobs_w;
    u32 gobs_h = (p.by + 7) / 8;
    u32 bh = 1u << p.bhs, bd = 1u << p.bds;
    u64 blocks = (u64)gobs_w * ((gobs_h + bh - 1) / bh) * ((p.d + bd - 1) / bd);
    p.size = blocks * (512ull << (p.bhs + p.bds));
    return p;
}

// Block-linear <-> linear copy of one mip level (all slices). linear rows are width_bytes apart.
void bl_copy(const LevelParams& p, u8* guest, u8* linear, bool to_linear, u64 guest_size) {
    u32 gobs_w = p.gobs_w;
    u32 gobs_h = (p.by + 7) / 8;
    u32 bh = 1u << p.bhs, bd = 1u << p.bds;
    u32 blocks_h = (gobs_h + bh - 1) / bh;
    // row pitch in GOBs follows the level size computation (tile width spacing ignored here)
    u64 block_size = 512ull << (p.bhs + p.bds);
    u64 pitch = p.width_bytes;
    for (u32 z = 0; z < p.d; z++) {
        for (u32 gy = 0; gy < gobs_h; gy++) {
            for (u32 gx = 0; gx < gobs_w; gx++) {
                u64 base = (((u64)(z / bd) * blocks_h + gy / bh) * gobs_w + gx) * block_size +
                           ((u64)(z % bd) * bh + gy % bh) * 512;
                if (base + 512 > guest_size) continue;
                for (u32 y = 0; y < 8; y++) {
                    u32 row = gy * 8 + y;
                    if (row >= p.by) break;
                    u8* lrow = linear + ((u64)z * p.by + row) * pitch;
                    for (u32 c = 0; c < 4; c++) {
                        u32 x = gx * 64 + c * 16;
                        if (x >= p.width_bytes) break;
                        u32 n = std::min(16u, p.width_bytes - x);
                        u8* g = guest + base + (c >> 1) * 256 + (y >> 1) * 64 + (c & 1) * 32 + (y & 1) * 16;
                        if (to_linear) memcpy(lrow + x, g, n);
                        else memcpy(g, lrow + x, n);
                    }
                }
            }
        }
    }
}

VkImageAspectFlags aspect_of(const FormatInfo& f) {
    VkImageAspectFlags a = 0;
    if (f.depth) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (f.stencil) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return a ? a : VK_IMAGE_ASPECT_COLOR_BIT;
}

bool formats_compatible(const FormatInfo& a, const FormatInfo& b) {
    if (a.depth || a.stencil || b.depth || b.stencil) return (a.depth || a.stencil) && (b.depth || b.stencil);
    if (a.vk == b.vk) return true;
    if (a.astc != b.astc) return false;
    if (a.compressed || b.compressed) return false;
    return vk_format_bytes(a.vk) == vk_format_bytes(b.vk);
}
}  // namespace

u64 TextureCache::compute_layout(const ImageInfo& info, u64* level_offset, u64* layer_stride) {
    u64 off = 0;
    for (u32 l = 0; l < info.levels && l < 16; l++) {
        if (level_offset) level_offset[l] = off;
        off += level_params(info, l).size;
    }
    u64 stride = off;
    if (info.layers > 1) {
        if (info.layer_stride) {
            stride = info.layer_stride;
        } else if (info.block_linear) {
            u32 bh = info.bh, bd = info.bd;
            u32 rows = (info.height + info.fmt.bh - 1) / info.fmt.bh;
            if (info.tws) {
                u64 a = 512ull << (info.tws + bh + bd);
                stride = (stride + a - 1) & ~(a - 1);
            } else {
                while (bh && rows <= (8u << (bh - 1))) bh--;
                while (bd && info.depth <= (1u << (bd - 1))) bd--;
                u64 a = 512ull << (bh + bd);
                stride = (stride + a - 1) & ~(a - 1);
            }
        }
    }
    if (layer_stride) *layer_stride = stride;
    return stride * (info.layers - 1) + off;
}

void TextureCache::link(Image* img) {
    for (u64 p = img->addr >> 16; p <= (img->addr + img->guest_size - 1) >> 16; p++) pages_[p].push_back(img);
}
void TextureCache::unlink(Image* img) {
    for (u64 p = img->addr >> 16; p <= (img->addr + img->guest_size - 1) >> 16; p++) {
        auto it = pages_.find(p);
        if (it == pages_.end()) continue;
        auto& v = it->second;
        v.erase(std::remove(v.begin(), v.end(), img), v.end());
        if (v.empty()) pages_.erase(it);
    }
}

void TextureCache::collect_overlaps(u64 addr, u64 size, std::vector<Image*>& out) {
    out.clear();
    if (!size) size = 1;
    for (u64 p = addr >> 16; p <= (addr + size - 1) >> 16; p++) {
        auto it = pages_.find(p);
        if (it == pages_.end()) continue;
        for (Image* img : it->second) {
            if (img->addr < addr + size && addr < img->addr + img->guest_size &&
                std::find(out.begin(), out.end(), img) == out.end())
                out.push_back(img);
        }
    }
}

bool TextureCache::has_same_addr_alias(Image* img) {
    auto it = pages_.find(img->addr >> 16);
    if (it == pages_.end()) return false;
    for (Image* o : it->second)
        if (o != img && o->addr == img->addr) return true;
    return false;
}

// Level-0 transfer between two images of one guest surface: a copy when both have the same internal
// scale, a blit (resampling) when they differ.
static void transfer_image(Image* src, Image* dst, u32 layers) {
    rec().barrier();
    if (src->scale == dst->scale && src->scale_y == dst->scale_y) {
        VkImageCopy c{};
        c.srcSubresource = {src->aspect, 0, 0, layers};
        c.dstSubresource = {dst->aspect, 0, 0, layers};
        c.extent = {std::min(src->level_width(0), dst->level_width(0)), std::min(src->level_height(0), dst->level_height(0)),
                    std::min(src->info.depth, dst->info.depth)};
        vkCmdCopyImage(rec().cmd(), src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    } else {
        VkImageBlit b{};
        b.srcSubresource = {src->aspect, 0, 0, layers};
        b.dstSubresource = {dst->aspect, 0, 0, layers};
        b.srcOffsets[1] = {(s32)src->level_width(0), (s32)src->level_height(0), 1};
        b.dstOffsets[1] = {(s32)dst->level_width(0), (s32)dst->level_height(0), 1};
        bool depth = src->aspect != VK_IMAGE_ASPECT_COLOR_BIT;
        vkCmdBlitImage(rec().cmd(), src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                       depth ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    }
    rec().barrier();
}

Image* TextureCache::find_newest(u64 addr) {
    auto it = pages_.find(addr >> 16);
    if (it == pages_.end()) return nullptr;
    Image* best = nullptr;
    for (Image* img : it->second) {
        if (img->addr != addr) continue;
        if (!best || img->write_seq > best->write_seq || (img->write_seq == best->write_seq && img->last_used > best->last_used))
            best = img;
    }
    return best;
}

void TextureCache::validate_now(Image* img) {
    if (!img->cpu_dirty && !img->gpu_modified) {
        img->checked_epoch = 0;
        if (hash_budget_epoch_ != epoch_) {
            hash_budget_epoch_ = epoch_;
            hash_budget_ = 48u << 20;
        }
        hash_budget_ += img->readable;
    }
    validate(img);
}

Image* TextureCache::find_exact(u64 addr) {
    auto it = pages_.find(addr >> 16);
    if (it == pages_.end()) return nullptr;
    Image* best = nullptr;
    for (Image* img : it->second)
        if (img->addr == addr && (!best || img->last_used > best->last_used)) best = img;
    return best;
}

bool TextureCache::match_subresource(Image* img, u64 addr, const ImageInfo& info, Subresource* sub) {
    if (addr < img->addr || addr >= img->addr + img->guest_size) return false;
    if (!formats_compatible(img->info.fmt, info.fmt)) return false;
    u64 off = addr - img->addr;
    u32 layer = img->layer_stride ? (u32)(off / img->layer_stride) : 0;
    u64 rem = img->layer_stride ? off % img->layer_stride : off;
    if (layer >= img->info.layers) return false;
    for (u32 l = 0; l < img->info.levels; l++) {
        if (img->level_offset[l] != rem) continue;
        if (img->level_width(l) != info.width || img->level_height(l) != info.height) return false;
        if (info.type == ImageType::e3D && img->level_depth(l) != info.depth) return false;
        if (info.levels > img->info.levels - l) return false;
        if (info.layers > img->info.layers - layer) return false;
        sub->level = l;
        sub->layer = layer;
        return true;
    }
    return false;
}

Image* TextureCache::get(u64 addr, const ImageInfo& info, Subresource* sub) {
    SlowTimer timer("texture get", 30.0);
    *sub = {};
    ProfScope* ps = new ProfScope(kProfGetLayout);
    u64 size = compute_layout(info, nullptr, nullptr);
    delete ps;
    ps = new ProfScope(kProfGetOverlaps);
    std::vector<Image*> overlaps;
    collect_overlaps(addr, size, overlaps);
    delete ps;
    ps = new ProfScope(kProfGetMatch);
    Image* hit = nullptr;
    for (Image* img : overlaps) {
        if (img->addr == addr && img->info.type == info.type && img->info.width == info.width &&
            img->info.height == info.height && (info.type != ImageType::e3D || img->info.depth == info.depth) &&
            formats_compatible(img->info.fmt, info.fmt) && img->info.levels >= info.levels &&
            img->info.layers >= info.layers) {
            if (!hit || img->last_used > hit->last_used) hit = img;
        }
    }
    if (!hit) {
        for (Image* img : overlaps) {
            Subresource s;
            if (match_subresource(img, addr, info, &s)) {
                hit = img;
                *sub = s;
                break;
            }
        }
    }
    delete ps;
    if (hit) {
        ProfScope vs(kProfGetValidate);
        hit->last_used = rec().serial();
        // Render targets are allocated with widths padded to 64 bytes and then sampled at their real
        // size, so one address carries two images. Refresh this one from a newer GPU-written alias.
        Image* newest = nullptr;
        for (Image* o : overlaps) {
            if (o == hit || !o->gpu_modified || o->write_seq <= hit->write_seq) continue;
            if (o->addr != hit->addr || o->info.type != hit->info.type || o->aspect != hit->aspect) continue;
            if (o->info.fmt.compressed || hit->info.fmt.compressed) continue;
            if (vk_format_bytes(o->info.fmt.vk) != vk_format_bytes(hit->info.fmt.vk)) continue;
            if (!newest || o->write_seq > newest->write_seq) newest = o;
        }
        if (newest) {
            transfer_image(newest, hit, std::min(newest->info.layers, hit->info.layers));
            hit->gpu_modified = true;
            hit->write_seq = newest->write_seq;
            hit->cpu_dirty = false;
            return hit;
        }
        validate(hit);
        return hit;
    }
    ProfScope ms(kProfGetMiss);
    // Miss: write back GPU data that the new image does not take over by copy.
    if (!info.render_target) {
        Image probe;
        probe.info = info;
        probe.addr = addr;
        probe.guest_size = compute_layout(info, probe.level_offset, &probe.layer_stride);
        for (Image* o : overlaps) {
            Subresource s;
            // Same base address and texel size: the overlapping region is copied on the GPU below
            // (render targets are allocated with aligned widths and then sampled at their real size).
            bool same_base = o->addr == addr && !o->info.fmt.compressed && !info.fmt.compressed &&
                             o->info.type == info.type && o->aspect == aspect_of(info.fmt) &&
                             vk_format_bytes(o->info.fmt.vk) == vk_format_bytes(info.fmt.vk);
            bool copyable = match_subresource(&probe, o->addr, o->info, &s) || same_base;
            if (o->gpu_modified && !copyable) {
                static int logged = 0;
                if (logged++ < 20)
                    hw_log("vk: miss downloads %ux%u fmt %d levels %u layers %u (%s) for request %ux%u fmt %d levels %u layers %u at +0x%llx",
                           o->info.width, o->info.height, (int)o->info.fmt.vk, o->info.levels, o->info.layers,
                           o->info.render_target ? "rt" : "tex", info.width, info.height, (int)info.fmt.vk, info.levels, info.layers,
                           (unsigned long long)(addr - o->addr));
                download(o);
            }
        }
    }
    Image* img = create(addr, info);
    for (Image* o : overlaps) {
        if (o->gpu_modified) {
            Subresource s;
            bool done = false;
            if (match_subresource(img, o->addr, o->info, &s) && !o->info.fmt.compressed && !img->info.fmt.compressed &&
                o->scale == img->scale && o->scale_y == img->scale_y) {
                VkCommandBuffer cb = rec().cmd();
                std::vector<VkImageCopy> regions;
                for (u32 l = 0; l < o->info.levels && s.level + l < img->info.levels; l++) {
                    VkImageCopy c{};
                    u32 layers = std::min(o->info.layers, img->info.layers - s.layer);
                    c.srcSubresource = {o->aspect, l, 0, layers};
                    c.dstSubresource = {img->aspect, s.level + l, s.layer, layers};
                    c.extent = {o->level_width(l), o->level_height(l), o->level_depth(l)};
                    regions.push_back(c);
                }
                if (o->aspect == img->aspect && !regions.empty()) {
                    rec().barrier();
                    vkCmdCopyImage(cb, o->image, VK_IMAGE_LAYOUT_GENERAL, img->image, VK_IMAGE_LAYOUT_GENERAL,
                                   (u32)regions.size(), regions.data());
                    rec().barrier();
                    img->gpu_modified = true;
                    done = true;
                }
            }
            if (!done && o->addr == addr && o->info.type == img->info.type && o->aspect == img->aspect &&
                !o->info.fmt.compressed && !img->info.fmt.compressed &&
                vk_format_bytes(o->info.fmt.vk) == vk_format_bytes(img->info.fmt.vk)) {
                transfer_image(o, img, std::min(o->info.layers, img->info.layers));
                img->gpu_modified = true;
            }
        }
        // Keep overlapping images alive unless the new one fully covers them from a different base
        // (memory reuse). Same-address aliases (a render target allocated with a padded width and
        // sampled at its real size, format reinterpretations) and images that merely touch the new
        // range are still valid; destroying them here made the two aliases recreate each other on
        // every lookup (thousands of image creations and gigabytes of re-uploads per minute of battle).
        bool subsumed = o->addr != addr && o->addr >= addr && o->addr + o->guest_size <= addr + img->guest_size;
        if (subsumed) destroy(o);
    }
    img->last_used = rec().serial();
    return img;
}

void TextureCache::validate(Image* img) {
    if (img->cpu_dirty) {
        img->cpu_dirty = false;
        upload(img);
        img->gpu_modified = false;
        return;
    }
    if (img->gpu_modified || img->checked_epoch == epoch_) return;
    // CPU writes that bypass invalidate_region are caught by re-hashing, but hashing every bound
    // texture every frame costs more than the frame: re-check small textures often, big ones rarely,
    // and cap the bytes hashed per frame (the remainder is checked on later frames).
    // Small textures are often CPU-written lookup tables (per-frame curves): check them every frame.
    u64 interval = img->readable < (64u << 10) ? 1 : img->readable < (256u << 10) ? 2 : img->readable < (4u << 20) ? 8 : 32;
    if (epoch_ - img->checked_epoch < interval) return;
    if (hash_budget_epoch_ != epoch_) {
        hash_budget_epoch_ = epoch_;
        hash_budget_ = 48u << 20;
    }
    if (img->readable > hash_budget_) return;
    hash_budget_ -= img->readable;
    img->checked_epoch = epoch_;
    u8* p = (u8*)img->addr;
    u64 h;
    {
        SlowTimer t("texture hash check", 30.0);
        h = hash_bytes(p, img->readable);
    }
    if (h != img->hash) upload(img);
}

void TextureCache::alloc_image(Image* img) {
    const ImageInfo& info = img->info;
    ProfScope* ps = new ProfScope(kProfCreateImage);
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = info.type == ImageType::e1D ? VK_IMAGE_TYPE_1D
                   : info.type == ImageType::e3D ? VK_IMAGE_TYPE_3D
                                                 : VK_IMAGE_TYPE_2D;
    ci.format = info.fmt.vk;
    ci.extent = {img->phys_w(), info.type == ImageType::e1D ? 1 : img->phys_h(),
                 info.type == ImageType::e3D ? info.depth : 1};
    ci.mipLevels = info.levels;
    ci.arrayLayers = info.layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkFormatFeatureFlags ff = format_features(info.fmt.vk);
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (ff & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ci.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) ci.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ci.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (!info.fmt.depth && !info.fmt.stencil && !info.fmt.compressed) {
        ci.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (info.fmt.vk == VK_FORMAT_R32G32B32_SFLOAT || info.fmt.vk == VK_FORMAT_R32G32B32_UINT ||
            info.fmt.vk == VK_FORMAT_R32G32B32_SINT)
            ci.usage &= ~(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    }
    if (info.type == ImageType::e2D && info.layers >= 6 && info.width == info.height)
        ci.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    if (info.type == ImageType::e3D) ci.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    VkResult r = vkCreateImage(dev(), &ci, nullptr, &img->image);
    if (r != VK_SUCCESS) {
        hw_log("vk: vkCreateImage failed (%d) fmt %d %ux%ux%u layers %u levels %u", (int)r, (int)ci.format,
               info.width, info.height, info.depth, info.layers, info.levels);
        hw_fatal("vk: image creation");
    }
    delete ps;
    ps = new ProfScope(kProfAllocBind);
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev(), img->image, &req);
    Alloc a = allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkBindImageMemory(dev(), img->image, a.mem, a.offset));
    g_allocs[(u64)img->image] = a;
    delete ps;
    // initial layout transition to GENERAL
    VkCommandBuffer cb = rec().cmd();
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img->image;
    b.subresourceRange = {img->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    // Fresh VRAM holds whatever the previous owner left there. Depth images are never uploaded, and
    // colour images only receive the committed part of guest memory, so define every texel first:
    // render targets the game draws before clearing (first battle frames) otherwise show garbage.
    if (!vk_format_compressed(info.fmt.vk)) {  // clears are not allowed on block-compressed formats
        VkImageSubresourceRange all{img->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        if (img->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
            static const float init_depth = getenv("HWDER_VK_INIT_DEPTH") ? (float)atof(getenv("HWDER_VK_INIT_DEPTH")) : 1.0f;
            VkClearDepthStencilValue ds{init_depth, 0};
            vkCmdClearDepthStencilImage(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &ds, 1, &all);
        } else {
            VkClearColorValue zero{};
            vkCmdClearColorImage(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
        }
        rec().barrier();
    }
}

bool TextureCache::scalable(const ImageInfo& info) {
    return info.render_target && info.type == ImageType::e2D && info.levels == 1 && info.width >= 1280 && !info.fmt.compressed &&
           !info.fmt.astc;
}

// Runtime scale change (F1 menu): drop every image whose physical size would differ under the new
// scale; render targets are redrawn each frame, so they simply come back at the new size.
void TextureCache::apply_scale(float sx, float sy) {
    if (sx == g_res_scale && sy == g_res_scale_y) return;
    g_res_scale = sx;
    g_res_scale_y = sy;
    bool any = sx != 1.0f || sy != 1.0f;
    std::vector<Image*> dead;
    for (auto& [pg, v] : pages_)
        for (Image* img : v)
            if ((img->scale != 1.0f || img->scale_y != 1.0f || (any && scalable(img->info))) && (img->scale != sx || img->scale_y != sy) &&
                std::find(dead.begin(), dead.end(), img) == dead.end())
                dead.push_back(img);
    rec().wait_idle();
    for (Image* img : dead) destroy(img);
    hw_log("vk: internal resolution scale -> %.3fx x %.3fx (%zu render targets recreated)", sx, sy, dead.size());
}

Image* TextureCache::make_shadow(Image* src) {
    Image* img = new Image;
    img->info = src->info;
    img->addr = src->addr;
    img->guest_size = 0;  // never linked into the page map
    img->readable = 0;
    img->aspect = src->aspect;
    img->scale = src->scale;
    img->scale_y = src->scale_y;
    img->level_offset[0] = 0;
    alloc_image(img);
    img->gpu_modified = true;
    return img;
}

Image* TextureCache::create(u64 addr, const ImageInfo& info) {
    n_creates++;
    generation++;
    Image* img = new Image;
    img->info = info;
    img->addr = addr;
    {
        static bool init = false;
        if (!init) {
            init = true;
            const char* e = getenv("HWDER_RES_SCALE");
            const char* ex = getenv("HWDER_RES_SCALE_X");  // test aid: emulate another window aspect
            const char* ey = getenv("HWDER_RES_SCALE_Y");
            g_res_scale = g_res_scale_y = e ? (float)atof(e) : settings::video().res_scale;
            if (!e && settings::video().unlocked_res) {  // render at the screen's size and aspect
                g_res_scale = GetSystemMetrics(SM_CXSCREEN) / 1920.0f;
                g_res_scale_y = GetSystemMetrics(SM_CYSCREEN) / 1080.0f;
            }
            if (ex) g_res_scale = (float)atof(ex);
            if (ey) g_res_scale_y = (float)atof(ey);
            g_res_scale = std::clamp(g_res_scale, 0.5f, 4.0f);
            g_res_scale_y = std::clamp(g_res_scale_y, 0.5f, 4.0f);
            if (g_res_scale != 1.0f || g_res_scale_y != 1.0f)
                hw_log("vk: internal resolution scale %.3fx x %.3fx for large render targets", g_res_scale, g_res_scale_y);
        }
    }
    // Scale the scene-sized render targets (and their depth buffers); small post-process targets,
    // shadow maps and all sampled textures stay at guest size.
    if ((g_res_scale != 1.0f || g_res_scale_y != 1.0f) && scalable(info)) {
        img->scale = g_res_scale;
        img->scale_y = g_res_scale_y;
    }
    img->guest_size = compute_layout(info, img->level_offset, &img->layer_stride);
    {
        ProfScope ps(kProfReadable);
        img->readable = readable_size(addr, img->guest_size);
    }
    img->aspect = aspect_of(info.fmt);
    alloc_image(img);
    link(img);
    if (has_same_addr_alias(img)) tex_gen++;  // plain-texture TIC cache entries at this address must go through get() again
    upload(img);
    return img;
}

void TextureCache::destroy(Image* img, bool count) {
    generation++;
    tex_gen++;
    if (count) destroy_count++;
    unlink(img);
    VkImage image = img->image;
    std::vector<VkImageView> views;
    for (auto& [k, v] : img->views) views.push_back(v);
    Alloc a = g_allocs[(u64)image];
    g_allocs.erase((u64)image);
    delete img;
    // Keep alive until the GPU (and the presenter, a few frames behind) is done with it.
    retire_resource([image, views, a] {
        for (VkImageView v : views) vkDestroyImageView(dev(), v, nullptr);
        vkDestroyImage(dev(), image, nullptr);
        release(a);
    });
}

// A guest-sized scratch VkImage with the format of `img`, for transfers to/from scaled render targets.
struct ScratchImage {
    VkImage image = VK_NULL_HANDLE;
    Alloc alloc{};
};
static ScratchImage make_scratch(Image* img) {
    ScratchImage t;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = img->info.fmt.vk;
    ci.extent = {img->info.width, img->info.height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = img->info.layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (vkCreateImage(dev(), &ci, nullptr, &t.image) != VK_SUCCESS) return t;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev(), t.image, &req);
    t.alloc = allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkBindImageMemory(dev(), t.image, t.alloc.mem, t.alloc.offset);
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.image;
    b.subresourceRange = {img->aspect, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(rec().cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    return t;
}
static void blit_scratch(VkImage src, u32 sw, u32 sh, VkImage dst, u32 dw, u32 dh, VkImageAspectFlags aspect, u32 layers) {
    rec().barrier();
    VkImageBlit b{};
    b.srcSubresource = {aspect, 0, 0, layers};
    b.dstSubresource = {aspect, 0, 0, layers};
    b.srcOffsets[1] = {(s32)sw, (s32)sh, 1};
    b.dstOffsets[1] = {(s32)dw, (s32)dh, 1};
    vkCmdBlitImage(rec().cmd(), src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                   aspect == VK_IMAGE_ASPECT_COLOR_BIT ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    rec().barrier();
}
static void free_scratch(ScratchImage t) {
    retire_resource([t] {
        vkDestroyImage(dev(), t.image, nullptr);
        release(t.alloc);
    });
}

void TextureCache::upload(Image* img) {
    SlowTimer timer("texture upload", 30.0);
    ProfScope prof(kProfUpload);
    n_uploads++;
    // CPU data is the newest content of this memory: an older GPU-written alias must not be copied over it.
    img->write_seq = ++write_counter_;
    upload_bytes += img->readable;
    const ImageInfo& info = img->info;
    u8* guest = (u8*)img->addr;
    img->hash = hash_bytes(guest, img->readable);
    img->checked_epoch = epoch_;
    if (info.fmt.depth || info.fmt.stencil) return;  // depth contents come from the GPU
    // linear staging size
    u64 total = 0;
    for (u32 l = 0; l < info.levels; l++) {
        LevelParams p = level_params(info, l);
        u64 slice = info.fmt.astc ? (u64)p.w * p.h * 4 : (u64)p.width_bytes * p.by;
        total += slice * p.d * info.layers;
    }
    Slice st;
    {
        ProfScope ps(kProfStage);
        st = rec().stage(total, 16);
    }
    ProfScope decode_scope(kProfDecode);
    std::vector<VkBufferImageCopy> regions;
    std::vector<u8> tmp;
    u64 off = 0;
    for (u32 layer = 0; layer < info.layers; layer++) {
        for (u32 l = 0; l < info.levels; l++) {
            LevelParams p = level_params(info, l);
            u64 goff = layer * img->layer_stride + img->level_offset[l];
            if (goff >= img->readable) continue;
            u64 avail = img->readable - goff;
            u64 lin_size = (u64)p.width_bytes * p.by * p.d;
            u8* dst = st.ptr + off;
            u8* lin = dst;
            if (info.fmt.astc) {
                tmp.resize(lin_size);
                lin = tmp.data();
            }
            if (info.block_linear) {
                bl_copy(p, guest + goff, lin, true, avail);
            } else {
                u32 pitch = info.pitch ? info.pitch : p.width_bytes;
                for (u32 z = 0; z < p.d; z++)
                    for (u32 y = 0; y < p.by; y++) {
                        u64 src = (u64)z * pitch * p.by + (u64)y * pitch;
                        if (src + p.width_bytes > avail) break;
                        memcpy(lin + ((u64)z * p.by + y) * p.width_bytes, guest + goff + src, p.width_bytes);
                    }
            }
            u64 out_size = lin_size;
            if (info.fmt.astc) {
                out_size = (u64)p.w * p.h * p.d * 4;
                // Decoded textures are cached on disk by content hash: the CPU decode is paid once per machine.
                static bool cache_ok = CreateDirectoryA("cache", nullptr) || GetLastError() == ERROR_ALREADY_EXISTS;
                static bool cache_dir = cache_ok && (CreateDirectoryA("cache\astc", nullptr) || GetLastError() == ERROR_ALREADY_EXISTS);
                char path[128];
                u64 h = hash_bytes(lin, lin_size, ((u64)p.w << 40) ^ ((u64)p.h << 20) ^ p.d ^ ((u64)info.fmt.bw << 56) ^ ((u64)info.fmt.bh << 60));
                snprintf(path, sizeof(path), "cache\astc\%016llx.rgba", (unsigned long long)h);
                bool loaded = false;
                if (cache_dir) {
                    if (FILE* f = fopen(path, "rb")) {
                        loaded = fread(dst, 1, out_size, f) == out_size;
                        fclose(f);
                    }
                }
                if (!loaded) {
                    decode_astc(lin, p.w, p.h, p.d, info.fmt.bw, info.fmt.bh, dst);
                    if (cache_dir) {
                        if (FILE* f = fopen(path, "wb")) {
                            fwrite(dst, 1, out_size, f);
                            fclose(f);
                        }
                    }
                }
            }
            VkBufferImageCopy c{};
            c.bufferOffset = st.offset + off;
            c.imageSubresource = {img->aspect, l, layer, 1};
            c.imageExtent = {p.w, info.type == ImageType::e1D ? 1 : p.h, p.d};
            regions.push_back(c);
            off += out_size;
        }
    }
    if (regions.empty()) return;
    VkCommandBuffer cb = rec().cmd();
    rec().barrier();
    if (img->scale != 1.0f || img->scale_y != 1.0f) {
        ScratchImage t = make_scratch(img);
        if (t.image) {
            vkCmdCopyBufferToImage(cb, st.buf, t.image, VK_IMAGE_LAYOUT_GENERAL, (u32)regions.size(), regions.data());
            blit_scratch(t.image, img->info.width, img->info.height, img->image, img->phys_w(), img->phys_h(), img->aspect, img->info.layers);
            free_scratch(t);
        }
    } else {
        vkCmdCopyBufferToImage(cb, st.buf, img->image, VK_IMAGE_LAYOUT_GENERAL, (u32)regions.size(), regions.data());
    }
    rec().barrier();
}

void TextureCache::download(Image* img) {
    SlowTimer timer("texture download", 30.0);
    n_downloads++;
    const ImageInfo& info = img->info;
    if (info.fmt.depth || info.fmt.stencil || info.fmt.astc || info.fmt.compressed) {
        img->gpu_modified = false;
        return;
    }
    u64 total = 0;
    for (u32 l = 0; l < info.levels; l++) {
        LevelParams p = level_params(info, l);
        total += (u64)p.width_bytes * p.by * p.d * info.layers;
    }
    Buffer rb = create_buffer(total, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    std::vector<VkBufferImageCopy> regions;
    u64 off = 0;
    for (u32 layer = 0; layer < info.layers; layer++)
        for (u32 l = 0; l < info.levels; l++) {
            LevelParams p = level_params(info, l);
            VkBufferImageCopy c{};
            c.bufferOffset = off;
            c.imageSubresource = {img->aspect, l, layer, 1};
            c.imageExtent = {p.w, info.type == ImageType::e1D ? 1 : p.h, p.d};
            regions.push_back(c);
            off += (u64)p.width_bytes * p.by * p.d;
        }
    ScratchImage scratch;
    VkImage src_image = img->image;
    if (img->scale != 1.0f || img->scale_y != 1.0f) {  // scaled render target: downsample to the guest size first
        scratch = make_scratch(img);
        if (scratch.image) {
            blit_scratch(img->image, img->phys_w(), img->phys_h(), scratch.image, img->info.width, img->info.height, img->aspect, img->info.layers);
            src_image = scratch.image;
        }
    }
    rec().barrier();
    vkCmdCopyImageToBuffer(rec().cmd(), src_image, VK_IMAGE_LAYOUT_GENERAL, rb.buf, (u32)regions.size(),
                           regions.data());
    rec().barrier();
    rec().wait_idle();
    if (scratch.image) free_scratch(scratch);
    VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    mr.memory = rb.mem;
    mr.size = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(dev(), 1, &mr);
    u8* guest = (u8*)img->addr;
    off = 0;
    for (u32 layer = 0; layer < info.layers; layer++)
        for (u32 l = 0; l < info.levels; l++) {
            LevelParams p = level_params(info, l);
            u64 goff = layer * img->layer_stride + img->level_offset[l];
            u64 lin_size = (u64)p.width_bytes * p.by * p.d;
            if (goff < img->readable) {
                u64 avail = img->readable - goff;
                if (info.block_linear) {
                    bl_copy(p, guest + goff, rb.map + off, false, avail);
                } else {
                    u32 pitch = info.pitch ? info.pitch : p.width_bytes;
                    for (u32 z = 0; z < p.d; z++)
                        for (u32 y = 0; y < p.by; y++) {
                            u64 d = (u64)z * pitch * p.by + (u64)y * pitch;
                            if (d + p.width_bytes > avail) break;
                            memcpy(guest + goff + d, rb.map + off + ((u64)z * p.by + y) * p.width_bytes,
                                   p.width_bytes);
                        }
                }
            }
            off += lin_size;
        }
    destroy_buffer(rb);
    img->gpu_modified = false;
    img->hash = hash_bytes(guest, img->readable);
    img->checked_epoch = epoch_;
}

void TextureCache::invalidate(u64 addr, u64 size) {
    std::vector<Image*> v;
    collect_overlaps(addr, size, v);
    for (Image* img : v) img->cpu_dirty = true;
}

void TextureCache::flush(u64 addr, u64 size) {
    SlowTimer timer("texture flush", 30.0);
    std::vector<Image*> v;
    collect_overlaps(addr, size, v);
    for (Image* img : v)
        if (img->gpu_modified) download(img);
}

bool TextureCache::is_gpu_modified(u64 addr, u64 size) {
    std::vector<Image*> v;
    collect_overlaps(addr, size, v);
    for (Image* img : v)
        if (img->gpu_modified) return true;
    return false;
}

VkImageView TextureCache::view(Image* img, VkFormat fmt, VkImageViewType type, u32 base_level, u32 levels,
                               u32 base_layer, u32 layers, VkComponentMapping swz, VkImageAspectFlags aspect) {
    if (!fmt) fmt = img->info.fmt.vk;
    if (img->info.fmt.depth || img->info.fmt.stencil || img->info.fmt.compressed || img->info.fmt.astc)
        fmt = img->info.fmt.vk;
    else if (vk_format_bytes(fmt) != vk_format_bytes(img->info.fmt.vk) || vk_format_compressed(fmt))
        fmt = img->info.fmt.vk;
    if (!aspect) aspect = img->aspect;
    base_level = std::min(base_level, img->info.levels - 1);
    levels = std::max(1u, std::min(levels, img->info.levels - base_level));
    if (img->info.type == ImageType::e3D) {
        base_layer = 0;
        layers = 1;
        if (type != VK_IMAGE_VIEW_TYPE_3D) type = VK_IMAGE_VIEW_TYPE_3D;
    } else {
        if (type == VK_IMAGE_VIEW_TYPE_3D) type = VK_IMAGE_VIEW_TYPE_2D;
        base_layer = std::min(base_layer, img->info.layers - 1);
        layers = std::max(1u, std::min(layers, img->info.layers - base_layer));
        if ((type == VK_IMAGE_VIEW_TYPE_CUBE || type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) &&
            (layers < 6 || img->info.width != img->info.height))
            type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        if (type == VK_IMAGE_VIEW_TYPE_CUBE) layers = 6;
        if (type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) layers -= layers % 6;
        if (type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_1D) layers = 1;
        if (img->info.type == ImageType::e1D && type != VK_IMAGE_VIEW_TYPE_1D && type != VK_IMAGE_VIEW_TYPE_1D_ARRAY)
            type = layers > 1 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
        if (img->info.type == ImageType::e2D && (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY))
            type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
    ViewKey k{fmt, type, base_level, levels, base_layer, layers,
              (u32)swz.r | ((u32)swz.g << 8) | ((u32)swz.b << 16) | ((u32)swz.a << 24), aspect};
    auto it = img->views.find(k);
    if (it != img->views.end()) return it->second;
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image = img->image;
    ci.viewType = type;
    ci.format = fmt;
    ci.components = swz;
    ci.subresourceRange = {aspect, base_level, levels, base_layer, layers};
    VkImageView v = VK_NULL_HANDLE;
    VkResult r = vkCreateImageView(dev(), &ci, nullptr, &v);
    if (r != VK_SUCCESS) {
        hw_log("vk: vkCreateImageView failed %d (fmt %d type %d)", (int)r, (int)fmt, (int)type);
        return VK_NULL_HANDLE;
    }
    img->views[k] = v;
    return v;
}

VkSampler TextureCache::sampler(const u32* tsc) {
    const int aniso_override = settings::video().aniso;
    u64 key = hash_bytes(tsc, 32, (u64)aniso_override);
    auto it = samplers_.find(key);
    if (it != samplers_.end()) return it->second;
    auto wrap = [](u32 w) {
        switch (w) {
        case 0: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case 1: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case 2: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 3: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        default: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        }
    };
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.addressModeU = wrap(tsc[0] & 7);
    ci.addressModeV = wrap((tsc[0] >> 3) & 7);
    ci.addressModeW = wrap((tsc[0] >> 6) & 7);
    ci.compareEnable = (tsc[0] >> 9) & 1;
    ci.compareOp = (VkCompareOp)((tsc[0] >> 10) & 7);
    u32 aniso = (tsc[0] >> 20) & 7;
    static const float kAniso[] = {1, 2, 4, 6, 8, 10, 12, 16};
    if (aniso) {
        ci.anisotropyEnable = VK_TRUE;
        ci.maxAnisotropy = kAniso[aniso];
    }
    if (aniso_override > 1 && ((tsc[1] >> 6) & 3) == 3) {  // user override, trilinear (mipmapped) samplers only
        ci.anisotropyEnable = VK_TRUE;
        ci.maxAnisotropy = std::max(ci.maxAnisotropy, (float)aniso_override);
    }
    u32 mag = tsc[1] & 3, min = (tsc[1] >> 4) & 3, mip = (tsc[1] >> 6) & 3;
    ci.magFilter = mag == 2 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter = min == 2 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.mipmapMode = mip == 3 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    s32 bias = (s32)(((tsc[1] >> 12) & 0x1FFF) << 19) >> 19;
    ci.mipLodBias = bias / 256.0f;
    ci.minLod = (tsc[2] & 0xFFF) / 256.0f;
    ci.maxLod = ((tsc[2] >> 12) & 0xFFF) / 256.0f;
    if (mip <= 1) ci.maxLod = std::min(ci.maxLod, 0.25f), ci.minLod = std::min(ci.minLod, ci.maxLod);
    float border[4];
    memcpy(border, &tsc[4], 16);
    if (border[3] < 0.5f) ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    else if (border[0] > 0.5f) ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    else ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    ci.unnormalizedCoordinates = VK_FALSE;
    VkSampler s = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(dev(), &ci, nullptr, &s));
    samplers_[key] = s;
    return s;
}

bool TextureCache::info_from_tic(const u32* tic, ImageInfo& info, u64& addr, u32& base_level, u32& levels,
                                 u32& base_layer) {
    info = ImageInfo{};
    info.fmt = tic_format(tic);
    if (!info.fmt.valid()) return false;
    u64 va = tic[1] | ((u64)(tic[2] & 0xFFFF) << 32);
    u8* p = mm_->translate(va);
    if (!p) return false;
    addr = (u64)p;
    u32 hv = (tic[2] >> 21) & 7;
    u32 type = (tic[4] >> 23) & 0xF;
    info.width = (tic[4] & 0xFFFF) + 1;
    info.height = (tic[5] & 0xFFFF) + 1;
    u32 depth = ((tic[5] >> 16) & 0x3FFF) + 1;
    info.block_linear = hv == 3 || hv == 4;
    if (info.block_linear) {
        info.bh = (tic[3] >> 3) & 7;
        info.bd = (tic[3] >> 6) & 7;
        info.tws = (tic[3] >> 10) & 7;
    } else if (hv == 1 || hv == 2) {
        info.pitch = (tic[3] & 0xFFFF) << 5;
    } else {
        return false;  // buffer textures are handled by the caller
    }
    info.levels = info.block_linear ? ((tic[3] >> 28) & 0xF) + 1 : 1;
    switch (type) {
    case 0: info.type = ImageType::e1D; info.height = 1; break;
    case 1: case 7: info.type = ImageType::e2D; break;
    case 2: info.type = ImageType::e3D; info.depth = depth; break;
    case 3: info.type = ImageType::e2D; info.layers = 6; info.cube = true; break;
    case 4: info.type = ImageType::e1D; info.height = 1; info.layers = depth; break;
    case 5: info.type = ImageType::e2D; info.layers = depth; break;
    case 8: info.type = ImageType::e2D; info.layers = depth * 6; info.cube = true; break;
    default: info.type = ImageType::e2D; break;
    }
    if (type == 7) info.levels = 1;
    u32 res_min = tic[7] & 0xF, res_max = (tic[7] >> 4) & 0xF;
    base_level = std::min(res_min, info.levels - 1);
    levels = res_max >= res_min ? res_max - res_min + 1 : 1;
    base_layer = ((tic[4] >> 16) & 7) | (((tic[2] >> 16) & 0x1F) << 3) | (((tic[2] >> 29) & 7) << 8);
    return true;
}

void TextureCache::collect_garbage() {}

}  // namespace gpu::vk
