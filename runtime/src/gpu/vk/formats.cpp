// Maxwell texture / render target / vertex formats -> Vulkan formats.
#include <mutex>

#include "vk_common.h"

namespace gpu::vk {

namespace {
enum Comp : u32 { SNORM = 1, UNORM = 2, SINT = 3, UINT = 4, SNORM16 = 5, UNORM16 = 6, FLOAT = 7 };

struct TicRow {
    u32 code;
    u8 bpb, bw, bh;
    VkFormat unorm, snorm, uint, sint, flt, srgb;
};
#define U VK_FORMAT_UNDEFINED
const TicRow kTic[] = {
    {0x01, 16, 1, 1, U, U, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, U},
    {0x02, 12, 1, 1, U, U, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32_SFLOAT, U},
    {0x03, 8, 1, 1, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT,
     VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R16G16B16A16_SFLOAT, U},
    {0x04, 8, 1, 1, U, U, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32_SFLOAT, U},
    {0x07, 4, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
     U, VK_FORMAT_R8G8B8A8_SRGB},
    {0x08, 4, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
     U, VK_FORMAT_R8G8B8A8_SRGB},
    {0x09, 4, 1, 1, VK_FORMAT_A2B10G10R10_UNORM_PACK32, U, VK_FORMAT_A2B10G10R10_UINT_PACK32, U, U, U},
    {0x0C, 4, 1, 1, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT,
     VK_FORMAT_R16G16_SFLOAT, U},
    {0x0F, 4, 1, 1, U, U, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R32_SFLOAT, U},
    {0x10, 16, 4, 4, U, U, U, U, VK_FORMAT_BC6H_SFLOAT_BLOCK, U},
    {0x11, 16, 4, 4, U, U, U, U, VK_FORMAT_BC6H_UFLOAT_BLOCK, U},
    {0x12, 2, 1, 1, VK_FORMAT_A4B4G4R4_UNORM_PACK16, U, U, U, U, U},
    {0x14, 2, 1, 1, VK_FORMAT_A1R5G5B5_UNORM_PACK16, U, U, U, U, U},
    {0x15, 2, 1, 1, VK_FORMAT_B5G6R5_UNORM_PACK16, U, U, U, U, U},
    {0x17, 16, 4, 4, VK_FORMAT_BC7_UNORM_BLOCK, U, U, U, U, VK_FORMAT_BC7_SRGB_BLOCK},
    {0x18, 2, 1, 1, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, U, U},
    {0x1B, 2, 1, 1, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT,
     VK_FORMAT_R16_SFLOAT, U},
    {0x1D, 1, 1, 1, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, U, U},
    {0x1E, 1, 1, 1, VK_FORMAT_R4G4_UNORM_PACK8, U, U, U, U, U},
    {0x20, 4, 1, 1, U, U, U, U, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, U},
    {0x21, 4, 1, 1, U, U, U, U, VK_FORMAT_B10G11R11_UFLOAT_PACK32, U},
    {0x24, 8, 4, 4, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, U, U, U, U, VK_FORMAT_BC1_RGBA_SRGB_BLOCK},
    {0x25, 16, 4, 4, VK_FORMAT_BC2_UNORM_BLOCK, U, U, U, U, VK_FORMAT_BC2_SRGB_BLOCK},
    {0x26, 16, 4, 4, VK_FORMAT_BC3_UNORM_BLOCK, U, U, U, U, VK_FORMAT_BC3_SRGB_BLOCK},
    {0x27, 8, 4, 4, VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_BC4_SNORM_BLOCK, U, U, U, U},
    {0x28, 16, 4, 4, VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC5_SNORM_BLOCK, U, U, U, U},
};
#undef U

struct AstcRow {
    u32 code;
    u8 bw, bh;
};
const AstcRow kAstc[] = {{0x40, 4, 4},  {0x41, 5, 5},  {0x42, 6, 6},   {0x44, 8, 8},  {0x45, 10, 10},
                         {0x46, 12, 12}, {0x50, 5, 4}, {0x51, 6, 5},   {0x52, 8, 6},  {0x53, 10, 8},
                         {0x54, 12, 10}, {0x55, 8, 5}, {0x56, 10, 5},  {0x57, 10, 6}};

FormatInfo make(VkFormat f, u8 bpb, u8 bw = 1, u8 bh = 1) {
    FormatInfo fi;
    fi.vk = f;
    fi.bpb = bpb;
    fi.bw = bw;
    fi.bh = bh;
    fi.compressed = vk_format_compressed(f);
    switch (f) {
    case VK_FORMAT_D16_UNORM: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_X8_D24_UNORM_PACK32: fi.depth = true; break;
    case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT: fi.depth = fi.stencil = true; break;
    case VK_FORMAT_S8_UINT: fi.stencil = true; break;
    default: break;
    }
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32_SINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
        fi.integer = true;
        break;
    default: break;
    }
    return fi;
}
}  // namespace

bool vk_format_compressed(VkFormat f) {
    return (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
}

u32 vk_format_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R4G4_UNORM_PACK8: case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT: case VK_FORMAT_S8_UINT:
        return 1;
    case VK_FORMAT_A4B4G4R4_UNORM_PACK16: case VK_FORMAT_R4G4B4A4_UNORM_PACK16: case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
    case VK_FORMAT_R5G5B5A1_UNORM_PACK16: case VK_FORMAT_B5G6R5_UNORM_PACK16: case VK_FORMAT_R5G6B5_UNORM_PACK16:
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SNORM: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_D16_UNORM:
        return 2;
    case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK: case VK_FORMAT_BC4_SNORM_BLOCK:
        return 8;
    case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32_SINT: case VK_FORMAT_R32G32B32_SFLOAT:
        return 12;
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK: case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK: case VK_FORMAT_BC5_UNORM_BLOCK: case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK: case VK_FORMAT_BC6H_UFLOAT_BLOCK: case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
        return 16;
    default:
        return 4;
    }
}

VkFormatFeatureFlags format_features(VkFormat f) {
    static std::mutex m;
    static std::unordered_map<u32, VkFormatFeatureFlags> cache;
    std::lock_guard<std::mutex> l(m);
    auto it = cache.find((u32)f);
    if (it != cache.end()) return it->second;
    VkFormatProperties p{};
    vkGetPhysicalDeviceFormatProperties(display::device().phys, f, &p);
    cache[(u32)f] = p.optimalTilingFeatures;
    return p.optimalTilingFeatures;
}

FormatInfo tic_format(const u32* tic) {
    u32 code = tic[0] & 0x7F;
    u32 r = (tic[0] >> 7) & 7;
    bool srgb = (tic[4] >> 22) & 1;
    for (const AstcRow& a : kAstc) {
        if (a.code != code) continue;
        FormatInfo fi = make(srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM, 16, a.bw, a.bh);
        fi.astc = true;
        return fi;
    }
    switch (code) {  // depth / stencil
    case 0x05: case 0x30: return make(VK_FORMAT_D32_SFLOAT_S8_UINT, 8);
    case 0x0D: case 0x0E: case 0x29: case 0x2A: case 0x2B: return make(VK_FORMAT_D24_UNORM_S8_UINT, 4);
    case 0x2F: return make(VK_FORMAT_D32_SFLOAT, 4);
    case 0x3A: return make(VK_FORMAT_D16_UNORM, 2);
    default: break;
    }
    for (const TicRow& t : kTic) {
        if (t.code != code) continue;
        VkFormat f = VK_FORMAT_UNDEFINED;
        switch (r) {
        case SNORM: case SNORM16: f = t.snorm; break;
        case UNORM: case UNORM16: f = srgb && t.srgb ? t.srgb : t.unorm; break;
        case UINT: f = t.uint; break;
        case SINT: f = t.sint; break;
        case FLOAT: f = t.flt; break;
        }
        if (!f) f = t.flt ? t.flt : t.unorm ? t.unorm : t.uint ? t.uint : t.snorm;
        FormatInfo fi = make(f, t.bpb, t.bw, t.bh);
        if (code == 0x14) {  // A1B5G5R5 stored as A1R5G5B5: swap red and blue
            fi.swizzle = {VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_A};
        }
        return fi;
    }
    VK_LOG_ONCE("vk: unsupported texture format 0x%x (component %u)", code, r);
    return FormatInfo{};
}

FormatInfo rt_format(u32 f) {
    switch (f) {
    case 0xC0: return make(VK_FORMAT_R32G32B32A32_SFLOAT, 16);
    case 0xC1: return make(VK_FORMAT_R32G32B32A32_SINT, 16);
    case 0xC2: return make(VK_FORMAT_R32G32B32A32_UINT, 16);
    case 0xC3: return make(VK_FORMAT_R32G32B32A32_SFLOAT, 16);
    case 0xC4: return make(VK_FORMAT_R32G32B32A32_SINT, 16);
    case 0xC5: return make(VK_FORMAT_R32G32B32A32_UINT, 16);
    case 0xC6: return make(VK_FORMAT_R16G16B16A16_UNORM, 8);
    case 0xC7: return make(VK_FORMAT_R16G16B16A16_SNORM, 8);
    case 0xC8: return make(VK_FORMAT_R16G16B16A16_SINT, 8);
    case 0xC9: return make(VK_FORMAT_R16G16B16A16_UINT, 8);
    case 0xCA: return make(VK_FORMAT_R16G16B16A16_SFLOAT, 8);
    case 0xCB: return make(VK_FORMAT_R32G32_SFLOAT, 8);
    case 0xCC: return make(VK_FORMAT_R32G32_SINT, 8);
    case 0xCD: return make(VK_FORMAT_R32G32_UINT, 8);
    case 0xCE: return make(VK_FORMAT_R16G16B16A16_SFLOAT, 8);
    case 0xCF: return make(VK_FORMAT_B8G8R8A8_UNORM, 4);
    case 0xD0: return make(VK_FORMAT_B8G8R8A8_SRGB, 4);
    case 0xD1: return make(VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4);
    case 0xD2: return make(VK_FORMAT_A2B10G10R10_UINT_PACK32, 4);
    case 0xD5: return make(VK_FORMAT_R8G8B8A8_UNORM, 4);
    case 0xD6: return make(VK_FORMAT_R8G8B8A8_SRGB, 4);
    case 0xD7: return make(VK_FORMAT_R8G8B8A8_SNORM, 4);
    case 0xD8: return make(VK_FORMAT_R8G8B8A8_SINT, 4);
    case 0xD9: return make(VK_FORMAT_R8G8B8A8_UINT, 4);
    case 0xDA: return make(VK_FORMAT_R16G16_UNORM, 4);
    case 0xDB: return make(VK_FORMAT_R16G16_SNORM, 4);
    case 0xDC: return make(VK_FORMAT_R16G16_SINT, 4);
    case 0xDD: return make(VK_FORMAT_R16G16_UINT, 4);
    case 0xDE: return make(VK_FORMAT_R16G16_SFLOAT, 4);
    case 0xDF: return make(VK_FORMAT_A2R10G10B10_UNORM_PACK32, 4);
    case 0xE0: return make(VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4);
    case 0xE3: return make(VK_FORMAT_R32_SINT, 4);
    case 0xE4: return make(VK_FORMAT_R32_UINT, 4);
    case 0xE5: return make(VK_FORMAT_R32_SFLOAT, 4);
    case 0xE6: return make(VK_FORMAT_B8G8R8A8_UNORM, 4);
    case 0xE7: return make(VK_FORMAT_B8G8R8A8_SRGB, 4);
    case 0xE8: return make(VK_FORMAT_R5G6B5_UNORM_PACK16, 2);
    case 0xE9: return make(VK_FORMAT_A1R5G5B5_UNORM_PACK16, 2);
    case 0xEA: return make(VK_FORMAT_R8G8_UNORM, 2);
    case 0xEB: return make(VK_FORMAT_R8G8_SNORM, 2);
    case 0xEC: return make(VK_FORMAT_R8G8_SINT, 2);
    case 0xED: return make(VK_FORMAT_R8G8_UINT, 2);
    case 0xEE: return make(VK_FORMAT_R16_UNORM, 2);
    case 0xEF: return make(VK_FORMAT_R16_SNORM, 2);
    case 0xF0: return make(VK_FORMAT_R16_SINT, 2);
    case 0xF1: return make(VK_FORMAT_R16_UINT, 2);
    case 0xF2: return make(VK_FORMAT_R16_SFLOAT, 2);
    case 0xF3: return make(VK_FORMAT_R8_UNORM, 1);
    case 0xF4: return make(VK_FORMAT_R8_SNORM, 1);
    case 0xF5: return make(VK_FORMAT_R8_SINT, 1);
    case 0xF6: return make(VK_FORMAT_R8_UINT, 1);
    case 0xF7: return make(VK_FORMAT_R8_UNORM, 1);  // A8: alpha only (approximation)
    case 0xF8: return make(VK_FORMAT_A1R5G5B5_UNORM_PACK16, 2);
    case 0xF9: return make(VK_FORMAT_R8G8B8A8_UNORM, 4);
    case 0xFA: return make(VK_FORMAT_R8G8B8A8_SRGB, 4);
    case 0xFF: return make(VK_FORMAT_R32_SFLOAT, 4);
    default:
        if (f) VK_LOG_ONCE("vk: unsupported render target format 0x%x", f);
        return FormatInfo{};
    }
}

FormatInfo zeta_format(u32 f) {
    switch (f) {
    case 0x0A: return make(VK_FORMAT_D32_SFLOAT, 4);
    case 0x13: return make(VK_FORMAT_D16_UNORM, 2);
    case 0x14: case 0x15: case 0x16: case 0x18: return make(VK_FORMAT_D24_UNORM_S8_UINT, 4);
    case 0x17: return make(VK_FORMAT_D24_UNORM_S8_UINT, 1);  // S8: emulated
    case 0x19: return make(VK_FORMAT_D32_SFLOAT_S8_UINT, 8);
    default:
        if (f) VK_LOG_ONCE("vk: unsupported depth format 0x%x", f);
        return FormatInfo{};
    }
}

FormatInfo android_format(u32 f) {
    switch (f) {
    case 1: case 2: return make(VK_FORMAT_R8G8B8A8_UNORM, 4);   // RGBA_8888 / RGBX_8888
    case 4: return make(VK_FORMAT_R5G6B5_UNORM_PACK16, 2);      // RGB_565
    case 5: return make(VK_FORMAT_B8G8R8A8_UNORM, 4);           // BGRA_8888
    default: return make(VK_FORMAT_R8G8B8A8_UNORM, 4);
    }
}

VkFormat vertex_format(u32 attrib, u32* size_bytes) {
    u32 size = (attrib >> 21) & 0x3F, type = (attrib >> 27) & 7;
    bool bgra = attrib >> 31;
    struct Row {
        u32 size, bytes;
        VkFormat unorm, snorm, uint, sint, uscaled, sscaled, flt;
    };
    static const Row rows[] = {
        {0x01, 16, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
         VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT, VK_FORMAT_R32G32B32A32_SFLOAT},
        {0x02, 12, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT,
         VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32_SFLOAT},
        {0x03, 8, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT,
         VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R16G16B16A16_USCALED, VK_FORMAT_R16G16B16A16_SSCALED,
         VK_FORMAT_R16G16B16A16_SFLOAT},
        {0x04, 8, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
         VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32_SFLOAT},
        {0x05, 6, VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16_UINT,
         VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R16G16B16_USCALED, VK_FORMAT_R16G16B16_SSCALED,
         VK_FORMAT_R16G16B16_SFLOAT},
        {0x0A, 4, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
         VK_FORMAT_R8G8B8A8_USCALED, VK_FORMAT_R8G8B8A8_SSCALED, VK_FORMAT_UNDEFINED},
        {0x0F, 4, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT,
         VK_FORMAT_R16G16_USCALED, VK_FORMAT_R16G16_SSCALED, VK_FORMAT_R16G16_SFLOAT},
        {0x12, 4, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R32_UINT,
         VK_FORMAT_R32_SINT, VK_FORMAT_R32_SFLOAT},
        {0x13, 3, VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8_SINT,
         VK_FORMAT_R8G8B8_USCALED, VK_FORMAT_R8G8B8_SSCALED, VK_FORMAT_UNDEFINED},
        {0x18, 2, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT,
         VK_FORMAT_R8G8_USCALED, VK_FORMAT_R8G8_SSCALED, VK_FORMAT_UNDEFINED},
        {0x32, 2, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT,
         VK_FORMAT_R8G8_USCALED, VK_FORMAT_R8G8_SSCALED, VK_FORMAT_UNDEFINED},
        {0x1B, 2, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT,
         VK_FORMAT_R16_USCALED, VK_FORMAT_R16_SSCALED, VK_FORMAT_R16_SFLOAT},
        {0x1D, 1, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, VK_FORMAT_R8_USCALED,
         VK_FORMAT_R8_SSCALED, VK_FORMAT_UNDEFINED},
        {0x34, 1, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, VK_FORMAT_R8_USCALED,
         VK_FORMAT_R8_SSCALED, VK_FORMAT_UNDEFINED},
        {0x30, 4, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_SNORM_PACK32,
         VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_A2B10G10R10_SINT_PACK32, VK_FORMAT_A2B10G10R10_USCALED_PACK32,
         VK_FORMAT_A2B10G10R10_SSCALED_PACK32, VK_FORMAT_UNDEFINED},
        {0x31, 4, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED,
         VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_FORMAT_B10G11R11_UFLOAT_PACK32},
        {0x33, 4, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
         VK_FORMAT_R8G8B8A8_USCALED, VK_FORMAT_R8G8B8A8_SSCALED, VK_FORMAT_UNDEFINED},
    };
    for (const Row& r : rows) {
        if (r.size != size) continue;
        if (size_bytes) *size_bytes = r.bytes;
        VkFormat f = VK_FORMAT_UNDEFINED;
        switch (type) {
        case 1: f = r.snorm; break;
        case 2: f = r.unorm; break;
        case 3: f = r.sint; break;
        case 4: f = r.uint; break;
        case 5: f = r.uscaled; break;
        case 6: f = r.sscaled; break;
        case 7: f = r.flt; break;
        }
        if (!f) f = r.flt ? r.flt : r.unorm;
        if (bgra && f == VK_FORMAT_R8G8B8A8_UNORM) f = VK_FORMAT_B8G8R8A8_UNORM;
        // Fall back to a 4-component format when 3-component vertex formats are not supported.
        static std::unordered_map<u32, bool> ok;
        VkFormatProperties p{};
        vkGetPhysicalDeviceFormatProperties(display::device().phys, f, &p);
        if (!(p.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) {
            switch (f) {
            case VK_FORMAT_R8G8B8_UNORM: f = VK_FORMAT_R8G8B8A8_UNORM; break;
            case VK_FORMAT_R8G8B8_SNORM: f = VK_FORMAT_R8G8B8A8_SNORM; break;
            case VK_FORMAT_R8G8B8_UINT: f = VK_FORMAT_R8G8B8A8_UINT; break;
            case VK_FORMAT_R8G8B8_SINT: f = VK_FORMAT_R8G8B8A8_SINT; break;
            case VK_FORMAT_R8G8B8_USCALED: f = VK_FORMAT_R8G8B8A8_USCALED; break;
            case VK_FORMAT_R8G8B8_SSCALED: f = VK_FORMAT_R8G8B8A8_SSCALED; break;
            case VK_FORMAT_R16G16B16_UNORM: f = VK_FORMAT_R16G16B16A16_UNORM; break;
            case VK_FORMAT_R16G16B16_SNORM: f = VK_FORMAT_R16G16B16A16_SNORM; break;
            case VK_FORMAT_R16G16B16_UINT: f = VK_FORMAT_R16G16B16A16_UINT; break;
            case VK_FORMAT_R16G16B16_SINT: f = VK_FORMAT_R16G16B16A16_SINT; break;
            case VK_FORMAT_R16G16B16_USCALED: f = VK_FORMAT_R16G16B16A16_USCALED; break;
            case VK_FORMAT_R16G16B16_SSCALED: f = VK_FORMAT_R16G16B16A16_SSCALED; break;
            case VK_FORMAT_R16G16B16_SFLOAT: f = VK_FORMAT_R16G16B16A16_SFLOAT; break;
            default: VK_LOG_ONCE("vk: vertex format %d unsupported", (int)f); break;
            }
        }
        return f;
    }
    if (size_bytes) *size_bytes = 0;
    if (size) VK_LOG_ONCE("vk: unknown vertex attribute size 0x%x type %u", size, type);
    return VK_FORMAT_UNDEFINED;
}

}  // namespace gpu::vk
