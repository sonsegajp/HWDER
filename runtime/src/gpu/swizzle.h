// Maxwell block-linear surface layout. A GOB is 64 bytes x 8 rows (512 bytes); a block is
// 1 GOB wide, 2^block_height GOBs tall and 2^block_depth GOBs deep; blocks are laid out row-major.
#pragma once
#include <cstring>

#include "hwder/runtime.h"

namespace gpu {

// Byte offset inside a GOB of byte column x (0..63) and row y (0..7).
inline u32 gob_offset(u32 x, u32 y) {
    return ((x & 63) >> 5) * 256 + ((y & 7) >> 1) * 64 + ((x & 31) >> 4) * 32 + (y & 1) * 16 + (x & 15);
}

struct BlockLinear {
    u32 width_bytes, height, depth;  // surface size (width in bytes)
    u32 block_height_log2, block_depth_log2;

    u32 gobs_per_row() const { return (width_bytes + 63) / 64; }
    u32 blocks_high() const {
        u32 bh = 1u << block_height_log2;
        return ((height + 7) / 8 + bh - 1) / bh;
    }
    u64 block_size() const { return 512ull << (block_height_log2 + block_depth_log2); }
    u64 slice_size() const {  // bytes per (block depth) slab of blocks
        return (u64)gobs_per_row() * blocks_high() * block_size();
    }
    u64 size() const {
        u32 bd = 1u << block_depth_log2;
        return slice_size() * ((depth + bd - 1) / bd);
    }
    u64 offset(u32 x, u32 y, u32 z = 0) const {
        u32 bh = 1u << block_height_log2, bd = 1u << block_depth_log2;
        u32 gob_y = y >> 3;
        u64 block = ((u64)(z / bd) * blocks_high() + gob_y / bh) * gobs_per_row() + (x >> 6);
        return block * block_size() + ((u64)(z % bd) * bh + gob_y % bh) * 512 + gob_offset(x, y);
    }
};

// Copy between a block-linear surface and a pitch-linear buffer. Rows of 16 bytes are contiguous in
// both layouts, so the inner loop moves 16-byte chunks when aligned.
inline void deswizzle(const BlockLinear& bl, const u8* src, u8* dst, u32 pitch, u32 x0_bytes, u32 y0, u32 w_bytes,
                      u32 h, u32 z = 0) {
    for (u32 y = 0; y < h; y++) {
        u8* row = dst + (u64)y * pitch;
        u32 x = 0;
        while (x < w_bytes) {
            u32 sx = x0_bytes + x;
            u32 n = 16 - (sx & 15);
            if (n > w_bytes - x) n = w_bytes - x;
            memcpy(row + x, src + bl.offset(sx, y0 + y, z), n);
            x += n;
        }
    }
}
inline void swizzle(const BlockLinear& bl, const u8* src, u8* dst, u32 pitch, u32 x0_bytes, u32 y0, u32 w_bytes,
                    u32 h, u32 z = 0) {
    for (u32 y = 0; y < h; y++) {
        const u8* row = src + (u64)y * pitch;
        u32 x = 0;
        while (x < w_bytes) {
            u32 dx = x0_bytes + x;
            u32 n = 16 - (dx & 15);
            if (n > w_bytes - x) n = w_bytes - x;
            memcpy(dst + bl.offset(dx, y0 + y, z), row + x, n);
            x += n;
        }
    }
}

}  // namespace gpu
