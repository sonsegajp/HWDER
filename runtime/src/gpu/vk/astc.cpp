// ASTC (LDR profile) decoder. The GTX 1660 Ti has no ASTC support, so textures are decoded to RGBA8 on
// upload. Follows the Khronos Data Format Specification (ASTC chapter): block mode, integer sequence
// encoding, partition hash, colour endpoint modes 0-13 (LDR) and weight infill.
#include <thread>
#include <vector>

#include "vk_common.h"

namespace gpu::vk {

namespace {

struct Block {
    u8 b[16];
    u64 lo = 0, hi = 0;  // the 128 bits as two words (little endian)
    void load() {
        memcpy(&lo, b, 8);
        memcpy(&hi, b + 8, 8);
    }
    u32 bits(u32 pos, u32 n) const {  // n <= 32, pos + n <= 128
        if (!n) return 0;
        u64 v;
        if (pos < 64) {
            v = lo >> pos;
            if (pos + n > 64) v |= hi << (64 - pos);
        } else {
            v = hi >> (pos - 64);
        }
        return (u32)(v & ((n >= 32) ? 0xFFFFFFFFull : ((1ull << n) - 1)));
    }
};

// Quantization levels: {range, bits, trits, quints}
struct Quant {
    u32 range, bits, trits, quints;
};
const Quant kQuants[] = {{2, 1, 0, 0},   {3, 0, 1, 0},   {4, 2, 0, 0},   {5, 0, 0, 1},   {6, 1, 1, 0},   {8, 3, 0, 0},
                         {10, 1, 0, 1},  {12, 2, 1, 0},  {16, 4, 0, 0},  {20, 2, 0, 1},  {24, 3, 1, 0},  {32, 5, 0, 0},
                         {40, 3, 0, 1},  {48, 4, 1, 0},  {64, 6, 0, 0},  {80, 4, 0, 1},  {96, 5, 1, 0},  {128, 7, 0, 0},
                         {160, 5, 0, 1}, {192, 6, 1, 0}, {256, 8, 0, 0}};

u32 ise_bits(const Quant& q, u32 n) { return n * q.bits + (n * 8 * q.trits + 4) / 5 + (n * 7 * q.quints + 2) / 3; }

// Decode n integers; out[i] = {value bits (m), trit/quint digit}
struct IseValue {
    u32 m, d;
};
void decode_ise(const Block& blk, u32 pos, const Quant& q, u32 n, IseValue* out) {
    if (q.trits) {
        for (u32 base = 0; base < n; base += 5) {
            u32 cnt = std::min(5u, n - base);
            u32 m[5] = {}, T = 0;
            const u32 tw[5] = {2, 2, 1, 2, 1};
            u32 tshift = 0;
            for (u32 i = 0; i < cnt; i++) {
                m[i] = blk.bits(pos, q.bits);
                pos += q.bits;
                T |= blk.bits(pos, tw[i]) << tshift;
                pos += tw[i];
                tshift += tw[i];
            }
            u32 t[5], C;
            if (((T >> 2) & 7) == 7) {
                C = (((T >> 5) & 7) << 2) | 3;
                t[4] = t[3] = 2;
            } else {
                C = T & 0x1F;
                if (((T >> 5) & 3) == 3) {
                    t[4] = 2;
                    t[3] = (T >> 7) & 1;
                } else {
                    t[4] = (T >> 7) & 1;
                    t[3] = (T >> 5) & 3;
                }
            }
            if ((C & 3) == 3) {
                t[2] = 2;
                t[1] = (C >> 4) & 1;
                t[0] = ((C >> 3) & 1) << 1 | (((C >> 2) & 1) & ~((C >> 3) & 1));
            } else if (((C >> 2) & 3) == 3) {
                t[2] = 2;
                t[1] = 2;
                t[0] = C & 3;
            } else {
                t[2] = (C >> 4) & 1;
                t[1] = (C >> 2) & 3;
                t[0] = ((C >> 1) & 1) << 1 | ((C & 1) & ~((C >> 1) & 1));
            }
            for (u32 i = 0; i < cnt; i++) out[base + i] = {m[i], t[i]};
        }
    } else if (q.quints) {
        for (u32 base = 0; base < n; base += 3) {
            u32 cnt = std::min(3u, n - base);
            u32 m[3] = {}, Q = 0;
            const u32 qw[3] = {3, 2, 2};
            u32 qshift = 0;
            for (u32 i = 0; i < cnt; i++) {
                m[i] = blk.bits(pos, q.bits);
                pos += q.bits;
                Q |= blk.bits(pos, qw[i]) << qshift;
                pos += qw[i];
                qshift += qw[i];
            }
            u32 qv[3], C;
            if (((Q >> 1) & 3) == 3 && ((Q >> 5) & 3) == 0) {
                u32 q0 = Q & 1;
                qv[2] = (q0 << 2) | ((((Q >> 4) & 1) & ~q0) << 1) | (((Q >> 3) & 1) & ~q0);
                qv[1] = 4;
                qv[0] = 4;
            } else {
                if (((Q >> 1) & 3) == 3) {
                    qv[2] = 4;
                    C = (((Q >> 3) & 3) << 3) | ((~(Q >> 5) & 3) << 1) | (Q & 1);
                } else {
                    qv[2] = (Q >> 5) & 3;
                    C = Q & 0x1F;
                }
                if ((C & 7) == 5) {
                    qv[1] = 4;
                    qv[0] = (C >> 3) & 3;
                } else {
                    qv[1] = (C >> 3) & 3;
                    qv[0] = C & 7;
                }
            }
            for (u32 i = 0; i < cnt; i++) out[base + i] = {m[i], qv[i]};
        }
    } else {
        for (u32 i = 0; i < n; i++) {
            out[i] = {blk.bits(pos, q.bits), 0};
            pos += q.bits;
        }
    }
}

// Unquantize a weight to 0..64.
u32 unquant_weight(const Quant& q, IseValue v) {
    u32 m = v.m, r;
    if (!q.trits && !q.quints) {
        switch (q.bits) {
        case 1: r = m ? 63 : 0; break;
        case 2: r = m * 21; break;
        case 3: r = m * 9; break;
        case 4: r = (m << 2) | (m >> 2); break;
        default: r = (m << 1) | (m >> 4); break;
        }
    } else {
        u32 a = m & 1, b = (m >> 1) & 1, c = (m >> 2) & 1;
        u32 A = a ? 0x7F : 0, B = 0, C;
        if (q.trits) {
            switch (q.bits) {
            case 0: C = 32; break;  // range 3: D * 32
            case 1: C = 50; break;
            case 2: C = 23; B = (b << 6) | (b << 2) | b; break;
            default: C = 11; B = (c << 6) | (b << 5) | (c << 1) | b; break;
            }
        } else {
            switch (q.bits) {
            case 0: C = 16; break;  // range 5: D * 16
            case 1: C = 28; break;
            default: C = 13; B = (b << 6) | (b << 1); break;
            }
        }
        if (q.bits == 0) {
            r = v.d * C;
        } else {
            u32 T = v.d * C + B;
            T ^= A;
            r = (A & 0x20) | (T >> 2);
        }
    }
    if (r > 32) r += 1;
    return r;
}

// Unquantize a colour endpoint integer to 0..255.
u32 unquant_color(const Quant& q, IseValue v) {
    u32 m = v.m;
    if (!q.trits && !q.quints) {
        switch (q.bits) {
        case 1: return m ? 255 : 0;
        case 2: return m * 85;
        case 3: return (m << 5) | (m << 2) | (m >> 1);
        case 4: return m * 17;
        case 5: return (m << 3) | (m >> 2);
        case 6: return (m << 2) | (m >> 4);
        case 7: return (m << 1) | (m >> 6);
        default: return m;
        }
    }
    u32 a = m & 1, b = (m >> 1) & 1, c = (m >> 2) & 1, d = (m >> 3) & 1, e = (m >> 4) & 1, f = (m >> 5) & 1;
    u32 A = a ? 0x1FF : 0, B = 0, C;
    if (q.trits) {
        switch (q.bits) {
        case 0: return v.d * 127;  // range 3 -> 0,127,254 (spec: 0, 128, 255 nominal)
        case 1: C = 204; break;
        case 2: C = 93; B = (b << 8) | (b << 4) | (b << 2) | (b << 1); break;
        case 3: C = 44; B = (c << 8) | (b << 7) | (c << 3) | (b << 2) | (c << 1) | b; break;
        case 4: C = 22; B = (d << 8) | (c << 7) | (b << 6) | (d << 2) | (c << 1) | b; break;
        case 5: C = 11; B = (e << 8) | (d << 7) | (c << 6) | (b << 5) | (e << 1) | d; break;
        default: C = 5; B = (f << 8) | (e << 7) | (d << 6) | (c << 5) | (b << 4) | f; break;
        }
    } else {
        switch (q.bits) {
        case 0: return v.d * 64;  // range 5
        case 1: C = 113; break;
        case 2: C = 54; B = (b << 8) | (b << 3) | (b << 2); break;
        case 3: C = 26; B = (c << 8) | (b << 7) | (c << 2) | (b << 1) | c; break;
        case 4: C = 13; B = (d << 8) | (c << 7) | (b << 6) | (d << 1) | c; break;
        default: C = 6; B = (e << 8) | (d << 7) | (c << 6) | (b << 5) | e; break;
        }
    }
    u32 T = v.d * C + B;
    T ^= A;
    return (A & 0x80) | (T >> 2);
}

u32 hash52(u32 p) {
    p ^= p >> 15;
    p -= p << 17;
    p += p << 7;
    p += p << 4;
    p ^= p >> 5;
    p += p << 16;
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    return p;
}

// Partition function split into a per-block setup and a cheap per-texel evaluation.
struct PartitionFn {
    u32 count = 1;
    bool small_block = false;
    u32 s[13] = {};
    u32 rnum = 0;
    void init(u32 seed, u32 cnt, bool small) {
        count = cnt;
        small_block = small;
        if (count <= 1) return;
        seed += (count - 1) * 1024;
        rnum = hash52(seed);
        for (u32 i = 1; i <= 12; i++) {
            u32 v = (rnum >> ((i - 1) * 4)) & 0xF;
            s[i] = v * v;
        }
        u32 sh1, sh2;
        if (seed & 2) {
            sh1 = (seed & 1) ? 4 : 5;
            sh2 = count == 3 ? 6 : 5;
        } else {
            sh1 = count == 3 ? 6 : 5;
            sh2 = (seed & 1) ? 4 : 5;
        }
        u32 sh3 = (seed & 0x10) ? sh1 : sh2;
        s[1] >>= sh1; s[2] >>= sh2; s[3] >>= sh1; s[4] >>= sh2; s[5] >>= sh1; s[6] >>= sh2; s[7] >>= sh1; s[8] >>= sh2;
        s[9] >>= sh3; s[10] >>= sh3; s[11] >>= sh3; s[12] >>= sh3;
    }
    u32 at(u32 x, u32 y, u32 z) const {
        if (count <= 1) return 0;
        if (small_block) x <<= 1, y <<= 1, z <<= 1;
        u32 a = (s[1] * x + s[2] * y + s[11] * z + (rnum >> 14)) & 0x3F;
        u32 b = (s[3] * x + s[4] * y + s[12] * z + (rnum >> 10)) & 0x3F;
        u32 c = (s[5] * x + s[6] * y + s[9] * z + (rnum >> 6)) & 0x3F;
        u32 d = (s[7] * x + s[8] * y + s[10] * z + (rnum >> 2)) & 0x3F;
        if (count < 4) d = 0;
        if (count < 3) c = 0;
        if (a >= b && a >= c && a >= d) return 0;
        if (b >= c && b >= d) return 1;
        if (c >= d) return 2;
        return 3;
    }
};

inline s32 clamp255(s32 v) { return v < 0 ? 0 : v > 255 ? 255 : v; }
void bit_transfer_signed(s32& a, s32& b) {
    b >>= 1;
    b |= a & 0x80;
    a >>= 1;
    a &= 0x3F;
    if (a & 0x20) a -= 0x40;
}
void blue_contract(s32* c) {
    c[0] = (c[0] + c[2]) >> 1;
    c[1] = (c[1] + c[2]) >> 1;
}

// Endpoints e0/e1 (RGBA, 0..255) from unquantized colour integers v for colour endpoint mode cem.
bool decode_endpoints(u32 cem, const s32* v, s32* e0, s32* e1) {
    s32 a[8];
    memcpy(a, v, sizeof(a));
    switch (cem) {
    case 0:
        e0[0] = e0[1] = e0[2] = a[0]; e0[3] = 255;
        e1[0] = e1[1] = e1[2] = a[1]; e1[3] = 255;
        return true;
    case 1: {
        s32 l0 = (a[0] >> 2) | (a[1] & 0xC0), l1 = std::min(255, l0 + (a[1] & 0x3F));
        e0[0] = e0[1] = e0[2] = l0; e0[3] = 255;
        e1[0] = e1[1] = e1[2] = l1; e1[3] = 255;
        return true;
    }
    case 4:
        e0[0] = e0[1] = e0[2] = a[0]; e0[3] = a[1];
        e1[0] = e1[1] = e1[2] = a[2]; e1[3] = a[3];
        return true;
    case 5:
        bit_transfer_signed(a[1], a[0]);
        bit_transfer_signed(a[3], a[2]);
        e0[0] = e0[1] = e0[2] = a[0]; e0[3] = a[2];
        e1[0] = e1[1] = e1[2] = clamp255(a[0] + a[1]); e1[3] = clamp255(a[2] + a[3]);
        return true;
    case 6:
        e1[0] = a[0]; e1[1] = a[1]; e1[2] = a[2]; e1[3] = 255;
        e0[0] = (a[0] * a[3]) >> 8; e0[1] = (a[1] * a[3]) >> 8; e0[2] = (a[2] * a[3]) >> 8; e0[3] = 255;
        return true;
    case 8: case 12: {
        bool alpha = cem == 12;
        s32 s0 = a[0] + a[2] + a[4], s1 = a[1] + a[3] + a[5];
        s32 c0[4] = {a[0], a[2], a[4], alpha ? a[6] : 255}, c1[4] = {a[1], a[3], a[5], alpha ? a[7] : 255};
        if (s1 >= s0) {
            memcpy(e0, c0, 16);
            memcpy(e1, c1, 16);
        } else {
            blue_contract(c0);
            blue_contract(c1);
            memcpy(e0, c1, 16);
            memcpy(e1, c0, 16);
        }
        return true;
    }
    case 9: case 13: {
        bool alpha = cem == 13;
        bit_transfer_signed(a[1], a[0]);
        bit_transfer_signed(a[3], a[2]);
        bit_transfer_signed(a[5], a[4]);
        if (alpha) bit_transfer_signed(a[7], a[6]);
        s32 c0[4] = {a[0], a[2], a[4], alpha ? a[6] : 255};
        s32 c1[4] = {a[0] + a[1], a[2] + a[3], a[4] + a[5], alpha ? a[6] + a[7] : 255};
        if (a[1] + a[3] + a[5] >= 0) {
            memcpy(e0, c0, 16);
            memcpy(e1, c1, 16);
        } else {
            blue_contract(c0);
            blue_contract(c1);
            memcpy(e0, c1, 16);
            memcpy(e1, c0, 16);
        }
        for (int i = 0; i < 4; i++) e0[i] = clamp255(e0[i]), e1[i] = clamp255(e1[i]);
        return true;
    }
    case 10:
        e1[0] = a[0]; e1[1] = a[1]; e1[2] = a[2]; e1[3] = a[5];
        e0[0] = (a[0] * a[3]) >> 8; e0[1] = (a[1] * a[3]) >> 8; e0[2] = (a[2] * a[3]) >> 8; e0[3] = a[4];
        return true;
    default:
        return false;  // HDR modes
    }
}

void fill_error(u8* px, u32 n) {
    for (u32 i = 0; i < n; i++) {
        px[i * 4 + 0] = 255;
        px[i * 4 + 1] = 0;
        px[i * 4 + 2] = 255;
        px[i * 4 + 3] = 255;
    }
}

// Decode one 2D block of bw x bh texels into px (RGBA8, row major).
void decode_block(const u8* src, u32 bw, u32 bh, u8* px) {
    Block blk;
    memcpy(blk.b, src, 16);
    blk.load();
    u32 texels = bw * bh;
    u32 mode = blk.bits(0, 11);
    if ((mode & 0x1FF) == 0x1FC) {  // void extent
        for (u32 i = 0; i < texels; i++)
            for (u32 c = 0; c < 4; c++) px[i * 4 + c] = (u8)(blk.bits(64 + c * 16, 16) >> 8);
        return;
    }
    // ---- block mode
    u32 D = (mode >> 10) & 1, H = (mode >> 9) & 1, R = (mode >> 4) & 1;
    u32 A = (mode >> 5) & 3, N, M;
    if (mode & 3) {
        R |= (mode & 3) << 1;
        u32 B = (mode >> 7) & 3;
        switch ((mode >> 2) & 3) {
        case 0: N = B + 4; M = A + 2; break;
        case 1: N = B + 8; M = A + 2; break;
        case 2: N = A + 2; M = B + 8; break;
        default:
            B &= 1;
            if (mode & 0x100) N = B + 2, M = A + 2;
            else N = A + 2, M = B + 6;
            break;
        }
    } else {
        R |= ((mode >> 2) & 3) << 1;
        switch ((mode >> 7) & 3) {
        case 0: N = 12; M = A + 2; break;
        case 1: N = A + 2; M = 12; break;
        case 2: N = A + 6; M = ((mode >> 9) & 3) + 6; D = 0; H = 0; break;
        default:
            if (mode & 0x20) return fill_error(px, texels);  // reserved
            if ((mode >> 5) & 1) N = 10, M = 6;
            else N = 6, M = 10;
            break;
        }
    }
    if (R < 2) return fill_error(px, texels);
    static const u32 kWeightRange[2][8] = {{0, 0, 2, 3, 4, 5, 6, 8}, {0, 0, 10, 12, 16, 20, 24, 32}};
    u32 wrange = kWeightRange[H][R];
    const Quant* wq = nullptr;
    for (const Quant& q : kQuants)
        if (q.range == wrange) wq = &q;
    if (!wq || N > bw || M > bh) return fill_error(px, texels);
    u32 nweights = N * M * (D ? 2 : 1);
    u32 weight_bits = ise_bits(*wq, nweights);
    if (nweights > 64 || weight_bits < 24 || weight_bits > 96) return fill_error(px, texels);

    // ---- partitions and colour endpoint modes
    u32 parts = ((mode >> 11) & 3) + 1;
    u32 parts_field = blk.bits(11, 2) + 1;
    parts = parts_field;
    u32 seed = 0, cem[4] = {}, color_start, below_weights = 128 - weight_bits;
    if (parts == 1) {
        cem[0] = blk.bits(13, 4);
        color_start = 17;
    } else {
        seed = blk.bits(13, 10);
        u32 enc = blk.bits(23, 6);
        u32 extra = 3 * parts - 4;
        below_weights -= extra;
        enc |= blk.bits(below_weights, extra) << 6;
        color_start = 29;
        if ((enc & 3) == 0) {
            for (u32 i = 0; i < parts; i++) cem[i] = enc >> 2;
        } else {
            u32 base = (enc & 3) - 1;
            u32 pos = 2;
            u32 cls[4];
            for (u32 i = 0; i < parts; i++) cls[i] = base + ((enc >> pos++) & 1);
            for (u32 i = 0; i < parts; i++) {
                u32 m = (enc >> pos) & 3;
                pos += 2;
                cem[i] = (cls[i] << 2) | m;
            }
        }
    }
    u32 ccs = 0;
    if (D) {
        below_weights -= 2;
        ccs = blk.bits(below_weights, 2);
    }
    if (D && parts == 4) return fill_error(px, texels);
    u32 nints = 0;
    for (u32 i = 0; i < parts; i++) nints += ((cem[i] >> 2) + 1) * 2;
    if (below_weights < color_start) return fill_error(px, texels);
    u32 color_bits = below_weights - color_start;
    const Quant* cq = nullptr;
    for (int i = (int)(sizeof(kQuants) / sizeof(kQuants[0])) - 1; i >= 0; i--) {
        if (kQuants[i].range < 6) break;
        if (ise_bits(kQuants[i], nints) <= color_bits) {
            cq = &kQuants[i];
            break;
        }
    }
    if (!cq) return fill_error(px, texels);
    IseValue cv[18];
    decode_ise(blk, color_start, *cq, nints, cv);
    s32 e0[4][4], e1[4][4];
    u32 vi = 0;
    for (u32 p = 0; p < parts; p++) {
        s32 v[8] = {};
        u32 n = ((cem[p] >> 2) + 1) * 2;
        for (u32 i = 0; i < n; i++) v[i] = (s32)unquant_color(*cq, cv[vi++]);
        if (!decode_endpoints(cem[p], v, e0[p], e1[p])) return fill_error(px, texels);
    }

    // ---- weights (stored bit-reversed from the top of the block)
    Block rev;
    for (u32 i = 0; i < 16; i++) {
        u8 b = blk.b[15 - i];
        b = (u8)(((b * 0x0802u & 0x22110u) | (b * 0x8020u & 0x88440u)) * 0x10101u >> 16);
        rev.b[i] = b;
    }
    rev.load();
    IseValue wv[64];
    decode_ise(rev, 0, *wq, nweights, wv);
    u32 w[64];
    for (u32 i = 0; i < nweights; i++) w[i] = unquant_weight(*wq, wv[i]);

    // ---- infill + interpolate
    u32 Ds = (1024 + bw / 2) / (bw - 1), Dt = (1024 + bh / 2) / (bh - 1);
    PartitionFn part;
    part.init(seed, parts, texels < 31);
    u32 planes = D ? 2 : 1;
    for (u32 y = 0; y < bh; y++) {
        for (u32 x = 0; x < bw; x++) {
            u32 cs = Ds * x, ct = Dt * y;
            u32 gs = (cs * (N - 1) + 32) >> 6, gt = (ct * (M - 1) + 32) >> 6;
            u32 js = gs >> 4, fs = gs & 0xF, jt = gt >> 4, ft = gt & 0xF;
            u32 w11 = (fs * ft + 8) >> 4, w10 = ft - w11, w01 = fs - w11, w00 = 16 - fs - ft + w11;
            u32 wgt[2];
            for (u32 pl = 0; pl < planes; pl++) {
                auto at = [&](u32 jx, u32 jy) -> u32 {
                    if (jx >= N) jx = N - 1;
                    if (jy >= M) jy = M - 1;
                    return w[(jy * N + jx) * planes + pl];
                };
                wgt[pl] = (at(js, jt) * w00 + at(js + 1, jt) * w01 + at(js, jt + 1) * w10 + at(js + 1, jt + 1) * w11 + 8) >> 4;
            }
            u32 p = part.at(x, y, 0);
            u8* o = px + (y * bw + x) * 4;
            for (u32 c = 0; c < 4; c++) {
                u32 wc = (D && c == ccs) ? wgt[1] : wgt[0];
                u32 a0 = (u32)e0[p][c] * 257, a1 = (u32)e1[p][c] * 257;
                u32 v = (a0 * (64 - wc) + a1 * wc + 32) >> 6;
                o[c] = (u8)(v >> 8);
            }
        }
    }
}

}  // namespace

void decode_astc(const u8* src, u32 width, u32 height, u32 depth, u32 bw, u32 bh, u8* dst) {
    if (bw < 4 || bh < 4 || bw > 12 || bh > 12) {
        VK_LOG_ONCE("vk: ASTC block %ux%u not supported", bw, bh);
        fill_error(dst, width * height * depth);
        return;
    }
    u32 blocks_w = (width + bw - 1) / bw, blocks_h = (height + bh - 1) / bh;
    auto rows = [&](u32 z, u32 by0, u32 by1) {
        u8 px[12 * 12 * 4];
        const u8* slice = src + (u64)z * blocks_w * blocks_h * 16;
        u8* out = dst + (u64)z * width * height * 4;
        for (u32 by = by0; by < by1; by++) {
            for (u32 bx = 0; bx < blocks_w; bx++) {
                decode_block(slice + ((u64)by * blocks_w + bx) * 16, bw, bh, px);
                u32 x0 = bx * bw, y0 = by * bh;
                u32 cw = std::min(bw, width - x0), ch = std::min(bh, height - y0);
                for (u32 y = 0; y < ch; y++) memcpy(out + ((u64)(y0 + y) * width + x0) * 4, px + y * bw * 4, (size_t)cw * 4);
            }
        }
    };
    // Large images are decoded by several threads (block rows are independent).
    u32 threads = std::min<u32>(std::max(1u, std::thread::hardware_concurrency()), 8);
    if ((u64)blocks_w * blocks_h * depth < 512) threads = 1;
    for (u32 z = 0; z < depth; z++) {
        if (threads == 1 || blocks_h < threads) {
            rows(z, 0, blocks_h);
            continue;
        }
        std::vector<std::thread> pool;
        for (u32 t = 0; t < threads; t++)
            pool.emplace_back([&, t] { rows(z, blocks_h * t / threads, blocks_h * (t + 1) / threads); });
        for (auto& th : pool) th.join();
    }
}

}  // namespace gpu::vk
