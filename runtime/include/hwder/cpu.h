// HWDER - guest CPU state and the primitive operations used by recompiled code.
//
// Guest memory is identity mapped: a guest virtual address is a valid host pointer.
// Every recompiled function has the signature `void f(Ctx* c)`.
#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef union V128 {
    u8 b[16];
    u16 h[8];
    u32 s[4];
    u64 d[2];
    s8 sb[16];
    s16 sh[8];
    s32 ss[4];
    s64 sd[2];
    float f[4];
    double fd[2];
} V128;

typedef struct Ctx {
    u64 x[32];  // x[31] unused (ZR/SP are handled by the translator)
    u64 sp;
    V128 v[32];
    u8 nf, zf, cf, vf;
    u32 fpcr, fpsr;
    u64 tpidr_el0, tpidrro_el0;
    u64 excl_addr, excl_val, excl_val2;
    u64 pc;        // branch target for hw_dispatch (indirect jumps)
    void* thread;  // runtime-owned per-thread data
} Ctx;

#define HW_INLINE static inline __attribute__((always_inline))

#if defined(__clang__) || defined(__GNUC__)
#define HW_MUSTTAIL __attribute__((musttail))
#else
#define HW_MUSTTAIL
#endif

static const V128 V_ZERO_ = {{0}};
#define V_ZERO (V_ZERO_)
HW_INLINE V128 V_LO(V128 v) { v.d[1] = 0; return v; }

// ------------------------------------------------------------------ memory
typedef u16 __attribute__((aligned(1), may_alias)) u16u;
typedef u32 __attribute__((aligned(1), may_alias)) u32u;
typedef u64 __attribute__((aligned(1), may_alias)) u64u;
typedef s16 __attribute__((aligned(1), may_alias)) s16u;
typedef s32 __attribute__((aligned(1), may_alias)) s32u;

#define HW_PTR(a) ((uintptr_t)(a))
#define LD8(a) (*(const u8*)HW_PTR(a))
#define LD16(a) (*(const u16u*)HW_PTR(a))
#define LD32(a) (*(const u32u*)HW_PTR(a))
#define LD64(a) (*(const u64u*)HW_PTR(a))
#define LDS8(a) (*(const s8*)HW_PTR(a))
#define LDS16(a) (*(const s16u*)HW_PTR(a))
#define LDS32(a) (*(const s32u*)HW_PTR(a))
#define ST8(a, v) (*(u8*)HW_PTR(a) = (u8)(v))
#define ST16(a, v) (*(u16u*)HW_PTR(a) = (u16)(v))
#define ST32(a, v) (*(u32u*)HW_PTR(a) = (u32)(v))
#define ST64(a, v) (*(u64u*)HW_PTR(a) = (u64)(v))
HW_INLINE V128 LD128(u64 a) { V128 r; memcpy(&r, (const void*)HW_PTR(a), 16); return r; }
HW_INLINE void ST128(u64 a, V128 v) { memcpy((void*)HW_PTR(a), &v, 16); }

// ------------------------------------------------------------------ atomics
#define ATOMIC_LOAD8(a) __atomic_load_n((u8*)HW_PTR(a), __ATOMIC_ACQUIRE)
#define ATOMIC_LOAD16(a) __atomic_load_n((u16*)HW_PTR(a), __ATOMIC_ACQUIRE)
#define ATOMIC_LOAD32(a) __atomic_load_n((u32*)HW_PTR(a), __ATOMIC_ACQUIRE)
#define ATOMIC_LOAD64(a) __atomic_load_n((u64*)HW_PTR(a), __ATOMIC_ACQUIRE)
#define ATOMIC_STORE8(a, v) __atomic_store_n((u8*)HW_PTR(a), (u8)(v), __ATOMIC_RELEASE)
#define ATOMIC_STORE16(a, v) __atomic_store_n((u16*)HW_PTR(a), (u16)(v), __ATOMIC_RELEASE)
#define ATOMIC_STORE32(a, v) __atomic_store_n((u32*)HW_PTR(a), (u32)(v), __ATOMIC_RELEASE)
#define ATOMIC_STORE64(a, v) __atomic_store_n((u64*)HW_PTR(a), (u64)(v), __ATOMIC_RELEASE)
#define HW_FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)

// Exclusive monitor emulation: LDXR records (addr, value); STXR succeeds iff the
// location still holds that value (compare-and-swap). Returns 1 on success.
#define HW_EXCL_STORE(bits, T)                                                         \
    HW_INLINE int EXCL_STORE##bits(Ctx* c, u64 a, u64 v) {                              \
        if (c->excl_addr != a) return 0;                                               \
        c->excl_addr = ~(u64)0;                                                        \
        T expected = (T)c->excl_val;                                                   \
        return __atomic_compare_exchange_n((T*)HW_PTR(a), &expected, (T)v, 0,          \
                                           __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);        \
    }
HW_EXCL_STORE(8, u8)
HW_EXCL_STORE(16, u16)
HW_EXCL_STORE(32, u32)
HW_EXCL_STORE(64, u64)

HW_INLINE void EXCL_LOADP32(Ctx* c, u64 a, u64* t1, u64* t2) {
    u64 v = __atomic_load_n((u64*)HW_PTR(a), __ATOMIC_ACQUIRE);
    c->excl_addr = a; c->excl_val = v;
    *t1 = (u32)v; *t2 = (u32)(v >> 32);
}
HW_INLINE int EXCL_STOREP32(Ctx* c, u64 a, u64 v1, u64 v2) {
    if (c->excl_addr != a) return 0;
    c->excl_addr = ~(u64)0;
    u64 expected = c->excl_val;
    return __atomic_compare_exchange_n((u64*)HW_PTR(a), &expected, (u64)(u32)v1 | ((u64)v2 << 32), 0,
                                       __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}
HW_INLINE void EXCL_LOADP64(Ctx* c, u64 a, u64* t1, u64* t2) {
    unsigned __int128 v = __atomic_load_n((unsigned __int128*)HW_PTR(a), __ATOMIC_ACQUIRE);
    c->excl_addr = a; c->excl_val = (u64)v; c->excl_val2 = (u64)(v >> 64);
    *t1 = (u64)v; *t2 = (u64)(v >> 64);
}
HW_INLINE int EXCL_STOREP64(Ctx* c, u64 a, u64 v1, u64 v2) {
    if (c->excl_addr != a) return 0;
    c->excl_addr = ~(u64)0;
    unsigned __int128 expected = (unsigned __int128)c->excl_val | ((unsigned __int128)c->excl_val2 << 64);
    unsigned __int128 nv = (unsigned __int128)v1 | ((unsigned __int128)v2 << 64);
    return __atomic_compare_exchange_n((unsigned __int128*)HW_PTR(a), &expected, nv, 0,
                                       __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

// ------------------------------------------------------------------ flags / integer ops
HW_INLINE void SETNZCV(Ctx* c, u32 nzcv) {
    c->nf = (nzcv >> 3) & 1; c->zf = (nzcv >> 2) & 1; c->cf = (nzcv >> 1) & 1; c->vf = nzcv & 1;
}

HW_INLINE u64 ADC64(Ctx* c, u64 a, u64 b, u32 cin) {
    u64 r1, r;
    u8 c1 = __builtin_add_overflow(a, b, &r1);
    u8 c2 = __builtin_add_overflow(r1, (u64)cin, &r);
    s64 sr;
    u8 v1 = __builtin_add_overflow((s64)a, (s64)b, &sr);
    u8 v2 = __builtin_add_overflow(sr, (s64)cin, &sr);
    c->nf = (u8)(r >> 63); c->zf = r == 0; c->cf = c1 | c2; c->vf = v1 ^ v2;
    return r;
}
HW_INLINE u32 ADC32(Ctx* c, u32 a, u32 b, u32 cin) {
    u64 wide = (u64)a + (u64)b + cin;
    s64 swide = (s64)(s32)a + (s64)(s32)b + cin;
    u32 r = (u32)wide;
    c->nf = (u8)(r >> 31); c->zf = r == 0; c->cf = (wide >> 32) != 0; c->vf = (s64)(s32)r != swide;
    return r;
}
HW_INLINE u64 ADDS64(Ctx* c, u64 a, u64 b) {
    u64 r; s64 sr;
    c->cf = __builtin_add_overflow(a, b, &r);
    c->vf = __builtin_add_overflow((s64)a, (s64)b, &sr);
    c->nf = (u8)(r >> 63); c->zf = r == 0;
    return r;
}
HW_INLINE u32 ADDS32(Ctx* c, u32 a, u32 b) {
    u32 r; s32 sr;
    c->cf = __builtin_add_overflow(a, b, &r);
    c->vf = __builtin_add_overflow((s32)a, (s32)b, &sr);
    c->nf = (u8)(r >> 31); c->zf = r == 0;
    return r;
}
HW_INLINE u64 SUBS64(Ctx* c, u64 a, u64 b) {
    u64 r = a - b; s64 sr;
    c->cf = a >= b;
    c->vf = __builtin_sub_overflow((s64)a, (s64)b, &sr);
    c->nf = (u8)(r >> 63); c->zf = r == 0;
    return r;
}
HW_INLINE u32 SUBS32(Ctx* c, u32 a, u32 b) {
    u32 r = a - b; s32 sr;
    c->cf = a >= b;
    c->vf = __builtin_sub_overflow((s32)a, (s32)b, &sr);
    c->nf = (u8)(r >> 31); c->zf = r == 0;
    return r;
}
HW_INLINE u64 LOGICFLAGS64(Ctx* c, u64 r) { c->nf = (u8)(r >> 63); c->zf = r == 0; c->cf = 0; c->vf = 0; return r; }
HW_INLINE u32 LOGICFLAGS32(Ctx* c, u32 r) { c->nf = (u8)(r >> 31); c->zf = r == 0; c->cf = 0; c->vf = 0; return r; }

HW_INLINE u64 ROR64(u64 v, u32 n) { n &= 63; return n ? (v >> n) | (v << (64 - n)) : v; }
HW_INLINE u32 ROR32(u32 v, u32 n) { n &= 31; return n ? (v >> n) | (v << (32 - n)) : v; }
HW_INLINE u64 UDIV64(u64 a, u64 b) { return b ? a / b : 0; }
HW_INLINE u32 UDIV32(u32 a, u32 b) { return b ? a / b : 0; }
HW_INLINE u64 SDIV64(u64 a, u64 b) {
    if (!b) return 0;
    if ((s64)a == INT64_MIN && (s64)b == -1) return a;
    return (u64)((s64)a / (s64)b);
}
HW_INLINE u32 SDIV32(u32 a, u32 b) {
    if (!b) return 0;
    if ((s32)a == INT32_MIN && (s32)b == -1) return a;
    return (u32)((s32)a / (s32)b);
}
HW_INLINE u64 SMULH(u64 a, u64 b) { return (u64)(((__int128)(s64)a * (__int128)(s64)b) >> 64); }
HW_INLINE u64 UMULH(u64 a, u64 b) { return (u64)(((unsigned __int128)a * (unsigned __int128)b) >> 64); }
HW_INLINE u64 CLZ64(u64 v) { return v ? (u64)__builtin_clzll(v) : 64; }
HW_INLINE u32 CLZ32(u32 v) { return v ? (u32)__builtin_clz(v) : 32; }
HW_INLINE u64 CLS64(u64 v) { return CLZ64((v ^ (v << 1)) | 1); }
HW_INLINE u32 CLS32(u32 v) { return CLZ32((v ^ (v << 1)) | 1); }
HW_INLINE u32 BSWAP32(u32 v) { return __builtin_bswap32(v); }
HW_INLINE u64 BSWAP64(u64 v) { return __builtin_bswap64(v); }
HW_INLINE u64 REV32_64(u64 v) { return ((u64)__builtin_bswap32((u32)(v >> 32)) << 32) | __builtin_bswap32((u32)v); }
HW_INLINE u32 REV16_32(u32 v) { return ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu); }
HW_INLINE u64 REV16_64(u64 v) { return ((v & 0x00FF00FF00FF00FFull) << 8) | ((v >> 8) & 0x00FF00FF00FF00FFull); }
HW_INLINE u32 RBIT32(u32 v) {
    v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);
    v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);
    v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);
    return __builtin_bswap32(v);
}
HW_INLINE u64 RBIT64(u64 v) { return ((u64)RBIT32((u32)v) << 32) | RBIT32((u32)(v >> 32)); }

// ------------------------------------------------------------------ floating point
HW_INLINE float FABSF(float a) { return __builtin_fabsf(a); }
HW_INLINE double FABSD(double a) { return __builtin_fabs(a); }
HW_INLINE float FSQRTF(float a) { return __builtin_sqrtf(a); }
HW_INLINE double FSQRTD(double a) { return __builtin_sqrt(a); }
HW_INLINE float FMAF(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
HW_INLINE double FMAD(double a, double b, double c) { return __builtin_fma(a, b, c); }
// ARM FMAX/FMIN: NaN propagates, and -0 < +0.
#define HW_FMAXMIN(T, SFX)                                                            \
    HW_INLINE T FMAX##SFX(T a, T b) {                                                 \
        if (a != a) return a + b;                                                     \
        if (b != b) return b + a;                                                     \
        if (a == 0 && b == 0) return __builtin_signbit(a) ? b : a;                    \
        return a > b ? a : b;                                                         \
    }                                                                                 \
    HW_INLINE T FMIN##SFX(T a, T b) {                                                 \
        if (a != a) return a + b;                                                     \
        if (b != b) return b + a;                                                     \
        if (a == 0 && b == 0) return __builtin_signbit(a) ? a : b;                    \
        return a < b ? a : b;                                                         \
    }                                                                                 \
    /* Only a quiet NaN loses to a number; signalling NaNs propagate (quieted). */    \
    HW_INLINE T FMAXNM##SFX(T a, T b) {                                               \
        if (ISSNAN##SFX(a) || ISSNAN##SFX(b)) return a + b;                           \
        if (a != a && b == b) return b;                                               \
        if (b != b && a == a) return a;                                               \
        return FMAX##SFX(a, b);                                                       \
    }                                                                                 \
    HW_INLINE T FMINNM##SFX(T a, T b) {                                               \
        if (ISSNAN##SFX(a) || ISSNAN##SFX(b)) return a + b;                           \
        if (a != a && b == b) return b;                                               \
        if (b != b && a == a) return a;                                               \
        return FMIN##SFX(a, b);                                                       \
    }                                                                                 \
    /* FRECPS / FRSQRTS: inf * 0 yields exactly 2.0 / 1.5. */                         \
    HW_INLINE T FRECPS##SFX(T a, T b) {                                               \
        if (a != a || b != b) return a + b;                                           \
        if ((__builtin_isinf(a) && b == 0) || (a == 0 && __builtin_isinf(b))) return (T)2.0; \
        return FMA##SFX(-a, b, (T)2.0);                                               \
    }                                                                                 \
    HW_INLINE T FRSQRTS##SFX(T a, T b) {                                              \
        if (a != a || b != b) return a + b;                                           \
        if ((__builtin_isinf(a) && b == 0) || (a == 0 && __builtin_isinf(b))) return (T)1.5; \
        return (T)HW_RSQRTSTEP(a, b);                                                 \
    }
// Single rounding of (3 - a*b) in the operand precision; halving is exact unless the
// intermediate overflowed, in which case redo it in double precision.
HW_INLINE double HW_RSQRTSTEP_F(float a, float b) {
    float r = __builtin_fmaf(-a, b, 3.0f);
    if (__builtin_isinf(r)) return (3.0 - (double)a * (double)b) * 0.5;
    return (double)(r * 0.5f);
}
#define HW_RSQRTSTEP(a, b) (sizeof(a) == 4 ? HW_RSQRTSTEP_F((float)(a), (float)(b)) \
                                           : __builtin_fma(-(double)(a), (double)(b), 3.0) * 0.5)
HW_INLINE int ISSNANF(float a) { u32 x; memcpy(&x, &a, 4); return (x & 0x7FC00000u) == 0x7F800000u && (x & 0x3FFFFFu); }
HW_INLINE int ISSNAND(double a) {
    u64 x; memcpy(&x, &a, 8);
    return (x & 0x7FF8000000000000ull) == 0x7FF0000000000000ull && (x & 0x7FFFFFFFFFFFFull);
}
HW_FMAXMIN(float, F)
HW_FMAXMIN(double, D)
HW_INLINE float FMULXF(float a, float b) {
    if ((__builtin_isinf(a) && b == 0) || (a == 0 && __builtin_isinf(b)))
        return (__builtin_signbit(a) ^ __builtin_signbit(b)) ? -2.0f : 2.0f;
    return a * b;
}
HW_INLINE double FMULXD(double a, double b) {
    if ((__builtin_isinf(a) && b == 0) || (a == 0 && __builtin_isinf(b)))
        return (__builtin_signbit(a) ^ __builtin_signbit(b)) ? -2.0 : 2.0;
    return a * b;
}
HW_INLINE float FRINTNF(float a) { return __builtin_rintf(a); }      // ties-to-even (host default)
HW_INLINE double FRINTND(double a) { return __builtin_rint(a); }
HW_INLINE float FRINTXF(float a) { return __builtin_rintf(a); }
HW_INLINE double FRINTXD(double a) { return __builtin_rint(a); }
HW_INLINE float FRINTPF(float a) { return __builtin_ceilf(a); }
HW_INLINE double FRINTPD(double a) { return __builtin_ceil(a); }
HW_INLINE float FRINTMF(float a) { return __builtin_floorf(a); }
HW_INLINE double FRINTMD(double a) { return __builtin_floor(a); }
HW_INLINE float FRINTZF(float a) { return __builtin_truncf(a); }
HW_INLINE double FRINTZD(double a) { return __builtin_trunc(a); }
HW_INLINE float FRINTAF(float a) { return __builtin_roundf(a); }
HW_INLINE double FRINTAD(double a) { return __builtin_round(a); }

// Saturating float->int conversion (ARM semantics: NaN -> 0, clamp to range).
HW_INLINE s32 F2S32(double x) {
    if (x != x) return 0;
    if (x >= 2147483647.0) return INT32_MAX;
    if (x <= -2147483648.0) return INT32_MIN;
    return (s32)x;
}
HW_INLINE u32 F2U32(double x) {
    if (x != x || x <= 0.0) return 0;
    if (x >= 4294967295.0) return UINT32_MAX;
    return (u32)x;
}
HW_INLINE s64 F2S64(double x) {
    if (x != x) return 0;
    if (x >= 9223372036854775808.0) return INT64_MAX;
    if (x <= -9223372036854775808.0) return INT64_MIN;
    return (s64)x;
}
HW_INLINE u64 F2U64(double x) {
    if (x != x || x <= 0.0) return 0;
    if (x >= 18446744073709551616.0) return UINT64_MAX;
    return (u64)x;
}

HW_INLINE void FCMP_FLAGS(Ctx* c, double a, double b) {
    if (a != a || b != b) { c->nf = 0; c->zf = 0; c->cf = 1; c->vf = 1; }
    else if (a == b) { c->nf = 0; c->zf = 1; c->cf = 1; c->vf = 0; }
    else if (a < b) { c->nf = 1; c->zf = 0; c->cf = 0; c->vf = 0; }
    else { c->nf = 0; c->zf = 0; c->cf = 1; c->vf = 0; }
}

HW_INLINE float H2F(u16 h) {
    u32 sign = (u32)(h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {  // subnormal
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
HW_INLINE u16 F2H(float f) {
    u32 x; memcpy(&x, &f, 4);
    u32 sign = (x >> 16) & 0x8000;
    s32 exp = (s32)((x >> 23) & 0xFF) - 127 + 15;
    u32 man = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) return (u16)(sign | 0x7C00 | (man ? 0x200 | (man >> 13) : 0));
    if (exp >= 31) return (u16)(sign | 0x7C00);
    if (exp <= 0) {
        if (exp < -10) return (u16)sign;
        man |= 0x800000;
        u32 shift = (u32)(14 - exp);
        u32 r = man >> shift, rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) r++;
        return (u16)(sign | r);
    }
    u32 r = (sign | ((u32)exp << 10) | (man >> 13));
    u32 rem = man & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
    return (u16)r;
}

// SIMD register-controlled shifts (shift amount is the signed low byte).
#define HW_VSHL(bits)                                                                         \
    HW_INLINE s64 SSHL##bits(s64 a, s8 sh) {                                                  \
        if (sh >= 0) return sh >= bits ? 0 : (s64)((u64)a << sh);                              \
        return -sh >= bits ? (a < 0 ? -1 : 0) : a >> -sh;                                      \
    }                                                                                         \
    HW_INLINE u64 USHL##bits(u64 a, s8 sh) {                                                  \
        if (sh >= 0) return sh >= bits ? 0 : a << sh;                                          \
        return -sh >= bits ? 0 : a >> -sh;                                                     \
    }
HW_VSHL(8)
HW_VSHL(16)
HW_VSHL(32)
HW_VSHL(64)

// ------------------------------------------------------------------ CRC32 / CRC32C
HW_INLINE u32 CRC32_UPDATE(u32 crc, u64 data, int bytes, u32 poly) {
    for (int i = 0; i < bytes; i++) {
        crc ^= (u8)(data >> (8 * i));
        for (int b = 0; b < 8; b++) crc = (crc >> 1) ^ (poly & (0u - (crc & 1)));
    }
    return crc;
}
#define CRC32(crc, data, bytes) CRC32_UPDATE((u32)(crc), (u64)(data), bytes, 0xEDB88320u)
HW_INLINE u32 CRC32C(u32 crc, u64 data, int bytes) {
#if defined(__SSE4_2__)
    switch (bytes) {
    case 1: return __builtin_ia32_crc32qi(crc, (u8)data);
    case 2: return __builtin_ia32_crc32hi(crc, (u16)data);
    case 4: return __builtin_ia32_crc32si(crc, (u32)data);
    default: return (u32)__builtin_ia32_crc32di(crc, data);
    }
#else
    return CRC32_UPDATE(crc, data, bytes, 0x82F63B78u);
#endif
}

// ------------------------------------------------------------------ AES / PMULL (x86 AES-NI, PCLMUL)
typedef long long hw_v2di __attribute__((vector_size(16)));
HW_INLINE hw_v2di V2DI(V128 v) { hw_v2di r; memcpy(&r, &v, 16); return r; }
HW_INLINE V128 FROM_V2DI(hw_v2di x) { V128 r; memcpy(&r, &x, 16); return r; }
// AESE: SubBytes(ShiftRows(d ^ k)) == AESENCLAST(d ^ k, 0)
HW_INLINE V128 AESE(V128 d, V128 k) {
    hw_v2di z = {0, 0};
    return FROM_V2DI(__builtin_ia32_aesenclast128(V2DI(d) ^ V2DI(k), z));
}
HW_INLINE V128 AESD(V128 d, V128 k) {
    hw_v2di z = {0, 0};
    return FROM_V2DI(__builtin_ia32_aesdeclast128(V2DI(d) ^ V2DI(k), z));
}
// AESMC: MixColumns(x) == AESENC(AESDECLAST(x, 0), 0) (the Sub/Shift steps cancel out)
HW_INLINE V128 AESMC(V128 d) {
    hw_v2di z = {0, 0};
    return FROM_V2DI(__builtin_ia32_aesenc128(__builtin_ia32_aesdeclast128(V2DI(d), z), z));
}
HW_INLINE V128 AESIMC(V128 d) { return FROM_V2DI(__builtin_ia32_aesimc128(V2DI(d))); }
HW_INLINE V128 PMULL64(u64 a, u64 b) {
    hw_v2di x = {(long long)a, 0}, y = {(long long)b, 0};
    return FROM_V2DI(__builtin_ia32_pclmulqdq128(x, y, 0));
}
HW_INLINE u16 PMUL8(u8 a, u8 b) {
    u16 r = 0;
    for (int i = 0; i < 8; i++)
        if (b & (1 << i)) r ^= (u16)a << i;
    return r;
}

// ------------------------------------------------------------------ SHA-1 / SHA-256 (ARMv8 crypto)
HW_INLINE u32 HW_ROL32(u32 x, int n) { return (x << n) | (x >> (32 - n)); }
HW_INLINE u32 HW_ROR32(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
// SHA1C/P/M: abcd = Qd, e = Sn, wk = Vm.4S
HW_INLINE V128 SHA1HASH(V128 abcd, u32 e, V128 wk, int kind) {
    u32 a = abcd.s[0], b = abcd.s[1], c = abcd.s[2], d = abcd.s[3];
    for (int i = 0; i < 4; i++) {
        u32 t = kind == 0 ? ((b & c) | (~b & d)) : kind == 1 ? (b ^ c ^ d) : ((b & c) | (b & d) | (c & d));
        e = e + HW_ROL32(a, 5) + t + wk.s[i];
        b = HW_ROL32(b, 30);
        u32 ne = d; d = c; c = b; b = a; a = e; e = ne;
    }
    V128 r = {{0}};
    r.s[0] = a; r.s[1] = b; r.s[2] = c; r.s[3] = d;
    return r;
}
HW_INLINE V128 SHA1SU0(V128 d, V128 n, V128 m) {
    V128 r;
    r.d[0] = d.d[1];
    r.d[1] = n.d[0];
    r.d[0] ^= d.d[0] ^ m.d[0];
    r.d[1] ^= d.d[1] ^ m.d[1];
    return r;
}
HW_INLINE V128 SHA1SU1(V128 d, V128 n) {
    V128 t, r;
    for (int i = 0; i < 4; i++) t.s[i] = d.s[i] ^ (i < 3 ? n.s[i + 1] : 0);
    for (int i = 0; i < 3; i++) r.s[i] = HW_ROL32(t.s[i], 1);
    r.s[3] = HW_ROL32(t.s[3], 1) ^ HW_ROL32(t.s[0], 2);
    return r;
}
HW_INLINE V128 SHA256HASH(V128 x, V128 y, V128 w, int part2) {
    u32 a = x.s[0], b = x.s[1], c = x.s[2], d = x.s[3], e = y.s[0], f = y.s[1], g = y.s[2], h = y.s[3];
    for (int i = 0; i < 4; i++) {
        u32 ch = (e & f) ^ (~e & g);
        u32 maj = (a & b) ^ (a & c) ^ (b & c);
        u32 s0 = HW_ROR32(a, 2) ^ HW_ROR32(a, 13) ^ HW_ROR32(a, 22);
        u32 s1 = HW_ROR32(e, 6) ^ HW_ROR32(e, 11) ^ HW_ROR32(e, 25);
        u32 t = h + s1 + ch + w.s[i];
        u32 na = t + s0 + maj;
        u32 ne = d + t;
        h = g; g = f; f = e; e = ne; d = c; c = b; b = a; a = na;
    }
    V128 r;
    if (!part2) { r.s[0] = a; r.s[1] = b; r.s[2] = c; r.s[3] = d; }
    else { r.s[0] = e; r.s[1] = f; r.s[2] = g; r.s[3] = h; }
    return r;
}
HW_INLINE V128 SHA256SU0(V128 d, V128 n) {
    V128 r;
    for (int i = 0; i < 4; i++) {
        u32 w = i < 3 ? d.s[i + 1] : n.s[0];
        r.s[i] = d.s[i] + (HW_ROR32(w, 7) ^ HW_ROR32(w, 18) ^ (w >> 3));
    }
    return r;
}
HW_INLINE V128 SHA256SU1(V128 d, V128 n, V128 m) {
    V128 r;
    u32 t[4];
    for (int i = 0; i < 4; i++) t[i] = d.s[i] + (i < 3 ? n.s[i + 1] : m.s[0]);
    for (int i = 0; i < 2; i++) {
        u32 w = m.s[i + 2];
        r.s[i] = t[i] + (HW_ROR32(w, 17) ^ HW_ROR32(w, 19) ^ (w >> 10));
    }
    for (int i = 2; i < 4; i++) {
        u32 w = r.s[i - 2];
        r.s[i] = t[i] + (HW_ROR32(w, 17) ^ HW_ROR32(w, 19) ^ (w >> 10));
    }
    return r;
}

// ------------------------------------------------------------------ saturating integer helpers
HW_INLINE s64 HW_SAT_S(s64 v, int bits) {
    s64 hi = (s64)((1ull << (bits - 1)) - 1), lo = -hi - 1;
    return v > hi ? hi : v < lo ? lo : v;
}
HW_INLINE u64 HW_SAT_U(s64 v, int bits) {
    u64 hi = bits == 64 ? ~0ull : (1ull << bits) - 1;
    return v < 0 ? 0 : (u64)v > hi ? hi : (u64)v;
}
HW_INLINE s64 HW_SQADD64(s64 a, s64 b) {
    s64 r;
    if (__builtin_add_overflow(a, b, &r)) return a < 0 ? INT64_MIN : INT64_MAX;
    return r;
}
HW_INLINE u64 HW_UQADD64(u64 a, u64 b) { u64 r = a + b; return r < a ? ~0ull : r; }
HW_INLINE s64 HW_SQSUB64(s64 a, s64 b) {
    s64 r;
    if (__builtin_sub_overflow(a, b, &r)) return a < 0 ? INT64_MIN : INT64_MAX;
    return r;
}
HW_INLINE u64 HW_UQSUB64(u64 a, u64 b) { return a > b ? a - b : 0; }

// ------------------------------------------------------------------ runtime hooks
u64 hw_cntvct(void);
void hw_yield(void);
void hw_svc(Ctx* c, u32 imm);
void hw_brk(Ctx* c, u64 pc, u32 imm);
void hw_unimpl(Ctx* c, u64 pc, u32 word);

#ifdef __cplusplus
}
#endif
