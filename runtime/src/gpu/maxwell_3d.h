// Maxwell B (class 0xB197) 3D engine method indices (byte offset / 4) and register helpers.
#pragma once
#include "hwder/runtime.h"

namespace gpu::m3d {

constexpr u32 kNumRegs = 0xE00;

enum : u32 {
    WaitForIdle = 0x44,
    LoadMmeInstructionRamPointer = 0x45,
    LoadMmeInstructionRam = 0x46,
    LoadMmeStartAddressRamPointer = 0x47,
    LoadMmeStartAddressRam = 0x48,
    ShadowRamControl = 0x49,
    // inline-to-memory (same layout as class A140)
    I2mLineLengthIn = 0x60, I2mLineCount = 0x61, I2mOffsetOutUpper = 0x62, I2mOffsetOut = 0x63,
    I2mPitchOut = 0x64, I2mDstBlockSize = 0x65, I2mDstWidth = 0x66, I2mDstHeight = 0x67, I2mDstDepth = 0x68,
    I2mDstLayer = 0x69, I2mDstOriginX = 0x6A, I2mDstOriginY = 0x6B, I2mLaunchDma = 0x6C, I2mLoadInlineData = 0x6D,
    SyncInfo = 0xB2,
    RenderTarget = 0x200,          // 8 x 16 regs: addr hi, addr lo, width, height, format, tile mode, depth/array, layer stride, ...
    ViewportTransform = 0x280,     // 16 x 8 regs: scale xyz, translate xyz, swizzle, snap
    Viewport = 0x300,              // 16 x 4 regs: x|w, y|h, depth near, depth far
    VertexBufferFirst = 0x35D,
    VertexBufferCount = 0x35E,
    DepthMode = 0x35F,
    ClearColor = 0x360,            // 4 floats
    ClearDepth = 0x364,
    ClearStencil = 0x368,
    PolygonModeFront = 0x36B, PolygonModeBack = 0x36C,
    ScissorTest = 0x380,           // 16 x 4 regs: enable, x min|max, y min|max
    StencilBackRef = 0x3D5, StencilBackMask = 0x3D6, StencilBackFuncMask = 0x3D7,
    ColorMaskCommon = 0x3E4,
    DepthBoundsMin = 0x3E7, DepthBoundsMax = 0x3E8,
    RtControlMrtEnable = 0x3EB,
    Zeta = 0x3F8,                  // addr hi, addr lo, format, tile mode, layer stride
    VertexAttribFormat = 0x458,    // 32 regs
    VertexArrayInstanceFirst = 0x485,
    VertexArrayInstanceSubsequent = 0x486,
    RtControl = 0x487,
    ZetaSize = 0x48A,              // width, height, depth/array mode
    SamplerBinding = 0x48D,
    DepthTestEnable = 0x4B3,
    BlendPerTargetEnabled = 0x4B9,
    DepthWriteEnable = 0x4BA,
    AlphaTestEnable = 0x4BB,
    InlineIndex4x8 = 0x4C0,
    DepthTestFunc = 0x4C3,
    AlphaTestRef = 0x4C4, AlphaTestFunc = 0x4C5,
    BlendColor = 0x4C7,
    Blend = 0x4CF,                 // common blend state
    StencilEnable = 0x4E0,
    StencilFrontOp = 0x4E1,        // fail, zfail, zpass, func
    StencilFrontRef = 0x4E5, StencilFrontFuncMask = 0x4E6, StencilFrontMask = 0x4E7,
    WindowOrigin = 0x4EB,
    LineWidthSmooth = 0x4EC, LineWidthAliased = 0x4ED,
    GlobalBaseVertexIndex = 0x50D,
    GlobalBaseInstanceIndex = 0x50E,
    ClearReportValue = 0x54C,
    ZetaEnable = 0x54E,
    TexSampler = 0x557,            // addr hi, addr lo, limit
    SlopeScaleDepthBias = 0x55B,
    TexHeader = 0x55D,             // addr hi, addr lo, limit
    StencilTwoSideEnable = 0x565,
    StencilBackOp = 0x566,
    FramebufferSrgb = 0x56E,
    DepthBias = 0x56F,
    DrawInlineIndex = 0x57A,
    InlineIndex2x16 = 0x57B,
    VertexGlobalBaseOffset = 0x57D,
    ProgramRegion = 0x582,         // addr hi, addr lo
    DrawEnd = 0x585,
    DrawBegin = 0x586,
    PrimitiveRestart = 0x591,      // enable, index
    ProvokingVertex = 0x5A1,
    IndexBuffer = 0x5F2,           // addr hi, addr lo, limit hi, limit lo, format, first, count
    IndexBuffer32First = 0x5F9, IndexBuffer16First = 0x5FA, IndexBuffer8First = 0x5FB,
    IndexBuffer32Subsequent = 0x5FC, IndexBuffer16Subsequent = 0x5FD, IndexBuffer8Subsequent = 0x5FE,
    VertexStreamInstances = 0x620, // 32 regs: per-stream instancing enable
    GlCullTestEnabled = 0x646, GlFrontFace = 0x647, GlCullFace = 0x648,
    ViewportClipControl = 0x64F,
    LogicOp = 0x671,               // enable, op
    ClearSurface = 0x674,
    ColorMask = 0x680,             // 8 regs
    ReportSemaphore = 0x6C0,       // addr hi, addr lo, payload, query
    VertexStreams = 0x700,         // 32 x 4 regs: config (stride|enable), addr hi, addr lo, frequency
    BlendPerTarget = 0x780,        // 8 x 8 regs
    VertexStreamLimits = 0x7C0,    // 32 x 2 regs: limit hi, limit lo
    Pipelines = 0x800,             // 6 x 16 regs: config (enable|program type), offset, ..., register count
    ConstBuffer = 0x8E0,           // size, addr hi, addr lo, offset, data[16]
    ConstBufferData = 0x8E4,
    BindGroups = 0x900,            // 5 x 8 regs; +4: cbuf bind (valid | index << 4)
    BindlessTextureConstBufferSlot = 0x982,
    MacroStart = 0xE00,
};

// CLEAR_SURFACE value fields
inline bool clear_z(u32 v) { return v & 1; }
inline bool clear_s(u32 v) { return (v >> 1) & 1; }
inline u32 clear_rgba_mask(u32 v) { return (v >> 2) & 0xF; }
inline u32 clear_rt(u32 v) { return (v >> 6) & 0xF; }
inline u32 clear_layer(u32 v) { return (v >> 10) & 0x7FF; }

inline u64 addr(const u32* regs, u32 hi_index) { return ((u64)regs[hi_index] << 32) | regs[hi_index + 1]; }

}  // namespace gpu::m3d
