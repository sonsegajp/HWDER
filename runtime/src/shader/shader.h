// Maxwell (SM5.x) shader -> SPIR-V translator. Contract between the translator (runtime/src/shader/*)
// and the Vulkan renderer (runtime/src/gpu/vk/*).
//
// Binding model (the renderer builds descriptor set layouts from Program::resources):
//   graphics: descriptor set = stage index (0 VertexB, 1 TessControl, 2 TessEval, 3 Geometry, 4 Fragment)
//   compute : descriptor set 0
//   bindings within a set are assigned sequentially in this order:
//     constant buffers (uniform buffers, std140-compatible vec4 arrays), storage buffers,
//     texel buffers, sampled textures (combined image samplers), storage images
//   Vertex inputs: location = generic attribute index (0..31).
//   Fragment outputs: location = render target index (0..7).
//   Push constants (vertex/fragment, offset 0): struct PushConstants below.
#pragma once
#include <vector>

#include "hwder/runtime.h"

namespace shader {

enum class Stage : u32 { VertexA = 0, VertexB, TessControl, TessEval, Geometry, Fragment, Compute };

enum class TextureType : u32 { Tex1D, Tex1DArray, Tex2D, Tex2DArray, Tex3D, Cube, CubeArray, Buffer };

// Numeric type of a vertex attribute / render target (Vulkan requires the shader interface type to match
// the format's numeric type). Added for Environment::attribute_type / render_target_type.
enum class AttributeType : u32 { Float = 0, SignedInt, UnsignedInt };

struct ConstBufferUse {
    u32 index;     // cbuf slot c[index]
    u32 size;      // bytes accessed (upper bound; 0x10000 when indexed dynamically)
};
struct StorageBufferUse {
    // Maxwell has no SSBO binding: the address/size pair is read from a constant buffer.
    u32 cbuf_index, cbuf_offset;
    bool written;
};
struct TextureUse {
    TextureType type;
    bool is_depth;          // shadow compare
    bool is_multisample;
    // Handle location: bound textures read c[cbuf_index][cbuf_offset] (u32 handle: tic index in bits
    // 0-19, tsc index in bits 20-31 unless the sampler is separate). Bindless uses the same fields.
    u32 cbuf_index, cbuf_offset;
    bool has_secondary;     // separate sampler handle at secondary_cbuf_index/offset
    u32 secondary_cbuf_index, secondary_cbuf_offset;
    u32 count;              // array size (1 for non-arrayed bindings)
};
struct ImageUse {
    TextureType type;
    u32 cbuf_index, cbuf_offset;
    bool is_written, is_read;
    u32 format_hint;        // 0 = unknown (renderer uses the TIC format)
};

struct Resources {
    std::vector<ConstBufferUse> const_buffers;
    std::vector<StorageBufferUse> storage_buffers;
    std::vector<TextureUse> texel_buffers;
    std::vector<TextureUse> textures;
    std::vector<ImageUse> images;
};

struct PushConstants {
    float viewport_scale[4];   // x, y scale applied for y-flip / render scaling (renderer-defined use)
    u32 base_vertex, base_instance;
    u32 rt_mask;               // enabled render targets
    u32 pad;
};

struct Program {
    Stage stage;
    std::vector<u32> spirv;
    Resources resources;
    u32 attributes_read = 0;     // vertex: bitmask of generic attributes read
    u32 outputs_written = 0;     // fragment: bitmask of render targets written; others: varyings mask
    bool writes_depth = false, uses_discard = false;
    u32 local_size[3] = {1, 1, 1};
    u32 shared_memory_size = 0, local_memory_size = 0;
    u32 geometry_output_topology = 0, geometry_max_vertices = 0, geometry_invocations = 1;
};

// Guest state the translator needs. The renderer implements it.
class Environment {
public:
    virtual ~Environment() = default;
    virtual u64 read_instruction(u32 address) = 0;   // 64-bit word at byte offset from program start (after SPH for graphics)
    virtual const u8* sph() = 0;                      // 0x50-byte shader program header (graphics stages)
    virtual u32 read_cbuf(u32 index, u32 offset) = 0;  // for bindless/handle discovery
    virtual TextureType texture_type(u32 handle) = 0;  // resolve type from the TIC entry
    virtual u32 texture_bound_buffer() = 0;           // CB_BIND slot holding bound texture handles (Maxwell: 2 for graphics)
    virtual u32 local_size(int axis) { return 1; }     // compute
    virtual u32 shared_memory_size() { return 0; }    // compute
    // Added (backwards compatible, defaults = Float). Integer vertex formats (*_UINT/*_SINT) must be
    // reported so the input is declared as uvec4/ivec4; same for integer colour render targets. The
    // returned types are baked into the SPIR-V, so they belong in the pipeline key when non-default.
    virtual AttributeType attribute_type(u32 index) { return AttributeType::Float; }
    virtual AttributeType render_target_type(u32 rt) { return AttributeType::Float; }
};

struct Options {
    // flip_y: the last pre-raster stage negates gl_Position.y (static; no push constant needed).
    // Alternatively leave it false and use a negative-height viewport.
    bool flip_y = false;
    bool convert_depth_mode = false;  // GL [-1,1] -> Vulkan [0,1] in the last pre-raster stage: z = (z + w) / 2
};
// Push constants: the translator only declares the block when a shader needs it (currently
// base_instance for InstanceId, which is gl_InstanceIndex - base_instance as on Maxwell).

// Translate one stage. vertex_a (optional) is the VertexA program merged in front of VertexB.
// Never throws: unsupported instructions are logged and translated to no-ops (or the shader falls
// back to a pass-through), so a draw is never skipped because of the translator.
Program translate(Environment& env, Stage stage, const Options& opt, Environment* vertex_a = nullptr);

// Debug: text listing of the Maxwell instruction stream (pc, raw word, opcode, register fields).
std::string listing(Environment& env, Stage stage);

// Stable hash of the program bytes (for pipeline cache keys).
u64 hash_program(Environment& env, Stage stage);

}  // namespace shader
