// Program-level pieces of the translator: types, register file, interface variables, constant buffers,
// textures, prologue/epilogue and final module assembly.
#include <algorithm>
#include <cstring>

#include "shader/translator.h"

namespace shader {

Translator::Translator(Environment& e, Stage s, const Options& o) : env(e), stage(s), opt(o) {
    if (stage != Stage::Compute) memcpy(sph, env.sph(), 0x50);
    is_vs = stage == Stage::VertexA || stage == Stage::VertexB;
    is_fs = stage == Stage::Fragment;
    prog.stage = stage;
}

void Translator::begin_program() {
    b.capability(spv::CapabilityShader);
    b.import_glsl();
    b.put(b.memory_model, spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
    t_void = b.t_void();
    t_bool = b.t_bool();
    t_u32 = b.t_uint();
    t_s32 = b.t_int();
    t_f32 = b.t_float();
    t_v2f = b.t_vec(t_f32, 2);
    t_v3f = b.t_vec(t_f32, 3);
    t_v4f = b.t_vec(t_f32, 4);
    t_v2u = b.t_vec(t_u32, 2);
    t_v3u = b.t_vec(t_u32, 3);
    t_v4u = b.t_vec(t_u32, 4);
    t_v4s = b.t_vec(t_s32, 4);
    t_v2b = b.t_vec(t_bool, 2);
    ptr_fn_u32 = b.t_ptr(spv::StorageClassFunction, t_u32);
    ptr_fn_bool = b.t_ptr(spv::StorageClassFunction, t_bool);
}

// ------------------------------------------------------------------------------ register file

Id Translator::R(u32 r) {
    if (r >= 255) return u(0);
    if (!regs[r]) regs[r] = b.local_var(ptr_fn_u32, u(0));
    return b.load(t_u32, regs[r]);
}
Id Translator::F(u32 r) { return r >= 255 ? fc(0) : bc_f(R(r)); }
void Translator::setR(u32 r, Id v) {
    if (r >= 255) return;
    if (!regs[r]) regs[r] = b.local_var(ptr_fn_u32, u(0));
    b.store(regs[r], v);
}
void Translator::setF(u32 r, Id v) {
    if (r >= 255) return;
    setR(r, bc_u(v));
}
Id Translator::pred(u32 p, bool neg) {
    Id v;
    if (p >= 7) {
        v = b.c_bool(true);
    } else {
        if (!preds[p]) preds[p] = b.local_var(ptr_fn_bool, b.c_bool(false));
        v = b.load(t_bool, preds[p]);
    }
    return neg ? lnot(v) : v;
}
void Translator::set_pred(u32 p, Id v) {
    if (p >= 7) return;
    if (!preds[p]) preds[p] = b.local_var(ptr_fn_bool, b.c_bool(false));
    b.store(preds[p], v);
}
Id Translator::flag(int i) {
    if (!flags[i]) flags[i] = b.local_var(ptr_fn_bool, b.c_bool(false));
    return b.load(t_bool, flags[i]);
}
void Translator::set_flag(int i, Id v) {
    if (!flags[i]) flags[i] = b.local_var(ptr_fn_bool, b.c_bool(false));
    b.store(flags[i], v);
}

Id Translator::flow_test(u32 cc) {
    auto Z = [&] { return flag(0); };
    auto S = [&] { return flag(1); };
    auto C = [&] { return flag(2); };
    auto O = [&] { return flag(3); };
    switch (cc) {
    case 0: return b.c_bool(false);
    case 1: return lxor(land(S(), lnot(Z())), O());
    case 2: return land(lnot(S()), Z());
    case 3: return lxor(S(), lor(Z(), O()));
    case 4: return land(lxor(lnot(S()), O()), lnot(Z()));
    case 5: return lnot(Z());
    case 6: return lnot(lxor(S(), O()));
    case 7: return lor(lnot(S()), lnot(Z()));
    case 8: return land(S(), Z());
    case 9: return lxor(S(), O());
    case 10: return Z();
    case 11: return lor(lxor(S(), O()), Z());
    case 12: return lxor(lnot(S()), lor(Z(), O()));
    case 13: return lor(S(), lnot(Z()));
    case 14: return lxor(lor(lnot(S()), Z()), O());
    case 15: return b.c_bool(true);
    case 16: return lnot(O());
    case 17: return lnot(C());
    case 18: return lnot(S());
    case 19: return lor(Z(), lnot(C()));
    case 20: return land(C(), lnot(Z()));
    case 21: return S();
    case 22: return C();
    case 23: return O();
    case 28: return lor(S(), Z());
    case 29: return land(lnot(S()), lnot(Z()));
    default:
        log_once("unsupported flow test " + std::to_string(cc));
        return b.c_bool(false);
    }
}

// --------------------------------------------------------------------------------- interface

Id Translator::io_var(spv::StorageClass sc, Id type, int location, int builtin) {
    Id v = b.id();
    b.put(b.globals, spv::OpVariable, {b.t_ptr(sc, type), v, u32(sc)});
    if (location >= 0) b.decorate(v, spv::DecorationLocation, {u32(location)});
    if (builtin >= 0) b.decorate(v, spv::DecorationBuiltIn, {u32(builtin)});
    io_vars.push_back(v);
    return v;
}

Id Translator::push_member(u32 index, Id type) {
    if (!push_var) {
        Id st = b.t_struct_fresh({t_v4f, t_u32, t_u32, t_u32, t_u32});
        b.decorate(st, spv::DecorationBlock);
        const u32 offs[] = {0, 16, 20, 24, 28};
        for (u32 i = 0; i < 5; i++) b.member_decorate(st, i, spv::DecorationOffset, {offs[i]});
        push_var = b.id();
        b.put(b.globals, spv::OpVariable,
              {b.t_ptr(spv::StorageClassPushConstant, st), push_var, u32(spv::StorageClassPushConstant)});
    }
    Id p = b.op(spv::OpAccessChain, b.t_ptr(spv::StorageClassPushConstant, type), {push_var, u(index)});
    return b.load(type, p);
}

Id Translator::input_generic(u32 i) {
    if (in_generic[i]) return in_generic[i];
    Id t = t_v4f;
    if (is_vs) {
        AttributeType at = env.attribute_type(i);
        in_generic_kind[i] = at == AttributeType::SignedInt ? 1 : at == AttributeType::UnsignedInt ? 2 : 0;
        t = in_generic_kind[i] == 1 ? t_v4s : in_generic_kind[i] == 2 ? t_v4u : t_v4f;
        prog.attributes_read |= 1u << i;
    }
    in_generic_t[i] = t;
    in_generic[i] = io_var(spv::StorageClassInput, t, int(i), -1);
    if (is_fs) {
        u8 m = sph[0x18 + i];
        bool flat = false, linear = false;
        for (int c = 0; c < 4; c++) {
            u32 mode = (m >> (c * 2)) & 3;
            if (mode == 1) flat = true;
            if (mode == 3) linear = true;
        }
        if (flat) b.decorate(in_generic[i], spv::DecorationFlat);
        else if (linear) b.decorate(in_generic[i], spv::DecorationNoPerspective);
    }
    return in_generic[i];
}

Id Translator::output_generic(u32 i) {
    if (!out_generic[i]) {
        out_generic[i] = io_var(spv::StorageClassOutput, t_v4f, int(i), -1);
        if (!is_fs) prog.outputs_written |= 1u << i;
    }
    return out_generic[i];
}

Id Translator::lane_id() {
    if (!subgroup_id) {
        b.capability(spv::CapabilityGroupNonUniform);
        subgroup_id = io_var(spv::StorageClassInput, t_u32, -1, spv::BuiltInSubgroupLocalInvocationId);
        if (is_fs) b.decorate(subgroup_id, spv::DecorationFlat);
    }
    return b.load(t_u32, subgroup_id);
}

static Id component_ptr(SpvBuilder& b, spv::StorageClass sc, Id elem_t, Id var, Id idx) {
    return b.op(spv::OpAccessChain, b.t_ptr(sc, elem_t), {var, idx});
}

// Reads input attribute word at byte address addr. Returns u32 bits.
Id Translator::load_attribute(u32 addr) {
    u32 comp = (addr >> 2) & 3;
    if (addr >= 0x80 && addr < 0x280) {
        u32 i = (addr - 0x80) >> 4;
        Id var = input_generic(i);
        u32 kind = in_generic_kind[i];
        Id et = kind == 1 ? t_s32 : kind == 2 ? t_u32 : t_f32;
        Id v = b.load(et, component_ptr(b, spv::StorageClassInput, et, var, u(comp)));
        return et == t_u32 ? v : b.bitcast(t_u32, v);
    }
    if (is_fs) {
        if (addr >= 0x70 && addr < 0x80) {
            if (!frag_coord) frag_coord = io_var(spv::StorageClassInput, t_v4f, -1, spv::BuiltInFragCoord);
            return bc_u(b.load(t_f32, component_ptr(b, spv::StorageClassInput, t_f32, frag_coord, u(comp))));
        }
        if (addr == 0x3FC) {
            if (!front_facing) front_facing = io_var(spv::StorageClassInput, t_bool, -1, spv::BuiltInFrontFacing);
            return select(t_u32, b.load(t_bool, front_facing), u(0xFFFFFFFF), u(0));
        }
        if (addr == 0x2E0 || addr == 0x2E4) {
            if (!point_coord) point_coord = io_var(spv::StorageClassInput, t_v2f, -1, spv::BuiltInPointCoord);
            return bc_u(b.load(t_f32, component_ptr(b, spv::StorageClassInput, t_f32, point_coord, u((addr >> 2) & 1))));
        }
        if (addr == 0x60) {
            if (!primitive_id) {
                b.capability(spv::CapabilityGeometry);
                primitive_id = io_var(spv::StorageClassInput, t_s32, -1, spv::BuiltInPrimitiveId);
                b.decorate(primitive_id, spv::DecorationFlat);
            }
            return bc_u(b.load(t_s32, primitive_id));
        }
    }
    if (is_vs) {
        if (addr == 0x2FC) {
            if (!vertex_index) vertex_index = io_var(spv::StorageClassInput, t_s32, -1, spv::BuiltInVertexIndex);
            return bc_u(b.load(t_s32, vertex_index));
        }
        if (addr == 0x2F8) {
            if (!instance_index) instance_index = io_var(spv::StorageClassInput, t_s32, -1, spv::BuiltInInstanceIndex);
            return op2(spv::OpISub, t_u32, bc_u(b.load(t_s32, instance_index)), push_member(2, t_u32));
        }
    }
    log_once("unhandled input attribute 0x" + std::to_string(addr));
    return u(0);
}

void Translator::store_attribute(u32 addr, Id v) {
    u32 comp = (addr >> 2) & 3;
    if (addr >= 0x80 && addr < 0x280) {
        Id var = output_generic((addr - 0x80) >> 4);
        b.store(component_ptr(b, spv::StorageClassOutput, t_f32, var, u(comp)), bc_f(v));
        return;
    }
    if (addr >= 0x70 && addr < 0x80) {
        b.store(component_ptr(b, spv::StorageClassOutput, t_f32, position, u(comp)), bc_f(v));
        return;
    }
    if (addr == 0x6C) {
        if (!point_size) point_size = io_var(spv::StorageClassOutput, t_f32, -1, spv::BuiltInPointSize);
        b.store(point_size, bc_f(v));
        return;
    }
    if (addr >= 0x2C0 && addr < 0x2E0) {
        if (!clip_distance) {
            b.capability(spv::CapabilityClipDistance);
            clip_distance = io_var(spv::StorageClassOutput, b.t_array(t_f32, 8), -1, spv::BuiltInClipDistance);
        }
        b.store(component_ptr(b, spv::StorageClassOutput, t_f32, clip_distance, u((addr - 0x2C0) >> 2)), bc_f(v));
        return;
    }
    if (addr == 0x64 || addr == 0x68) {
        Id& var = addr == 0x64 ? layer_out : viewport_out;
        if (!var) {
            b.capability(spv::CapabilityShaderViewportIndexLayerEXT);
            b.extension("SPV_EXT_shader_viewport_index_layer");
            var = io_var(spv::StorageClassOutput, t_s32, -1, addr == 0x64 ? spv::BuiltInLayer : spv::BuiltInViewportIndex);
        }
        b.store(var, b.bitcast(t_s32, v));
        return;
    }
    log_once("unhandled output attribute 0x" + std::to_string(addr));
}

// ---------------------------------------------------------------------------- constant buffers

Id Translator::cbuf_var(u32 index) {
    Cbuf& cb = cbufs[index];
    if (!cb.var) {
        if (!cbuf_struct) {
            Id arr = b.t_array_fresh(t_v4u, 4096);
            b.decorate(arr, spv::DecorationArrayStride, {16});
            cbuf_struct = b.t_struct_fresh({arr});
            b.decorate(cbuf_struct, spv::DecorationBlock);
            b.member_decorate(cbuf_struct, 0, spv::DecorationOffset, {0});
            cbuf_ptr_u32 = b.t_ptr(spv::StorageClassUniform, t_u32);
        }
        cb.var = b.id();
        b.put(b.globals, spv::OpVariable,
              {b.t_ptr(spv::StorageClassUniform, cbuf_struct), cb.var, u32(spv::StorageClassUniform)});
    }
    return cb.var;
}

Id Translator::cbuf_load(u32 index, u32 offset) {
    Id var = cbuf_var(index);
    Cbuf& cb = cbufs[index];
    cb.size = std::max(cb.size, std::min<u32>(0x10000, ((offset & ~3u) + 16) & ~15u));
    if (offset >= 0x10000) return u(0);
    Id p = b.op(spv::OpAccessChain, cbuf_ptr_u32, {var, u(0), u(offset >> 4), u((offset >> 2) & 3)});
    Id v = b.load(t_u32, p);
    if (offset & 3) v = op2(spv::OpShiftRightLogical, t_u32, v, u((offset & 3) * 8));
    return v;
}

Id Translator::cbuf_load_dyn(u32 index, Id off) {
    Id var = cbuf_var(index);
    cbufs[index].size = 0x10000;
    Id vec = op2(spv::OpShiftRightLogical, t_u32, off, u(4));
    vec = glsl(t_u32, GLSLstd450UMin, {vec, u(4095)});
    Id comp = op2(spv::OpBitwiseAnd, t_u32, op2(spv::OpShiftRightLogical, t_u32, off, u(2)), u(3));
    Id p = b.op(spv::OpAccessChain, cbuf_ptr_u32, {var, u(0), vec, comp});
    return b.load(t_u32, p);
}

// ------------------------------------------------------------------------------------ textures

Translator::Tex& Translator::texture(u32 cbuf_offset, TextureType type, bool depth) {
    for (auto& t : textures)
        if (t.cbuf_offset == cbuf_offset && t.type == type && t.depth == depth) return t;
    Tex t{};
    t.type = type;
    t.depth = depth;
    t.cbuf_index = env.texture_bound_buffer();
    t.cbuf_offset = cbuf_offset;
    spv::Dim dim = spv::Dim2D;
    u32 arrayed = 0;
    switch (type) {
    case TextureType::Tex1D: dim = spv::Dim1D; b.capability(spv::CapabilitySampled1D); break;
    case TextureType::Tex1DArray: dim = spv::Dim1D; arrayed = 1; b.capability(spv::CapabilitySampled1D); break;
    case TextureType::Tex2D: break;
    case TextureType::Tex2DArray: arrayed = 1; break;
    case TextureType::Tex3D: dim = spv::Dim3D; break;
    case TextureType::Cube: dim = spv::DimCube; break;
    case TextureType::CubeArray: dim = spv::DimCube; arrayed = 1; b.capability(spv::CapabilitySampledCubeArray); break;
    case TextureType::Buffer: dim = spv::DimBuffer; b.capability(spv::CapabilitySampledBuffer); break;
    }
    t.image_t = b.type(spv::OpTypeImage, {t_f32, u32(dim), depth ? 1u : 0u, arrayed, 0, 1, u32(spv::ImageFormatUnknown)});
    t.sampled_t = b.type(spv::OpTypeSampledImage, {t.image_t});
    t.var = b.id();
    b.put(b.globals, spv::OpVariable,
          {b.t_ptr(spv::StorageClassUniformConstant, t.sampled_t), t.var, u32(spv::StorageClassUniformConstant)});
    textures.push_back(t);
    return textures.back();
}

// ----------------------------------------------------------------------------- prologue/epilogue

void Translator::prologue() {
    if (is_vs) {
        position = io_var(spv::StorageClassOutput, t_v4f, -1, spv::BuiltInPosition);
        b.decorate(position, spv::DecorationInvariant);
        b.store(position, b.c_composite(t_v4f, {fc(0), fc(0), fc(0), fc(1)}));
        for (u32 i = 0; i < 32; i++)
            if ((sph[0x36 + i / 2] >> ((i & 1) * 4)) & 0xF) output_generic(i);
    }
    if (is_fs) {
        u32 target = 0, extra = 0;
        memcpy(&target, sph + 0x48, 4);
        memcpy(&extra, sph + 0x4C, 4);
        for (u32 rt = 0; rt < 8; rt++) {
            if (((target >> (rt * 4)) & 0xF) == 0) continue;
            AttributeType at = env.render_target_type(rt);
            Id t = at == AttributeType::SignedInt ? t_v4s : at == AttributeType::UnsignedInt ? t_v4u : t_v4f;
            frag_out[rt] = io_var(spv::StorageClassOutput, t, int(rt), -1);
            prog.outputs_written |= 1u << rt;
        }
        if (extra & 2) {
            frag_depth = io_var(spv::StorageClassOutput, t_f32, -1, spv::BuiltInFragDepth);
            prog.writes_depth = true;
        }
        if (extra & 1) sample_mask = io_var(spv::StorageClassOutput, b.t_array(t_s32, 1), -1, spv::BuiltInSampleMask);
    }
}

void Translator::emit_epilogue() {
    if (is_vs && position) {
        Id p = b.load(t_v4f, position);
        {   // aspect correction for non-16:9 render targets (push constant viewport_scale: [0] perspective,
            // [1] orthographic). Both are 1.0 in normal rendering.
            Id vs = push_member(0, t_v4f);
            Id sxp = b.op(spv::OpCompositeExtract, t_f32, {vs, 0});
            Id sxo = b.op(spv::OpCompositeExtract, t_f32, {vs, 1});
            Id w = b.op(spv::OpCompositeExtract, t_f32, {p, 3});
            Id ortho = b.op(spv::OpFOrdEqual, t_bool, {w, fc(1.0f)});
            Id sx = b.op(spv::OpSelect, t_f32, {ortho, sxo, sxp});
            Id x = b.op(spv::OpCompositeExtract, t_f32, {p, 0});
            p = b.op(spv::OpCompositeInsert, t_v4f, {fop(spv::OpFMul, t_f32, {x, sx}), p, 0});
        }
        if (opt.flip_y) {
            Id y = b.op(spv::OpCompositeExtract, t_f32, {p, 1});
            p = b.op(spv::OpCompositeInsert, t_v4f, {b.op(spv::OpFNegate, t_f32, {y}), p, 1});
        }
        if (opt.convert_depth_mode) {
            Id z = b.op(spv::OpCompositeExtract, t_f32, {p, 2});
            Id w = b.op(spv::OpCompositeExtract, t_f32, {p, 3});
            Id nz = fop(spv::OpFMul, t_f32, {fop(spv::OpFAdd, t_f32, {z, w}), fc(0.5f)});
            p = b.op(spv::OpCompositeInsert, t_v4f, {nz, p, 2});
        }
        b.store(position, p);
    }
    if (is_fs) {
        u32 target = 0, extra = 0;
        memcpy(&target, sph + 0x48, 4);
        memcpy(&extra, sph + 0x4C, 4);
        u32 reg = 0;
        for (u32 rt = 0; rt < 8; rt++) {
            if (((target >> (rt * 4)) & 0xF) == 0) continue;
            Id c[4];
            AttributeType at = env.render_target_type(rt);
            for (u32 i = 0; i < 4; i++) c[i] = at == AttributeType::Float ? F(reg + i)
                                                : at == AttributeType::SignedInt ? b.bitcast(t_s32, R(reg + i))
                                                                                 : R(reg + i);
            Id t = at == AttributeType::SignedInt ? t_v4s : at == AttributeType::UnsignedInt ? t_v4u : t_v4f;
            b.store(frag_out[rt], b.op(spv::OpCompositeConstruct, t, {c[0], c[1], c[2], c[3]}));
            reg += 4;
        }
        if (sample_mask)
            b.store(b.op(spv::OpAccessChain, b.t_ptr(spv::StorageClassOutput, t_s32), {sample_mask, u(0)}),
                    b.bitcast(t_s32, R(reg)));
        if (frag_depth) b.store(frag_depth, F(reg + 1));
    }
}

void Translator::finish_program() {
    u32 set = 0;
    switch (stage) {
    case Stage::VertexA:
    case Stage::VertexB: set = 0; break;
    case Stage::TessControl: set = 1; break;
    case Stage::TessEval: set = 2; break;
    case Stage::Geometry: set = 3; break;
    case Stage::Fragment: set = 4; break;
    case Stage::Compute: set = 0; break;
    }
    u32 binding = 0;
    for (auto& [index, cb] : cbufs) {
        b.decorate(cb.var, spv::DecorationDescriptorSet, {set});
        b.decorate(cb.var, spv::DecorationBinding, {binding++});
        prog.resources.const_buffers.push_back({index, std::max<u32>(cb.size, 16)});
    }
    for (auto& t : textures) {
        b.decorate(t.var, spv::DecorationDescriptorSet, {set});
        b.decorate(t.var, spv::DecorationBinding, {binding++});
        TextureUse use{};
        use.type = t.type;
        use.is_depth = t.depth;
        use.cbuf_index = t.cbuf_index;
        use.cbuf_offset = t.cbuf_offset;
        use.count = 1;
        prog.resources.textures.push_back(use);
    }
    prog.uses_discard = uses_discard;

    spv::ExecutionModel model = spv::ExecutionModelGLCompute;
    switch (stage) {
    case Stage::VertexA:
    case Stage::VertexB: model = spv::ExecutionModelVertex; break;
    case Stage::TessControl: model = spv::ExecutionModelTessellationControl; break;
    case Stage::TessEval: model = spv::ExecutionModelTessellationEvaluation; break;
    case Stage::Geometry: model = spv::ExecutionModelGeometry; break;
    case Stage::Fragment: model = spv::ExecutionModelFragment; break;
    case Stage::Compute: model = spv::ExecutionModelGLCompute; break;
    }
    std::vector<u32> ep{u32(model), main_fn};
    SpvBuilder::put_str(ep, "main");
    for (Id v : io_vars) ep.push_back(v);
    b.put(b.entry_points, spv::OpEntryPoint, ep);
    if (is_fs) {
        b.put(b.exec_modes, spv::OpExecutionMode, {main_fn, spv::ExecutionModeOriginUpperLeft});
        if (frag_depth) b.put(b.exec_modes, spv::OpExecutionMode, {main_fn, spv::ExecutionModeDepthReplacing});
    }
    if (stage == Stage::Compute) {
        for (int i = 0; i < 3; i++) prog.local_size[i] = std::max<u32>(1, env.local_size(i));
        b.put(b.exec_modes, spv::OpExecutionMode,
              {main_fn, spv::ExecutionModeLocalSize, prog.local_size[0], prog.local_size[1], prog.local_size[2]});
        prog.shared_memory_size = env.shared_memory_size();
    }
    prog.local_memory_size = local_words * 4;
    prog.spirv = b.assemble();
}

}  // namespace shader
