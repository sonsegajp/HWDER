// Memory, attribute, texture and warp instructions.
#include "shader/translator.h"

namespace shader {

// --------------------------------------------------------------------------------------- memory

static Id local_ptr(Translator& t, Id word) {
    return t.b.op(spv::OpAccessChain, t.b.t_ptr(spv::StorageClassFunction, t.t_u32), {t.local_mem, word});
}

void Translator::emit_mem(Inst in, Op op) {
    u32 d = in.bits(0, 8);
    switch (op) {
    case Op::LDC: {
        u32 size = in.bits(48, 3), mode = in.bits(44, 2), index = in.bits(36, 5), src = in.bits(8, 8);
        s32 off = in.sbits(20, 16);
        if (mode != 0) log_once("LDC mode " + std::to_string(mode) + " treated as default");
        if (src == 255) {
            u32 o = u32(off) & 0xFFFF;
            if (size == 5) {
                setR(d, cbuf_load(index, o));
                setR(d + 1, cbuf_load(index, o + 4));
                return;
            }
            Id v = cbuf_load(index, o);
            if (size < 4) v = bfe(v, u(0), u(size < 2 ? 8 : 16), size & 1);
            setR(d, v);
            return;
        }
        Id addr = iadd(R(src), u(u32(off)));
        if (size == 5) {
            setR(d, cbuf_load_dyn(index, addr));
            setR(d + 1, cbuf_load_dyn(index, iadd(addr, u(4))));
            return;
        }
        Id v = cbuf_load_dyn(index, addr);
        if (size < 4) {
            Id sh = op2(spv::OpIMul, t_u32, op2(spv::OpBitwiseAnd, t_u32, addr, u(3)), u(8));
            v = bfe(v, sh, u(size < 2 ? 8 : 16), size & 1);
        }
        setR(d, v);
        return;
    }
    case Op::ALD: {
        u32 n = in.bits(47, 2) + 1, offset = in.bits(20, 10);
        if (in.bits(8, 8) != 255 || in.bit(31) || in.bit(32)) {
            log_once("indexed/patch/output ALD unsupported");
            for (u32 e = 0; e < n; e++) setR(d + e, u(0));
            return;
        }
        for (u32 e = 0; e < n; e++) setR(d + e, load_attribute(offset + e * 4));
        return;
    }
    case Op::AST: {
        u32 n = in.bits(47, 2) + 1, offset = in.bits(20, 10);
        if (in.bits(8, 8) != 255 || in.bit(31)) {
            log_once("indexed/patch AST unsupported");
            return;
        }
        if (!is_vs) {
            log_once("AST outside vertex stage unsupported");
            return;
        }
        for (u32 e = 0; e < n; e++) store_attribute(offset + e * 4, R(d + e));
        return;
    }
    case Op::IPA:
        setF(d, ipa(in));
        return;
    case Op::S2R:
    case Op::CS2R: {
        u32 sr = in.bits(20, 8);
        Id v = u(0);
        auto builtin_v3 = [&](Id& var, spv::BuiltIn bi, u32 comp) {
            if (!var) var = io_var(spv::StorageClassInput, t_v3u, -1, bi);
            return b.load(t_u32, b.op(spv::OpAccessChain, b.t_ptr(spv::StorageClassInput, t_u32), {var, u(comp)}));
        };
        switch (sr) {
        case 0: v = lane_id(); break;
        case 18: v = u(0x3F800000); break;  // Y direction: +1.0
        case 29: v = u(0x00FF0000); break;
        case 32:
            if (stage == Stage::Compute) {
                Id x = builtin_v3(local_id, spv::BuiltInLocalInvocationId, 0);
                Id y = builtin_v3(local_id, spv::BuiltInLocalInvocationId, 1);
                Id z = builtin_v3(local_id, spv::BuiltInLocalInvocationId, 2);
                v = op2(spv::OpBitwiseOr, t_u32, x,
                        op2(spv::OpBitwiseOr, t_u32, op2(spv::OpShiftLeftLogical, t_u32, y, u(16)),
                            op2(spv::OpShiftLeftLogical, t_u32, z, u(26))));
            }
            break;
        case 33: case 34: case 35:
            if (stage == Stage::Compute) v = builtin_v3(local_id, spv::BuiltInLocalInvocationId, sr - 33);
            break;
        case 37: case 38: case 39:
            if (stage == Stage::Compute) v = builtin_v3(workgroup_id, spv::BuiltInWorkgroupId, sr - 37);
            break;
        default:
            log_once("S2R SR " + std::to_string(sr) + " reads 0");
        }
        setR(d, v);
        return;
    }
    case Op::LDL:
    case Op::STL: {
        if (!local_mem) {
            u32 lo = 0, hi = 0;
            memcpy(&lo, sph + 4, 4);
            memcpy(&hi, sph + 8, 4);
            u32 bytes = (lo & 0xFFFFFF) | ((hi & 0xFFFFFF) << 24);
            if (stage == Stage::Compute || bytes == 0) bytes = 0x1000;
            local_words = std::min<u32>(bytes / 4, 0x4000);
            local_mem = b.local_var(b.t_ptr(spv::StorageClassFunction, b.t_array(t_u32, local_words)));
        }
        u32 size = in.bits(48, 3);
        Id addr = iadd(R(in.bits(8, 8)), u(u32(in.sbits(20, 24))));
        Id word = glsl(t_u32, GLSLstd450UMin, {op2(spv::OpShiftRightLogical, t_u32, addr, u(2)), u(local_words - 4)});
        u32 n = size == 5 ? 2 : size == 6 ? 4 : 1;
        for (u32 e = 0; e < n; e++) {
            Id w = e ? iadd(word, u(e)) : word;
            if (op == Op::LDL) {
                Id v = b.load(t_u32, local_ptr(*this, w));
                if (size < 4) {
                    Id sh = op2(spv::OpIMul, t_u32, op2(spv::OpBitwiseAnd, t_u32, addr, u(3)), u(8));
                    v = bfe(v, sh, u(size < 2 ? 8 : 16), size & 1);
                }
                setR(d + e, v);
            } else {
                Id v = R(d + e);
                if (size < 4) {
                    Id old = b.load(t_u32, local_ptr(*this, w));
                    Id sh = op2(spv::OpIMul, t_u32, op2(spv::OpBitwiseAnd, t_u32, addr, u(3)), u(8));
                    v = b.op(spv::OpBitFieldInsert, t_u32, {old, v, sh, u(size < 2 ? 8 : 16)});
                }
                b.store(local_ptr(*this, w), v);
            }
        }
        return;
    }
    case Op::LDS:
    case Op::STS: {
        if (stage != Stage::Compute) {
            log_once("shared memory outside compute");
            if (op == Op::LDS) setR(d, u(0));
            return;
        }
        u32 words = std::max<u32>(1, std::min<u32>(env.shared_memory_size(), 0xC000) / 4);
        if (!shared_mem) {
            shared_mem = b.id();
            b.put(b.globals, spv::OpVariable,
                  {b.t_ptr(spv::StorageClassWorkgroup, b.t_array(t_u32, words)), shared_mem,
                   u32(spv::StorageClassWorkgroup)});
        }
        u32 size = in.bits(48, 3);
        Id addr = iadd(R(in.bits(8, 8)), u(u32(in.sbits(20, 24))));
        Id word = glsl(t_u32, GLSLstd450UMin, {op2(spv::OpShiftRightLogical, t_u32, addr, u(2)), u(words - 1)});
        u32 n = size == 5 ? 2 : size == 6 ? 4 : 1;
        Id pt = b.t_ptr(spv::StorageClassWorkgroup, t_u32);
        for (u32 e = 0; e < n; e++) {
            Id w = e ? glsl(t_u32, GLSLstd450UMin, {iadd(word, u(e)), u(words - 1)}) : word;
            Id p = b.op(spv::OpAccessChain, pt, {shared_mem, w});
            if (op == Op::LDS) setR(d + e, b.load(t_u32, p));
            else b.store(p, R(d + e));
        }
        return;
    }
    case Op::BAR:
        if (stage == Stage::Compute)
            b.op0(spv::OpControlBarrier, {u(spv::ScopeWorkgroup), u(spv::ScopeWorkgroup),
                                          u(spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask)});
        return;
    case Op::LD: case Op::LDG: case Op::ATOM: case Op::ATOMS: case Op::ATOM_cas: case Op::ATOMS_cas:
        unsupported(op);
        setR(d, u(0));
        return;
    default:
        unsupported(op);
    }
}

Id Translator::ipa(Inst in) {
    if (!is_fs) return fc(0);
    u32 addr = in.bits(30, 8) * 4;
    u32 mode = in.bits(54, 2), sample = in.bits(52, 2);
    if (in.bit(38) && in.bits(8, 8) != 255) {
        log_once("indexed IPA unsupported");
        return fc(0);
    }
    Id v = bc_f(load_attribute(addr));
    if (addr >= 0x80 && addr < 0x280) {
        u32 i = (addr - 0x80) >> 4;
        u8 m = sph[0x18 + i];
        u32 eff = 0;
        for (int c = 0; c < 4; c++)
            if ((m >> (c * 2)) & 3) {
                eff = (m >> (c * 2)) & 3;
                break;
            }
        if (eff == 2 || eff == 0) {
            if (!frag_coord) frag_coord = io_var(spv::StorageClassInput, t_v4f, -1, spv::BuiltInFragCoord);
            Id w = b.load(t_f32, b.op(spv::OpAccessChain, b.t_ptr(spv::StorageClassInput, t_f32), {frag_coord, u(3)}));
            v = fop(spv::OpFMul, t_f32, {v, w});
        }
        if (sample == 1 && !in_centroid[i] && in_generic[i]) {
            in_centroid[i] = true;
            b.decorate(in_generic[i], spv::DecorationCentroid);
        }
    }
    if (mode == 1) v = fop(spv::OpFMul, t_f32, {v, F(in.bits(20, 8))});
    if (in.bit(51)) v = fsat(v);
    return v;
}

// ------------------------------------------------------------------------------------- textures

namespace {
struct SampleArgs {
    Id coords = 0, lod = 0, bias = 0, dref = 0, offset = 0;
    bool explicit_lod = false;
};

// Returns vec4 (or float for depth compare).
Id sample(Translator& t, Translator::Tex& tex, SampleArgs a) {
    auto& b = t.b;
    Id img = b.load(tex.sampled_t, tex.var);
    if (!t.is_fs && !a.explicit_lod) {
        a.explicit_lod = true;
        a.lod = t.fc(0);
        a.bias = 0;
    }
    std::vector<u32> ops;
    u32 mask = 0;
    if (a.bias && !a.explicit_lod) mask |= spv::ImageOperandsBiasMask;
    if (a.explicit_lod) mask |= spv::ImageOperandsLodMask;
    if (a.offset) {
        mask |= spv::ImageOperandsOffsetMask;
        b.capability(spv::CapabilityImageGatherExtended);
    }
    std::vector<u32> args{img, a.coords};
    if (a.dref) args.push_back(a.dref);
    if (mask) {
        args.push_back(mask);
        if (mask & spv::ImageOperandsBiasMask) args.push_back(a.bias);
        if (mask & spv::ImageOperandsLodMask) args.push_back(a.lod ? a.lod : t.fc(0));
        if (mask & spv::ImageOperandsOffsetMask) args.push_back(a.offset);
    }
    spv::Op o = a.dref ? (a.explicit_lod ? spv::OpImageSampleDrefExplicitLod : spv::OpImageSampleDrefImplicitLod)
                       : (a.explicit_lod ? spv::OpImageSampleExplicitLod : spv::OpImageSampleImplicitLod);
    return b.op(o, a.dref ? t.t_f32 : t.t_v4f, args);
}

Id component(Translator& t, Id s, bool depth, u32 c) {
    if (depth) return c == 3 ? t.fc(1) : s;
    return t.b.op(spv::OpCompositeExtract, t.t_f32, {s, c});
}

Id array_layer(Translator& t, Id reg_bits) {
    return t.b.op(spv::OpConvertUToF, t.t_f32, {t.op2(spv::OpBitwiseAnd, t.t_u32, reg_bits, t.u(0xFFFF))});
}

TextureType tex_type(u32 ty) {
    switch (ty) {
    case 0: return TextureType::Tex1D;
    case 1: return TextureType::Tex1DArray;
    case 2: return TextureType::Tex2D;
    case 3: return TextureType::Tex2DArray;
    case 4: return TextureType::Tex3D;
    case 5: return TextureType::Tex3D;
    case 6: return TextureType::Cube;
    default: return TextureType::CubeArray;
    }
}

// Writes `n` floats either as 32-bit registers (a, a+1, b, b+1) or packed half pairs (TEXS-style).
void store_swizzled(Translator& t, u32 da, u32 db, std::vector<Id> vals, bool f16) {
    if (!f16) {
        u32 regs[4] = {da, da + 1, db, db + 1};
        for (size_t i = 0; i < vals.size(); i++) t.setF(regs[i], vals[i]);
        return;
    }
    auto pack = [&](Id lo, Id hi) {
        return t.glsl(t.t_u32, GLSLstd450PackHalf2x16, {t.b.op(spv::OpCompositeConstruct, t.t_v2f, {lo, hi})});
    };
    size_t n = vals.size();
    if (n == 0) return;
    t.setR(da, pack(vals[0], n > 1 ? vals[1] : t.fc(0)));
    if (n > 2) t.setR(db, pack(vals[2], n > 3 ? vals[3] : t.fc(0)));
}

const u32 RG_LUT[8] = {1, 2, 4, 8, 3, 9, 10, 12};
const u32 RGBA_LUT[5] = {7, 11, 13, 14, 15};
}  // namespace

void Translator::emit_tex(Inst in, Op op) {
    switch (op) {
    case Op::TEXS: {
        u32 enc = in.bits(53, 4);
        u32 ra = in.bits(8, 8), rb = in.bits(20, 8);
        u32 handle = in.bits(36, 13) * 4;
        bool f16 = !in.bit(59);
        SampleArgs a;
        TextureType ty = TextureType::Tex2D;
        bool depth = false;
        auto v2 = [&](Id x, Id y) { return b.op(spv::OpCompositeConstruct, t_v2f, {x, y}); };
        auto v3 = [&](Id x, Id y, Id z) { return b.op(spv::OpCompositeConstruct, t_v3f, {x, y, z}); };
        switch (enc) {
        case 0: ty = TextureType::Tex1D; a.coords = F(ra); a.explicit_lod = true; a.lod = fc(0); break;
        case 1: a.coords = v2(F(ra), F(rb)); break;
        case 2: a.coords = v2(F(ra), F(rb)); a.explicit_lod = true; a.lod = fc(0); break;
        case 3: a.coords = v2(F(ra), F(ra + 1)); a.explicit_lod = true; a.lod = F(rb); break;
        case 4: depth = true; a.coords = v2(F(ra), F(ra + 1)); a.dref = F(rb); break;
        case 5: depth = true; a.coords = v2(F(ra), F(ra + 1)); a.dref = F(rb + 1); a.explicit_lod = true; a.lod = F(rb); break;
        case 6: depth = true; a.coords = v2(F(ra), F(ra + 1)); a.dref = F(rb); a.explicit_lod = true; a.lod = fc(0); break;
        case 7: ty = TextureType::Tex2DArray; a.coords = v3(F(ra + 1), F(rb), array_layer(*this, R(ra))); break;
        case 8: ty = TextureType::Tex2DArray; a.coords = v3(F(ra + 1), F(rb), array_layer(*this, R(ra))); a.explicit_lod = true; a.lod = fc(0); break;
        case 9: ty = TextureType::Tex2DArray; depth = true; a.coords = v3(F(ra + 1), F(rb), array_layer(*this, R(ra))); a.dref = F(rb + 1); a.explicit_lod = true; a.lod = fc(0); break;
        case 10: ty = TextureType::Tex3D; a.coords = v3(F(ra), F(ra + 1), F(rb)); break;
        case 11: ty = TextureType::Tex3D; a.coords = v3(F(ra), F(ra + 1), F(rb)); a.explicit_lod = true; a.lod = fc(0); break;
        case 12: ty = TextureType::Cube; a.coords = v3(F(ra), F(ra + 1), F(rb)); break;
        case 13: ty = TextureType::Cube; a.coords = v3(F(ra), F(ra + 1), F(rb)); a.explicit_lod = true; a.lod = F(rb + 1); break;
        default: log_once("TEXS encoding " + std::to_string(enc)); return;
        }
        Tex& tex = texture(handle, ty, depth);
        Id s = sample(*this, tex, a);
        u32 swz = in.bits(50, 3);
        u32 db = in.bits(28, 8), da = in.bits(0, 8);
        u32 mask = db == 255 ? (swz < 8 ? RG_LUT[swz] : 1) : (swz < 5 ? RGBA_LUT[swz] : 15);
        std::vector<Id> vals;
        for (u32 c = 0; c < 4; c++)
            if ((mask >> c) & 1) vals.push_back(component(*this, s, depth, c));
        store_swizzled(*this, da, db, vals, f16);
        return;
    }
    case Op::TEX:
    case Op::TEX_b: {
        bool bindless = op == Op::TEX_b;
        u32 blod = bindless ? in.bits(37, 3) : in.bits(55, 3);
        bool aoffi = bindless ? in.bit(36) : in.bit(54);
        if (bindless) {
            log_once("bindless TEX unsupported");
            return;
        }
        u32 ty_bits = in.bits(28, 3);
        bool dc = in.bit(50);
        u32 creg = in.bits(8, 8), meta = in.bits(20, 8);
        SampleArgs a;
        auto vec = [&](std::vector<u32> parts) {
            return b.op(spv::OpCompositeConstruct, parts.size() == 2 ? t_v2f : parts.size() == 3 ? t_v3f : t_v4f, parts);
        };
        switch (ty_bits) {
        case 0: a.coords = F(creg); break;
        case 1: a.coords = vec({F(creg + 1), array_layer(*this, R(creg))}); break;
        case 2: a.coords = vec({F(creg), F(creg + 1)}); break;
        case 3: a.coords = vec({F(creg + 1), F(creg + 2), array_layer(*this, R(creg))}); break;
        case 4: case 5: a.coords = vec({F(creg), F(creg + 1), F(creg + 2)}); break;
        case 6: a.coords = vec({F(creg), F(creg + 1), F(creg + 2)}); break;
        case 7: a.coords = vec({F(creg + 1), F(creg + 2), F(creg + 3), array_layer(*this, R(creg))}); break;
        }
        u32 m = meta;
        switch (blod) {
        case 1: a.explicit_lod = true; a.lod = fc(0); break;  // LZ
        case 2: case 6: a.bias = F(m++); break;            // LB, LBA
        case 3: case 7: a.explicit_lod = true; a.lod = F(m++); break;  // LL, LLA
        default: break;
        }
        if (aoffi) {
            Id v = R(m++);
            auto sx = [&](u32 off) { return b.bitcast(t_s32, bfe(v, u(off), u(4), true)); };
            if (ty_bits <= 1) a.offset = sx(0);
            else if (ty_bits <= 3) a.offset = b.op(spv::OpCompositeConstruct, b.t_vec(t_s32, 2), {sx(0), sx(4)});
            else if (ty_bits <= 5) a.offset = b.op(spv::OpCompositeConstruct, b.t_vec(t_s32, 3), {sx(0), sx(4), sx(8)});
        }
        if (dc) a.dref = F(m++);
        Tex& tex = texture(in.bits(36, 13) * 4, tex_type(ty_bits), dc);
        Id s = sample(*this, tex, a);
        u32 mask = in.bits(31, 4);
        u32 dst = in.bits(0, 8);
        for (u32 c = 0; c < 4; c++)
            if ((mask >> c) & 1) setF(dst++, component(*this, s, dc, c));
        return;
    }
    default:
        unsupported(op);
    }
}

// ----------------------------------------------------------------------------------------- warp

void Translator::emit_warp(Inst in, Op op) {
    u32 d = in.bits(0, 8);
    switch (op) {
    case Op::SHFL: {
        bool idx_imm = in.bit(28), mask_imm = in.bit(29);
        u32 mode = in.bits(30, 2);
        u32 idx_v = in.bits(20, 5), mask_v = in.bits(34, 13);
        Id val = R(in.bits(8, 8));
        Id scope = u(spv::ScopeSubgroup);
        if (is_fs && idx_imm && mask_imm && mask_v == ((28u << 8) | 3u)) {
            b.capability(spv::CapabilityGroupNonUniformQuad);
            if (mode == 0 && idx_v <= 3) {
                setR(d, b.op(spv::OpGroupNonUniformQuadBroadcast, t_u32, {scope, val, u(idx_v)}));
                set_pred(in.bits(48, 3), b.c_bool(true));
                return;
            }
            if (mode == 3 && idx_v >= 1 && idx_v <= 3) {
                setR(d, b.op(spv::OpGroupNonUniformQuadSwap, t_u32, {scope, val, u(idx_v - 1)}));
                set_pred(in.bits(48, 3), b.c_bool(true));
                return;
            }
        }
        b.capability(spv::CapabilityGroupNonUniformShuffle);
        Id index = idx_imm ? u(idx_v) : R(in.bits(20, 8));
        Id lane = lane_id();
        Id id;
        switch (mode) {
        case 0: id = index; break;
        case 1: id = op2(spv::OpISub, t_u32, lane, index); break;
        case 2: id = iadd(lane, index); break;
        default: id = op2(spv::OpBitwiseXor, t_u32, lane, index); break;
        }
        (void)mask_imm;
        setR(d, b.op(spv::OpGroupNonUniformShuffle, t_u32, {scope, val, id}));
        set_pred(in.bits(48, 3), b.c_bool(true));
        return;
    }
    case Op::VOTE: {
        b.capability(spv::CapabilityGroupNonUniformVote);
        b.capability(spv::CapabilityGroupNonUniformBallot);
        Id p = pred(in.bits(39, 3), in.bit(42));
        Id scope = u(spv::ScopeSubgroup);
        u32 vop = in.bits(48, 2);
        Id r = vop == 0 ? b.op(spv::OpGroupNonUniformAll, t_bool, {scope, p})
             : vop == 1 ? b.op(spv::OpGroupNonUniformAny, t_bool, {scope, p})
                        : b.op(spv::OpGroupNonUniformAllEqual, t_bool, {scope, p});
        set_pred(in.bits(45, 3), r);
        Id ballot = b.op(spv::OpGroupNonUniformBallot, t_v4u, {scope, p});
        setR(d, b.op(spv::OpCompositeExtract, t_u32, {ballot, 0}));
        return;
    }
    default:
        unsupported(op);
    }
}

}  // namespace shader
