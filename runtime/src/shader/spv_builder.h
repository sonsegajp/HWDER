// Minimal SPIR-V module builder (HWDER's own). Types and constants are de-duplicated; the module is
// assembled from sections in the order required by the SPIR-V logical layout.
#pragma once
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <spirv/unified1/spirv.hpp>
#include <spirv/unified1/GLSL.std.450.h>

#include "hwder/runtime.h"

namespace shader {

using Id = u32;

class SpvBuilder {
public:
    std::vector<u32> capabilities, extensions, ext_imports, memory_model, entry_points, exec_modes, debug,
        annotations, globals, functions;
    u32 bound = 1;
    Id glsl = 0;

    Id id() { return bound++; }

    static void put(std::vector<u32>& s, spv::Op op, const std::vector<u32>& args) {
        s.push_back(u32(args.size() + 1) << 16 | u32(op));
        s.insert(s.end(), args.begin(), args.end());
    }
    static void put_str(std::vector<u32>& v, const char* str) {
        size_t n = strlen(str) + 1;
        size_t w = (n + 3) / 4;
        size_t base = v.size();
        v.resize(base + w, 0);
        memcpy(&v[base], str, n);
    }

    void capability(spv::Capability c) {
        for (size_t i = 0; i < capabilities.size(); i += 2)
            if (capabilities[i + 1] == u32(c)) return;
        put(capabilities, spv::OpCapability, {u32(c)});
    }
    void extension(const char* name) {
        std::vector<u32> a;
        put_str(a, name);
        for (auto& e : ext_names) if (e == name) return;
        ext_names.push_back(name);
        put(extensions, spv::OpExtension, a);
    }
    Id import_glsl() {
        if (!glsl) {
            glsl = id();
            std::vector<u32> a{glsl};
            put_str(a, "GLSL.std.450");
            put(ext_imports, spv::OpExtInstImport, a);
        }
        return glsl;
    }
    void name(Id target, const char* n) {
        std::vector<u32> a{target};
        put_str(a, n);
        put(debug, spv::OpName, a);
    }
    void decorate(Id target, spv::Decoration d, std::vector<u32> extra = {}) {
        std::vector<u32> a{target, u32(d)};
        a.insert(a.end(), extra.begin(), extra.end());
        put(annotations, spv::OpDecorate, a);
    }
    void member_decorate(Id target, u32 member, spv::Decoration d, std::vector<u32> extra = {}) {
        std::vector<u32> a{target, member, u32(d)};
        a.insert(a.end(), extra.begin(), extra.end());
        put(annotations, spv::OpMemberDecorate, a);
    }

    // ---- types (deduplicated)
    Id type(spv::Op op, std::vector<u32> args) {
        std::vector<u32> key{u32(op)};
        key.insert(key.end(), args.begin(), args.end());
        auto it = type_cache.find(key);
        if (it != type_cache.end()) return it->second;
        Id r = id();
        std::vector<u32> a{r};
        a.insert(a.end(), args.begin(), args.end());
        put(globals, op, a);
        type_cache[key] = r;
        return r;
    }
    Id t_void() { return type(spv::OpTypeVoid, {}); }
    Id t_bool() { return type(spv::OpTypeBool, {}); }
    Id t_uint() { return type(spv::OpTypeInt, {32, 0}); }
    Id t_int() { return type(spv::OpTypeInt, {32, 1}); }
    Id t_float() { return type(spv::OpTypeFloat, {32}); }
    Id t_vec(Id comp, u32 n) { return n == 1 ? comp : type(spv::OpTypeVector, {comp, n}); }
    Id t_ptr(spv::StorageClass sc, Id t) { return type(spv::OpTypePointer, {u32(sc), t}); }
    Id t_func(Id ret, std::vector<Id> params = {}) {
        params.insert(params.begin(), ret);
        return type(spv::OpTypeFunction, params);
    }
    Id t_array(Id elem, u32 len) { return type(spv::OpTypeArray, {elem, c_uint(len)}); }
    // Fresh (non-deduplicated) aggregate types, for types that carry decorations.
    Id t_array_fresh(Id elem, u32 len) {
        Id r = id();
        put(globals, spv::OpTypeArray, {r, elem, c_uint(len)});
        return r;
    }
    Id t_runtime_array_fresh(Id elem) {
        Id r = id();
        put(globals, spv::OpTypeRuntimeArray, {r, elem});
        return r;
    }
    Id t_struct_fresh(std::vector<Id> members) {
        Id r = id();
        members.insert(members.begin(), r);
        put(globals, spv::OpTypeStruct, members);
        return r;
    }

    // ---- constants (deduplicated)
    Id constant(Id t, u32 bits) {
        auto key = (u64(t) << 32) | bits;
        auto it = const_cache.find(key);
        if (it != const_cache.end()) return it->second;
        Id r = id();
        put(globals, spv::OpConstant, {t, r, bits});
        const_cache[key] = r;
        return r;
    }
    Id c_uint(u32 v) { return constant(t_uint(), v); }
    Id c_int(s32 v) { return constant(t_int(), u32(v)); }
    Id c_float(float f) {
        u32 b;
        memcpy(&b, &f, 4);
        return constant(t_float(), b);
    }
    Id c_bool(bool b) {
        Id& slot = b ? c_true : c_false;
        if (!slot) {
            slot = id();
            put(globals, b ? spv::OpConstantTrue : spv::OpConstantFalse, {t_bool(), slot});
        }
        return slot;
    }
    Id c_composite(Id t, std::vector<Id> parts) {
        std::vector<u32> key{0xFFFFFFFFu, t};
        key.insert(key.end(), parts.begin(), parts.end());
        auto it = type_cache.find(key);
        if (it != type_cache.end()) return it->second;
        Id r = id();
        parts.insert(parts.begin(), r);
        parts.insert(parts.begin(), t);
        put(globals, spv::OpConstantComposite, parts);
        type_cache[key] = r;
        return r;
    }
    Id c_null(Id t) {
        std::vector<u32> key{0xFFFFFFFEu, t};
        auto it = type_cache.find(key);
        if (it != type_cache.end()) return it->second;
        Id r = id();
        put(globals, spv::OpConstantNull, {t, r});
        type_cache[key] = r;
        return r;
    }

    // ---- global variables
    Id global_var(Id ptr_type, spv::StorageClass sc, Id init = 0) {
        Id r = id();
        if (init)
            put(globals, spv::OpVariable, {ptr_type, r, u32(sc), init});
        else
            put(globals, spv::OpVariable, {ptr_type, r, u32(sc)});
        interface.push_back(r);
        return r;
    }

    // ---- function emission. Function-scope variables are hoisted to the entry block.
    std::vector<u32> fn_vars, fn_body;
    bool block_open = false;
    Id cur_fn_type = 0, cur_fn = 0;

    Id begin_function(Id ret, Id fn_type, std::vector<std::pair<Id, Id>>* params = nullptr) {
        cur_fn = id();
        put(functions, spv::OpFunction, {ret, cur_fn, 0, fn_type});
        if (params)
            for (auto& p : *params) {
                p.second = id();
                put(functions, spv::OpFunctionParameter, {p.first, p.second});
            }
        fn_vars.clear();
        fn_body.clear();
        Id entry = id();
        put(functions, spv::OpLabel, {entry});
        block_open = true;
        return cur_fn;
    }
    void end_function() {
        if (block_open) put(fn_body, spv::OpUnreachable, {});
        functions.insert(functions.end(), fn_vars.begin(), fn_vars.end());
        functions.insert(functions.end(), fn_body.begin(), fn_body.end());
        put(functions, spv::OpFunctionEnd, {});
        block_open = false;
    }
    Id local_var(Id ptr_type, Id init = 0) {
        Id r = id();
        if (init)
            put(fn_vars, spv::OpVariable, {ptr_type, r, u32(spv::StorageClassFunction), init});
        else
            put(fn_vars, spv::OpVariable, {ptr_type, r, u32(spv::StorageClassFunction)});
        return r;
    }
    // Emit an instruction with a result.
    Id op(spv::Op o, Id rt, std::vector<u32> args) {
        if (!block_open) start_dead_block();
        Id r = id();
        args.insert(args.begin(), r);
        args.insert(args.begin(), rt);
        put(fn_body, o, args);
        return r;
    }
    // Emit an instruction without a result.
    void op0(spv::Op o, std::vector<u32> args) {
        if (!block_open) start_dead_block();
        put(fn_body, o, args);
    }
    void label(Id l) {
        if (block_open) put(fn_body, spv::OpBranch, {l});
        put(fn_body, spv::OpLabel, {l});
        block_open = true;
    }
    void terminate(spv::Op o, std::vector<u32> args = {}) {
        if (!block_open) start_dead_block();
        put(fn_body, o, args);
        block_open = false;
    }
    void branch(Id l) { terminate(spv::OpBranch, {l}); }
    void start_dead_block() {
        put(fn_body, spv::OpLabel, {id()});
        block_open = true;
    }

    // Convenience
    Id load(Id t, Id ptr) { return op(spv::OpLoad, t, {ptr}); }
    void store(Id ptr, Id v) { op0(spv::OpStore, {ptr, v}); }
    Id ext(Id t, u32 inst, std::vector<u32> args) {
        args.insert(args.begin(), inst);
        args.insert(args.begin(), import_glsl());
        return op(spv::OpExtInst, t, args);
    }
    Id bitcast(Id t, Id v) { return op(spv::OpBitcast, t, {v}); }

    std::vector<Id> interface;

    std::vector<u32> assemble(u32 version = 0x10300) {
        std::vector<u32> out{spv::MagicNumber, version, 0x48574452 /* 'HWDR' generator */, bound, 0};
        for (auto* s : {&capabilities, &extensions, &ext_imports, &memory_model, &entry_points, &exec_modes, &debug,
                        &annotations, &globals, &functions})
            out.insert(out.end(), s->begin(), s->end());
        return out;
    }

private:
    std::map<std::vector<u32>, Id> type_cache;
    std::map<u64, Id> const_cache;
    std::vector<std::string> ext_names;
    Id c_true = 0, c_false = 0;
};

}  // namespace shader
