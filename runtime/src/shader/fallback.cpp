// Pass-through programs used when translation fails (a draw is never skipped because of the translator).
#include "shader/internal.h"

namespace shader {

Program make_fallback(Stage stage) {
    Program p;
    p.stage = stage;
    SpvBuilder b;
    b.capability(spv::CapabilityShader);
    b.import_glsl();
    b.put(b.memory_model, spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
    Id f = b.t_float(), v4 = b.t_vec(f, 4);
    Id fn_t = b.t_func(b.t_void());
    std::vector<Id> io;
    spv::ExecutionModel model = spv::ExecutionModelGLCompute;
    Id in = 0, out = 0;
    if (stage == Stage::VertexA || stage == Stage::VertexB) {
        model = spv::ExecutionModelVertex;
        in = b.global_var(b.t_ptr(spv::StorageClassInput, v4), spv::StorageClassInput);
        b.decorate(in, spv::DecorationLocation, {0});
        out = b.global_var(b.t_ptr(spv::StorageClassOutput, v4), spv::StorageClassOutput);
        b.decorate(out, spv::DecorationBuiltIn, {spv::BuiltInPosition});
        p.attributes_read = 1;
    } else if (stage == Stage::Fragment) {
        model = spv::ExecutionModelFragment;
        out = b.global_var(b.t_ptr(spv::StorageClassOutput, v4), spv::StorageClassOutput);
        b.decorate(out, spv::DecorationLocation, {0});
        p.outputs_written = 1;
    } else if (stage == Stage::Geometry) {
        model = spv::ExecutionModelGeometry;
    } else if (stage == Stage::TessControl) {
        model = spv::ExecutionModelTessellationControl;
    } else if (stage == Stage::TessEval) {
        model = spv::ExecutionModelTessellationEvaluation;
    }
    Id main = b.begin_function(b.t_void(), fn_t);
    if (model == spv::ExecutionModelVertex) {
        b.store(out, b.load(v4, in));
    } else if (model == spv::ExecutionModelFragment) {
        b.store(out, b.c_composite(v4, {b.c_float(1), b.c_float(0), b.c_float(1), b.c_float(1)}));
    }
    b.terminate(spv::OpReturn);
    b.end_function();
    std::vector<u32> ep{u32(model), main};
    SpvBuilder::put_str(ep, "main");
    if (model == spv::ExecutionModelVertex || model == spv::ExecutionModelFragment)
        for (Id v : b.interface) ep.push_back(v);
    b.put(b.entry_points, spv::OpEntryPoint, ep);
    if (model == spv::ExecutionModelFragment)
        b.put(b.exec_modes, spv::OpExecutionMode, {main, spv::ExecutionModeOriginUpperLeft});
    if (model == spv::ExecutionModelGLCompute)
        b.put(b.exec_modes, spv::OpExecutionMode, {main, spv::ExecutionModeLocalSize, 1, 1, 1});
    if (model == spv::ExecutionModelGeometry) {
        b.capability(spv::CapabilityGeometry);
        b.put(b.exec_modes, spv::OpExecutionMode, {main, spv::ExecutionModeInputPoints});
        b.put(b.exec_modes, spv::OpExecutionMode, {main, spv::ExecutionModeOutputPoints});
        b.put(b.exec_modes, spv::OpExecutionMode, {main, spv::ExecutionModeOutputVertices, 1});
        b.put(b.exec_modes, spv::OpExecutionMode, {main, spv::ExecutionModeInvocations, 1});
    }
    if (model == spv::ExecutionModelTessellationControl) {
        b.capability(spv::CapabilityTessellation);
        b.put(b.exec_modes, spv::OpExecutionMode, {main, spv::ExecutionModeOutputVertices, 1});
    }
    if (model == spv::ExecutionModelTessellationEvaluation) {
        b.capability(spv::CapabilityTessellation);
        b.put(b.exec_modes, spv::OpExecutionMode, {main, spv::ExecutionModeTriangles});
    }
    p.spirv = b.assemble();
    return p;
}

}  // namespace shader
