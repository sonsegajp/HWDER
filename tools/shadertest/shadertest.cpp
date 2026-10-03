// Offline harness for the Maxwell -> SPIR-V translator.
//   shadertest hist <dir>            opcode histogram over every program in <dir>
//   shadertest run <dir> <outdir>    translate every program, write <outdir>/<name>.spv
//   shadertest one <file.bin> <out.spv>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "shader/decode.h"
#include "shader/shader.h"

extern "C" void hw_log(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}
extern "C" void hw_fatal(const char* why) {
    fprintf(stderr, "FATAL %s\n", why);
    exit(1);
}

namespace fs = std::filesystem;

struct FileEnv : shader::Environment {
    std::vector<u8> data;
    u64 read_instruction(u32 address) override {
        size_t o = 0x50 + size_t(address);
        if (o + 8 > data.size()) return 0;
        u64 v;
        memcpy(&v, &data[o], 8);
        return v;
    }
    const u8* sph() override { return data.data(); }
    u32 read_cbuf(u32, u32) override { return 0; }
    shader::TextureType texture_type(u32) override { return shader::TextureType::Tex2D; }
    u32 texture_bound_buffer() override { return 2; }
};

static bool load(const fs::path& p, FileEnv& e) {
    FILE* f = _wfopen(p.c_str(), L"rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    e.data.resize(n);
    fread(e.data.data(), 1, n, f);
    fclose(f);
    return n >= 0x58;
}

static shader::Stage stage_of(const fs::path& p) {
    std::string s = p.stem().extension().string();
    if (s == ".vs") return shader::Stage::VertexB;
    if (s == ".fs") return shader::Stage::Fragment;
    if (s == ".gs") return shader::Stage::Geometry;
    if (s == ".tcs") return shader::Stage::TessControl;
    if (s == ".tes") return shader::Stage::TessEval;
    return shader::Stage::Compute;
}

static void write_spv(const fs::path& out, const std::vector<u32>& spv) {
    FILE* f = _wfopen(out.c_str(), L"wb");
    if (!f) return;
    fwrite(spv.data(), 4, spv.size(), f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: shadertest hist|run|one ...\n");
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "one") {
        FileEnv e;
        if (!load(argv[2], e)) return 1;
        shader::Options opt;
        opt.convert_depth_mode = true;
        auto prog = shader::translate(e, stage_of(argv[2]), opt);
        write_spv(argv[3], prog.spirv);
        printf("%zu words\n", prog.spirv.size());
        return 0;
    }
    std::vector<fs::path> files;
    for (auto& de : fs::directory_iterator(argv[2]))
        if (de.path().extension() == ".bin") files.push_back(de.path());
    if (mode == "hist") {
        std::map<std::string, std::pair<int, int>> h;  // name -> (instances, programs)
        for (auto& p : files) {
            FileEnv e;
            if (!load(p, e)) continue;
            std::map<std::string, int> local;
            size_t n = (e.data.size() - 0x50) / 8;
            for (size_t i = 0; i < n; i++) {
                if (i % 4 == 0) continue;
                u64 w = e.read_instruction(u32(i * 8));
                local[shader::op_name(shader::decode(w))]++;
            }
            for (auto& [k, v] : local) {
                h[k].first += v;
                h[k].second++;
            }
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k, c] : h) v.push_back({c.first, k});
        std::sort(v.rbegin(), v.rend());
        for (auto& [c, k] : v) printf("%-12s %8d %5d\n", k.c_str(), c, h[k].second);
        return 0;
    }
    if (mode == "run") {
        fs::path out = argv[3];
        fs::create_directories(out);
        int n = 0;
        for (auto& p : files) {
            FileEnv e;
            if (!load(p, e)) continue;
            shader::Options opt;
            opt.convert_depth_mode = true;
            auto prog = shader::translate(e, stage_of(p), opt);
            write_spv(out / (p.stem().string() + ".spv"), prog.spirv);
            n++;
        }
        printf("translated %d programs\n", n);
        return 0;
    }
    return 1;
}
