#include "shader/decode.h"

#include <mutex>

namespace shader {

namespace {
struct Pattern {
    Op op;
    const char* name;
    const char* bits;
};
const Pattern kPatterns[] = {
#define X(n, p) {Op::n, #n, p},
#include "shader/opcodes.inc"
#undef X
};

Op g_table[65536];
std::once_flag g_once;

void build_table() {
    u8 best[65536];
    for (int i = 0; i < 65536; i++) {
        g_table[i] = Op::INVALID;
        best[i] = 0;
    }
    for (const Pattern& p : kPatterns) {
        u16 mask = 0, value = 0;
        int fixed = 0, n = 0;
        for (const char* c = p.bits; *c; c++) {
            if (*c == ' ') continue;
            mask <<= 1;
            value <<= 1;
            if (*c != '-') {
                mask |= 1;
                value |= (*c == '1');
                fixed++;
            }
            n++;
        }
        for (int i = 0; i < 65536; i++)
            if ((i & mask) == value && fixed > best[i]) {
                best[i] = u8(fixed);
                g_table[i] = p.op;
            }
    }
}
}  // namespace

Op decode(u64 inst) {
    std::call_once(g_once, build_table);
    return g_table[inst >> 48];
}

const char* op_name(Op op) {
    for (const Pattern& p : kPatterns)
        if (p.op == op) return p.name;
    return "INVALID";
}

}  // namespace shader
