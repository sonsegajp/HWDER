// Guest CPU contexts. Each guest thread owns one Ctx (registers); recompiled code keeps all guest
// state in it, so the host stack only carries host frames.
#include <cstdlib>
#include <cstring>

#include "hwder/runtime.h"

static thread_local Ctx* t_ctx;

Ctx* hw_ctx_new(u64 stack_top) {
    Ctx* c = (Ctx*)_aligned_malloc(sizeof(Ctx), 64);
    memset(c, 0, sizeof(Ctx));
    c->sp = stack_top & ~15ull;
    c->excl_addr = ~0ull;
    return c;
}

void hw_ctx_bind(Ctx* c) { t_ctx = c; }
Ctx* hw_ctx_current() { return t_ctx; }
