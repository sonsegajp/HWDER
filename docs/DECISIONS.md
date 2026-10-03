# HWDER design decisions

## 2026-10-02 — Static recompilation of `main`, SDK replaced by HLE (superseded)
ARM64 → C recompiler for the game's `main` NSO; the ~355 SDK imports implemented natively;
NVN translated directly to Vulkan. Reached the game's main render loop at 60 fps with a Vulkan
window, but NVN's undocumented semantics (descriptor pools, per-stage programs, queue state)
had to be discovered one bug at a time. That prototype is retired.

## 2026-10-03 — Emulator-style system layer, our own implementation
Run *all* of the game's code — `rtld`, `main`, `sdk` (incl. the real NVN driver), `subsdk0`
(glslc), `subsdk1` (multimedia) — as statically recompiled C, and provide the layers below it
ourselves:

* **Kernel**: Horizon SVCs implemented natively. Unlike an emulator there is no JIT run loop:
  every guest thread is a real host thread executing recompiled code, and an `svc` instruction
  is a direct C call. No fibers, no context marshalling, no fastmem fault slow paths.
* **Services**: sm, fsp-srv, hid, appletOE, vi, nvdrv, audren, set, time, acc, … over real
  HIPC/CMIF IPC, implemented from public documentation (switchbrew).
* **GPU**: nvdrv devices + a Maxwell (GM20B) command processor and a Vulkan renderer, with our
  own Maxwell → SPIR-V shader translator, texture cache (block-linear) and buffer cache.
* **Video**: NVDEC-equivalent H.264 decode for the game's pre-rendered movies.

Everything below the game is written from public documentation of the platform (SVC and IPC
interfaces, hardware register layouts) and from observing what the game itself does at runtime.
Goal: a better end product than an emulator — native
code everywhere, simpler threading, AOT shader translation from the game's own shader packs.
