# HWDER - Hyrule Warriors: Definitive Edition Recomp

A static recompilation of *Hyrule Warriors: Definitive Edition* (Nintendo Switch) to a native
Windows x64 executable. The game's ARM64 code is translated ahead of time to C and compiled with
clang; everything the game expects underneath it (the Horizon kernel, system services, the GPU,
audio, video decoding, the file system) is implemented by our own runtime, written for this
purpose. There is no CPU emulation, no JIT and no emulator core. The result is one `hwder.exe`
that reads code and assets straight out of the player's own game dump at startup.

**This repository contains no game code and no game assets.** The recompiled C in `generated/`
is produced on your machine from your own dump, and the runtime decrypts the ExeFS and RomFS out
of your own `.nsp` with your own `prod.keys` at run time. Nothing is extracted to disk.

---

## Playing (release build)

1. Download the latest zip from Releases and unpack it anywhere.
2. Put your own dump of the game (`*.nsp`, title `0100AE00096EA000`, base game) and your
   console's `prod.keys` next to `hwder.exe`. Keys are also picked up from
   `%USERPROFILE%\.switch\prod.keys` and the yuzu / Ryujinx key folders.
3. Run `hwder.exe`.

Requirements: Windows 10/11 x64, a Vulkan 1.2 driver (NVIDIA, AMD, Intel), an AVX2 CPU, and
Media Foundation (built into Windows; "N" editions need the Media Feature Pack) for the movies.

In game: **F1** opens the options overlay (internal resolution scale, unlocked resolution for
ultrawide, vsync, borderless, anisotropic filtering, FPS counter, 60 Hz monitor switch, and a
save-file cheats tab). **F11** toggles borderless fullscreen. Any XInput controller works;
keyboard: WASD move, arrows camera, K/Enter = A, J/Backspace = B, I = X, L = Y, Q/E = L/R,
U/O = ZL/ZR, Space = +, Tab = -, 1-4 = D-pad, Z/C = stick clicks.

Files created beside the exe: `hwder_settings.ini`, `hwder.log` (attach to bug reports),
`save\` (console save layout), `cache\` and `pipeline_cache.bin` (safe to delete).

---

## Building from source

Toolchain: Visual Studio 2022 with the **LLVM (clang-cl)** component, CMake and Ninja (both ship
with VS), Python 3.10+ with `cryptography` (only for the extraction helper).

```
git clone --recursive https://github.com/sonsegajp/HWDER
cd HWDER

# 1. extract your dump once for the recompiler (exefs/ is all it needs; romfs/ is optional,
#    the runtime can read it from the NSP directly)
python tools/nx_extract.py "<game>.nsp" "<prod.keys>" data --no-romfs

# 2. ARM64 -> C (all five ExeFS modules, ~5 minutes, writes generated/)
cd tools && python -m recomp.gen_main && cd ..

# 3. build (clang-cl + Ninja inside the VS developer environment)
python tools/build.py

# 4. run: drop your .nsp + prod.keys next to build/release/hwder.exe, or point at a dump
set HWDER_EXEFS=data\exefs
set HWDER_ROMFS=data\romfs
build\release\hwder.exe

# package: dist/HWDER-<date>.zip (exe, pdb, README, default settings)
python tools/make_dist.py
```

If the game ever jumps to an address the recompiler did not know was code, the runtime logs it,
appends it to `extra_entries.txt` and exits; `tools/iterate.sh` automates the
regenerate-rebuild-rerun loop. The current entry list already covers normal play.

---

## How it works

### 1. Static recompilation (`tools/recomp`, `tools/nso.py`)

`nso.py` loads the five NSO modules of the ExeFS (`rtld`, `main`, `subsdk0`, `subsdk1`, `sdk`),
decompresses their LZ4 segments and applies the dynamic relocations, giving a fully linked
in-memory image of the process exactly as the loader on the console would have produced it.

`recomp/gen.py` then discovers functions in each module from several independent sources:
`.eh_frame` unwind entries, the entry point and init/fini arrays, exported dynamic symbols,
direct `BL` targets, tail-call targets, absolute code pointers found in relocated data, and
`ADRP+ADD` address materialisations. Inside each function it finds the intra-function branch
targets and recovers jump tables (the compiler's `ADR` + `LDRB/LDRH` + `ADD` + `BR` idiom), so
`switch` statements turn into real C `switch`es rather than indirect dispatch.

`recomp/a64.py` and `a64_simd.py` translate every AArch64 instruction (base integer, FP, the
full Advanced SIMD set, the AES/SHA/CRC extensions, saturating and narrowing arithmetic) into
C that operates on a guest register file `Ctx` (`runtime/include/hwder/cpu.h`). The translator
reaches 100% of the instructions in all five modules; anything it has no pattern for would
emit a call to a runtime hook that logs and halts, which never fires in practice.

Each guest function becomes one C function `f_<address>(Ctx*)`. Direct calls become direct C
calls. Indirect branches (`BLR`, `BR`, returns through unusual paths) go through a dense
address-to-function table built for every module's text segment, and the dispatcher uses
clang's `[[clang::musttail]]` so tail calls and returns cost a jump, not a growing host stack.
Functions that contain an unresolvable indirect branch are emitted as *resumable*: they carry a
label at every instruction and can be entered mid-function, which is how the game's hand-written
assembly and longjmp-style control flow keep working.

The generated program is split into `rc_NNNN.c` files of about 12k lines each so the build
parallelises. `module_info.c` records each module's load address and segment layout; at run
time `hwder.exe` maps the real NSO segments from the player's dump to those same addresses, so
the recompiled code and the original data (rodata, vtables, string tables, relocated pointers)
line up byte for byte.

### 2. The kernel (`runtime/src/kernel`)

Guest threads are real Windows threads. `svcCreateThread` creates a host thread whose stack
and TLS live in the guest address space, and the thread simply calls the recompiled entry
function. An `svc` instruction in the original code is a direct C call into `svc.cpp`, which
implements the Horizon system calls the game and its SDK use: memory management (`SetHeapSize`,
`MapMemory`, `QueryMemory` with the exact page attribute model), threads and their priorities,
core masks and the `GetCurrentProcessorNumber` rule the SDK's job system relies on, all the
synchronisation primitives (`WaitSynchronization`, `ArbitrateLock/Unlock`, condition variables,
`WaitForAddress`/`SignalToAddress`), events, transfer memory, sessions and IPC, and `GetInfo`.

Guest memory is a flat reservation at the addresses the console would use (code at `0xF_FFE0_0000`
and up, a heap region, stack and TLS regions), so a guest pointer is a host pointer. There is no
page table emulation and no fault-based fastmem: the game runs on the host MMU directly.

### 3. Services and IPC (`runtime/src/service`, `fs`, `hid`, `audio`, `video`, `gpu`)

The SDK talks to system services over HIPC. `ipc.cpp` implements the real message format
(CMIF headers, descriptors for buffers, handles and domains), so the unmodified SDK code in the
`sdk` module marshals requests exactly as on hardware and our services unmarshal them. Services
implemented: `sm`, `fsp-srv` (RomFS storage, save data, SD card), `hid` (shared memory input
state fed from XInput/keyboard), `appletOE`/`applet` (focus, operation mode, performance),
`vi` + `nvnflinger` (display layers and the buffer queue), `nvdrv` (the GPU driver interface),
`audren` (the audio renderer), `set`, `time`, `acc`, `pctl`, `psm`, `nifm`, `ns`, `mm`, `lm`
and friends.

**File system.** `fs/nsp.cpp` reads the game straight out of the `.nsp`: it parses the PFS0
container, decrypts the program NCA header (AES-XTS, `header_key`), derives the content key
from the key area (or from the title's ticket when rights-ID crypto is used) and serves the
ExeFS partition and the RomFS section (IVFC level 5) through on-demand AES-CTR decryption with
Windows CNG. The SDK's own RomFS driver then parses that image through the `IStorage`
interface, as it would on the console. An extracted `exefs/` + `romfs/` pair works too: `romfs.cpp`
synthesises a RomFS image (header, hash tables, directory and file tables) over the host files
so the same driver path is used. Save data maps to `save\<title>\<save id>\` on the host.

**Audio.** `audio/audio.cpp` implements the `audren` renderer: it parses the per-frame update
(voices, mixes, splitters, effects, sinks, wave buffers), decodes PCM16 and ADPCM voices with
pitch resampling, routes them through the mix graph into the sink and plays the result with
WASAPI in shared mode at 48 kHz.

**Video.** The game's cutscenes are H.264 streams fed to the console's NVDEC through `nvdrv`.
`video/host1x.cpp` and `decoder.cpp` reconstruct the bitstream (SPS/PPS synthesised from the
decoder registers, slices re-wrapped as Annex-B), decode it with Media Foundation in low-latency
mode, and `vic` composition converts the result into the surface format the game expects.

### 4. The GPU (`runtime/src/gpu`, `shader`)

The game renders through NVN, which on hardware is a user-space driver that writes Maxwell
(GM20B) command buffers and submits them through `nvhost-gpu`. Because the real NVN driver in
the `sdk` module is recompiled along with everything else, we never had to reverse NVN itself.
Instead `gpu.cpp` is a Maxwell command processor: it executes GPFIFO entries, parses pushbuffer
methods for the 3D, compute, 2D, DMA copy and inline-to-memory engines, runs the macro
interpreter for the driver's uploaded macros, keeps the full register state, and turns draws,
clears, copies and dispatches into work for the renderer.

`shader/` is our SASS to SPIR-V translator. Maxwell shaders are decoded from the game's
precompiled shader binaries (`decode.cpp`, `opcodes.inc`), a control-flow graph is built
(`translate.cpp`), and `emit_alu.cpp` / `emit_mem.cpp` lower the ISA to SPIR-V through a small
builder (`spv_builder.h`) with the attribute, constant buffer, texture and image bindings
mapped from the pipeline state. `program.cpp` assembles a complete pipeline program and adds
the epilogue that applies our viewport scaling.

`gpu/vk/vk_renderer.cpp` is the Vulkan backend: pipeline state is derived from the Maxwell
registers (vertex attribute formats, blending, depth/stencil, culling with the Maxwell window
origin rule, render target formats), pipelines are created asynchronously and cached on disk,
and the draw path uses several caches (program state, texture descriptors, index ranges, vertex
buffer slices, redundant state elimination) to stay around 25 microseconds per draw.
`texture_cache.cpp` manages guest textures and render targets: block-linear deswizzling, format
conversion (including ASTC decoding with a disk cache), aliasing between differently sized
views of the same memory, CPU-write tracking and GPU-write ordering so that post-processing
chains read what the game last wrote. The same cache implements the **internal resolution
scaler**: render targets are allocated at a scaled physical size while the guest keeps seeing
its native dimensions, with viewports, scissors, blits and copies rescaled per axis. That is
also what makes **ultrawide** work: the targets are scaled non-uniformly to the window and the
projection is corrected in the shader epilogue, so 3D fills the screen while the 2D UI stays
correctly proportioned.

`display.cpp` owns the window and the swapchain, presents at a fixed 60 Hz with a precise
limiter (the game's logic is frame-locked, so presentation never changes its speed), and hosts
the Dear ImGui overlay (`overlay.cpp`, `settings.cpp`).

### 5. Why native, not emulated

* No instruction decoding at run time and no JIT warm-up: the whole game is C compiled with
  `-O2 /arch:AVX2`.
* Guest threads are host threads and SVCs are function calls, so synchronisation is cheap and
  the scheduler is the OS scheduler.
* Guest memory is host memory: no page-table lookups, no fault handling on the hot path.
* The shader translator runs ahead of time on the game's own shader packs and its output is
  cached, so there is no in-game stutter from shader discovery after the first run.

---

## Repository layout

```
tools/recomp/      ARM64 -> C recompiler (a64.py, a64_simd.py, gen.py, gen_main.py)
tools/nso.py       NSO loader / relocator
tools/nx_extract.py  NSP -> exefs/ romfs/ extraction (for the recompiler's input)
tools/build.py     clang-cl + Ninja build driver;  make_dist.py: release zip;  iterate.sh: entry-point loop
runtime/include/   Ctx (guest register file), dispatch and module interfaces used by generated code
runtime/src/       core (loader, dispatch, crash reporting), kernel/, service/, fs/, hid/, audio/, video/, gpu/, shader/
third_party/       imgui, Vulkan and SPIR-V headers (submodules)
docs/              design decisions
```

## Environment variables (debugging)

`HWDER_NSP`, `HWDER_KEYS`, `HWDER_EXEFS`, `HWDER_ROMFS` select the game data; `HWDER_LOG` the
log file; `HWDER_MUTE=1` silences audio; `HWDER_RES_SCALE` overrides the resolution scale;
`HWDER_VALIDATION=1` enables Vulkan validation layers; `HWDER_VK_FRAME_DUMP`, `HWDER_GPU_TRACE`,
`HWDER_IPC_TRACE`, `HWDER_FS_TRACE` and the other `HWDER_*_TRACE` switches dump the respective
subsystem's activity to the log.

## Status

Boots, menus, movies, Adventure/Legend mode battles, saves and audio work; gameplay runs at the
game's native 60 Hz (30 Hz where the game itself drops to half rate) at native or scaled
resolution. Known issues are tracked in the issue tracker.

## License

MIT for everything in this repository (see `LICENSE`). Hyrule Warriors: Definitive Edition is
the property of Nintendo and Koei Tecmo; you need your own copy of the game.
