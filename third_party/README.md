# Third-party code

Git submodules (`git submodule update --init --recursive`):

| Path | Source | License |
|---|---|---|
| imgui | https://github.com/ocornut/imgui | MIT |
| spirv-headers | https://github.com/KhronosGroup/SPIRV-Headers | MIT |
| vulkan-headers | https://github.com/KhronosGroup/Vulkan-Headers | Apache-2.0 / MIT |

Everything else in HWDE is our own code: the recompiler, the kernel and service layer, the
Maxwell command processor, the SASS to SPIR-V shader translator, the Vulkan renderer, the
texture cache, the audio renderer and the video path.
