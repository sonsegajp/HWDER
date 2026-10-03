# Owned by the gpu subsystem. Paths relative to the repo root.
list(APPEND RUNTIME_SOURCES
    runtime/src/gpu/vk_loader.cpp
    runtime/src/gpu/display.cpp
    runtime/src/gpu/settings.cpp
    runtime/src/gpu/overlay.cpp
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_vulkan.cpp
    third_party/imgui/backends/imgui_impl_win32.cpp
    runtime/src/gpu/services.cpp
    runtime/src/gpu/nvdrv.cpp
    runtime/src/gpu/gpu.cpp
    runtime/src/gpu/vi.cpp
)
