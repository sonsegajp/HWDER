#include "vk_loader.h"

#define HW_VK_DEFINE(fn) PFN_##fn fn;
HW_VK_GLOBAL_FUNCS(HW_VK_DEFINE)
HW_VK_INSTANCE_FUNCS(HW_VK_DEFINE)
HW_VK_DEVICE_FUNCS(HW_VK_DEFINE)
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

bool hw_vk_load_global() {
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) return false;
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(lib, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) return false;
#define LOAD(fn) fn = (PFN_##fn)vkGetInstanceProcAddr(nullptr, #fn);
    HW_VK_GLOBAL_FUNCS(LOAD)
#undef LOAD
    return vkCreateInstance != nullptr;
}

void hw_vk_load_instance(VkInstance inst) {
#define LOAD(fn) fn = (PFN_##fn)vkGetInstanceProcAddr(inst, #fn);
    HW_VK_INSTANCE_FUNCS(LOAD)
#undef LOAD
}

void hw_vk_load_device(VkDevice dev) {
#define LOAD(fn) fn = (PFN_##fn)vkGetDeviceProcAddr(dev, #fn);
    HW_VK_DEVICE_FUNCS(LOAD)
#undef LOAD
}
