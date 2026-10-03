// Window, Vulkan device and presentation.
#pragma once
#include <atomic>
#include <mutex>
#include <string>

#include "hwder/runtime.h"
#include "vk_loader.h"

namespace display {

struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;          // renderer queue
    VkQueue present_queue = VK_NULL_HANDLE;  // presentation (same as queue when the family has one queue)
    u32 queue_family = 0;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem{};
    std::mutex queue_lock;
};
Device& device();

// Opens the window and initializes Vulkan; starts the presenter (boot screen until frames arrive).
void init();

// Present a guest frame (an image in GENERAL layout). Paced to the guest swap interval.
// wait_sem/wait_value: timeline semaphore the presentation blit waits on (the renderer's last submit).
void present_image(VkImage image, u32 width, u32 height, u32 swap_interval, VkSemaphore wait_sem = VK_NULL_HANDLE,
                   u64 wait_value = 0);

// Status shown in the title bar while booting.
void set_status(const std::string& s);
// Frame interpolation: pace presents at `hz` (0 = the 60 Hz game grid with swap-interval slots).
void set_output_hz(double hz);
void client_size(unsigned* w, unsigned* h);  // current swapchain / window client size (0 if none yet)
double monitor_hz();

}  // namespace display
