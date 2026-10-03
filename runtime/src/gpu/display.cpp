// Window + Vulkan device + swapchain presentation, with frame pacing.
#include "gpu/overlay.h"
#include "gpu/settings.h"
#include "display.h"

#include <timeapi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace display {

static Device g_dev;
Device& device() { return g_dev; }

static HWND g_hwnd;
static std::atomic<bool> g_window_ready{false}, g_resize{false};
static VkSurfaceKHR g_surface;
static VkSwapchainKHR g_swapchain;
static VkExtent2D g_extent;
static std::vector<VkImage> g_images;
static VkCommandPool g_pool;
static std::mutex g_status_lock;
static std::string g_status = "booting";
static std::atomic<u64> g_last_guest_frame{0};
static double g_present_ms = 0, g_limiter_ms = 0;  // accumulated per 300 frames

#define VKCHECK(x)                                                                   \
    do {                                                                             \
        VkResult r_ = (x);                                                           \
        if (r_ != VK_SUCCESS) {                                                      \
            hw_log("Vulkan error %d at %s:%d: %s", (int)r_, __FILE__, __LINE__, #x); \
            hw_fatal("vulkan");                                                      \
        }                                                                            \
    } while (0)

static u64 now_ms() { return GetTickCount64(); }

// ------------------------------------------------------------------ window
// Borderless fullscreen on the window's monitor (F11 / overlay option).
static void set_borderless(HWND h, bool on) {
    static WINDOWPLACEMENT prev = {sizeof(prev)};
    DWORD style = GetWindowLong(h, GWL_STYLE);
    bool is_on = !(style & WS_OVERLAPPEDWINDOW);
    if (on == is_on) return;
    if (on) {
        MONITORINFO mi = {sizeof(mi)};
        GetWindowPlacement(h, &prev);
        GetMonitorInfo(MonitorFromWindow(h, MONITOR_DEFAULTTOPRIMARY), &mi);
        SetWindowLong(h, GWL_STYLE, style & ~WS_OVERLAPPEDWINDOW);
        SetWindowPos(h, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                     mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    } else {
        SetWindowLong(h, GWL_STYLE, style | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(h, &prev);
        SetWindowPos(h, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    }
    settings::video().borderless = on;
}

// Temporary (CDS_FULLSCREEN) 60 Hz mode on the window's monitor; Windows restores it when we exit.
static void set_display_60hz(HWND h, bool on) {
    MONITORINFOEXW mi{};
    mi.cbSize = sizeof(mi);
    GetMonitorInfoW(MonitorFromWindow(h, MONITOR_DEFAULTTOPRIMARY), &mi);
    if (!on) {
        ChangeDisplaySettingsExW(mi.szDevice, nullptr, nullptr, 0, nullptr);
        hw_log("display: monitor refresh restored");
        return;
    }
    DEVMODEW cur{};
    cur.dmSize = sizeof(cur);
    EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &cur);
    if (cur.dmDisplayFrequency == 60) return;
    DEVMODEW dm = cur;
    dm.dmDisplayFrequency = 60;
    dm.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY | DM_BITSPERPEL;
    LONG r = ChangeDisplaySettingsExW(mi.szDevice, &dm, nullptr, CDS_FULLSCREEN, nullptr);
    hw_log("display: monitor %ls -> 60 Hz (%s)", mi.szDevice, r == DISP_CHANGE_SUCCESSFUL ? "ok" : "failed");
}

static void set_client_size(HWND h, int w, int h_) {
    RECT r{0, 0, w, h_};
    AdjustWindowRect(&r, GetWindowLong(h, GWL_STYLE), FALSE);
    SetWindowPos(h, nullptr, 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER);
}

static LRESULT CALLBACK wnd_proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (overlay::wndproc(h, msg, wp, lp)) return 0;
    switch (msg) {
    case WM_SIZE: g_resize = true; return 0;
    case WM_CLOSE: hw_log("window closed - exiting"); ExitProcess(0);
    case overlay::kMsgBorderless: set_borderless(h, lp != 0); return 0;
    case overlay::kMsgClientSize: set_client_size(h, (int)(lp >> 16), (int)(lp & 0xFFFF)); return 0;
    case overlay::kMsgDisplay60: set_display_60hz(h, lp != 0); return 0;
    case WM_KEYDOWN:
        if (wp == VK_F1) overlay::toggle();
        if (wp == VK_F11) {
            set_borderless(h, !settings::video().borderless);
            settings::save();
        }
        return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

static void window_thread() {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    WNDCLASSEXW wc = {sizeof(wc)};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"HWDER";
    RegisterClassExW(&wc);
    RECT r = {0, 0, 1280, 720};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    g_hwnd = CreateWindowExW(0, L"HWDER", L"Hyrule Warriors: Definitive Edition (HWDER)", WS_OVERLAPPEDWINDOW,
                             CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, nullptr, nullptr,
                             wc.hInstance, nullptr);
    // Automated test runs (they set HWDER_LOG) open minimized without stealing focus.
    ShowWindow(g_hwnd, getenv("HWDER_LOG") && !getenv("HWDER_SHOW") ? SW_SHOWMINNOACTIVE : SW_SHOW);
    g_window_ready = true;
    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
}

// ------------------------------------------------------------------ Vulkan
static bool has_ext(const std::vector<VkExtensionProperties>& v, const char* n) {
    for (auto& e : v)
        if (!strcmp(e.extensionName, n)) return true;
    return false;
}

static void create_swapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_dev.phys, g_surface, &caps);
    VkExtent2D ext = caps.currentExtent;
    if (ext.width == 0xFFFFFFFF) ext = {1280, 720};
    if (ext.width == 0 || ext.height == 0) return;
    u32 n = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_dev.phys, g_surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_dev.phys, g_surface, &n, modes.data());
    // Our frame limiter paces to the game's rate; MAILBOX avoids double pacing and adds no latency.
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end()) mode = VK_PRESENT_MODE_MAILBOX_KHR;
    if (settings::video().vsync) mode = VK_PRESENT_MODE_FIFO_KHR;
    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = g_surface;
    sci.minImageCount = std::max(3u, caps.minImageCount);
    if (caps.maxImageCount) sci.minImageCount = std::min(sci.minImageCount, caps.maxImageCount);
    sci.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    sci.imageExtent = ext;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = mode;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = g_swapchain;
    VkSwapchainKHR sc;
    VKCHECK(vkCreateSwapchainKHR(g_dev.dev, &sci, nullptr, &sc));
    if (g_swapchain) vkDestroySwapchainKHR(g_dev.dev, g_swapchain, nullptr);
    g_swapchain = sc;
    g_extent = ext;
    vkGetSwapchainImagesKHR(g_dev.dev, sc, &n, nullptr);
    g_images.resize(n);
    vkGetSwapchainImagesKHR(g_dev.dev, sc, &n, g_images.data());
    overlay::on_swapchain(g_images, sci.imageFormat);
}

static void init_vulkan() {
    if (!hw_vk_load_global()) hw_fatal("vulkan-1.dll not found - a Vulkan capable GPU driver is required");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "HWDER";
    app.apiVersion = VK_API_VERSION_1_3;
    const char* inst_ext[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = inst_ext;
    const char* layer = "VK_LAYER_KHRONOS_validation";
    if (getenv("HWDER_VALIDATION")) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer;
    }
    VKCHECK(vkCreateInstance(&ici, nullptr, &g_dev.instance));
    hw_vk_load_instance(g_dev.instance);
    u32 n = 0;
    vkEnumeratePhysicalDevices(g_dev.instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(g_dev.instance, &n, pds.data());
    for (auto pd : pds) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd, &p);
        if (!g_dev.phys || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            g_dev.phys = pd;
            g_dev.props = p;
        }
    }
    if (!g_dev.phys) hw_fatal("no Vulkan device");
    vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &g_dev.mem);
    hw_log("vulkan: %s (api %u.%u)", g_dev.props.deviceName, VK_API_VERSION_MAJOR(g_dev.props.apiVersion),
           VK_API_VERSION_MINOR(g_dev.props.apiVersion));
    VkWin32SurfaceCreateInfoKHR wsci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    wsci.hinstance = GetModuleHandleW(nullptr);
    wsci.hwnd = g_hwnd;
    VKCHECK(vkCreateWin32SurfaceKHR(g_dev.instance, &wsci, nullptr, &g_surface));
    vkGetPhysicalDeviceQueueFamilyProperties(g_dev.phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(n);
    vkGetPhysicalDeviceQueueFamilyProperties(g_dev.phys, &n, qfs.data());
    for (u32 i = 0; i < n; i++) {
        VkBool32 present = 0;
        vkGetPhysicalDeviceSurfaceSupportKHR(g_dev.phys, i, g_surface, &present);
        if ((qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && present) {
            g_dev.queue_family = i;
            break;
        }
    }
    vkEnumerateDeviceExtensionProperties(g_dev.phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(g_dev.phys, nullptr, &n, exts.data());
    std::vector<const char*> dev_ext = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
    for (const char* opt : {VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME,
                            VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME})
        if (has_ext(exts, opt)) dev_ext.push_back(opt);
    // Two queues when available: presentation must never sit in front of renderer work (a present
    // waiting for a swapchain image would otherwise block the fences the game waits on).
    u32 nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_dev.phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfp(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(g_dev.phys, &nqf, qfp.data());
    u32 qcount = std::min(2u, qfp[g_dev.queue_family].queueCount);
    float prio[2] = {1.0f, 1.0f};
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_dev.queue_family;
    qci.queueCount = qcount;
    qci.pQueuePriorities = prio;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    f13.maintenance4 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.shaderInt8 = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    f12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    f12.scalarBlockLayout = VK_TRUE;
    f12.shaderFloat16 = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    f11.storageBuffer16BitAccess = VK_TRUE;
    f11.uniformAndStorageBuffer16BitAccess = VK_TRUE;
    f11.shaderDrawParameters = VK_TRUE;
    VkPhysicalDeviceFeatures2 feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    VkPhysicalDeviceFeatures2 avail{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    vkGetPhysicalDeviceFeatures2(g_dev.phys, &avail);
    feat.features = avail.features;
    feat.features.robustBufferAccess = VK_FALSE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &feat};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (u32)dev_ext.size();
    dci.ppEnabledExtensionNames = dev_ext.data();
    VKCHECK(vkCreateDevice(g_dev.phys, &dci, nullptr, &g_dev.dev));
    hw_vk_load_device(g_dev.dev);
    vkGetDeviceQueue(g_dev.dev, g_dev.queue_family, 0, &g_dev.queue);
    vkGetDeviceQueue(g_dev.dev, g_dev.queue_family, qcount > 1 ? 1 : 0, &g_dev.present_queue);
    if (qcount > 1) hw_log("display: presentation uses a separate queue");
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = g_dev.queue_family;
    VKCHECK(vkCreateCommandPool(g_dev.dev, &cpi, nullptr, &g_pool));
    create_swapchain();
}

// ------------------------------------------------------------------ presentation
struct FrameSync {
    VkSemaphore acquired = VK_NULL_HANDLE, rendered = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
};
static FrameSync g_frames[3];
static u32 g_frame_idx;
static std::mutex g_present_lock;
static double g_trace_fence_ms, g_trace_acq_ms;  // HWDER_PACE_TRACE: last present's fence wait / acquire time
static double g_trace_overshoot_ms;               // HWDER_PACE_TRACE: last limiter sleep's wake-up lateness

static double g_output_hz = 0;
void set_output_hz(double hz) { g_output_hz = hz; }
void client_size(unsigned* w, unsigned* h) {
    *w = g_extent.width;
    *h = g_extent.height;
}
double monitor_hz() {
    DEVMODEW dm{};
    dm.dmSize = sizeof(dm);
    if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &dm) && dm.dmDisplayFrequency > 1) return (double)dm.dmDisplayFrequency;
    return 60.0;
}
static void frame_limit(u32 interval) {
    static LARGE_INTEGER freq, next;
    static HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    const char* env = getenv("HWDER_FPS");
    // The presenter runs at a fixed 60 Hz like the console's display. A swap interval of 2 means the
    // frame stays up for two vsyncs: pace such a present two slots ahead instead of re-targeting the
    // limiter to 30 fps (the title screen alternates 1/2 every second, which made that stutter).
    // The game is frame-locked (its logic advances per released frame), so the release grid stays at
    // 60 Hz whatever the display runs at; HWDER_FPS exists only for experiments.
    double fps = env ? atof(env) : (g_output_hz > 0 ? g_output_hz : 60.0);
    static u32 last_interval = 0;
    if (interval != last_interval) {
        last_interval = interval;
        hw_log("display: swap interval %u", interval);
    }
    u32 slots = g_output_hz > 0 ? 1 : std::max(1u, std::min(interval, 4u));
    // Experiment: release every frame at the 60 Hz grid even when the game asks for interval 2.
    static const bool force1 = getenv("HWDER_FORCE_INTERVAL1") != nullptr;
    if (force1) slots = 1;
    if (fps <= 0) return;
    // The compositor's timer wake-ups must not be delayed by other busy threads (a late wake-up shows
    // up directly as a late frame followed by a short one).
    static bool prio = false;
    if (!prio) {
        prio = true;
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    s64 period = (s64)(freq.QuadPart / fps);
    // `next` is the virtual vblank grid. In steady state we arrive here a little after the grid point
    // (the game produced its frame right after we released the previous buffer). After a real stall
    // (debt beyond a quarter period) re-anchor the grid to now instead of letting several presents
    // burst through at zero wait: a display never shows frames faster than its refresh, and the frame
    // after a hitch should again stay up for a full period.
    g_trace_overshoot_ms = 0;
    if (!next.QuadPart || now.QuadPart - next.QuadPart > period / 4) {
        // Late frame: show it immediately (never add a full period of latency to a frame that is
        // already late - when the game runs slower than the grid that would cost 16 ms on EVERY
        // frame) but anchor the grid so the following frame still stays up for a whole period.
        next.QuadPart = now.QuadPart + period * slots;
        return;
    }
    next.QuadPart += period * slots;
    s64 wait = next.QuadPart - now.QuadPart;
    if (wait <= 0) return;
    s64 sleep_100ns = wait * 10000000 / freq.QuadPart - 10000;
    if (sleep_100ns > 0 && timer) {
        LARGE_INTEGER due;
        due.QuadPart = -sleep_100ns;
        SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0);
        WaitForSingleObject(timer, INFINITE);
        LARGE_INTEGER woke;
        QueryPerformanceCounter(&woke);
        // How late the timer woke us relative to the requested due time (pacing diagnostics).
        g_trace_overshoot_ms = ((woke.QuadPart - now.QuadPart) * 1000.0 / freq.QuadPart) - sleep_100ns / 10000.0;
    }
    do QueryPerformanceCounter(&now);
    while (now.QuadPart < next.QuadPart);
}

static void update_title(bool guest) {
    static u64 frames, last = now_ms();
    frames++;
    u64 t = now_ms();
    if (t - last < 500) return;
    double fps = frames * 1000.0 / (t - last);
    frames = 0;
    last = t;
    std::string status;
    {
        std::lock_guard<std::mutex> g(g_status_lock);
        status = g_status;
    }
    wchar_t title[256];
    if (guest) swprintf(title, 256, L"Hyrule Warriors: Definitive Edition (HWDER) - %.1f fps", fps);
    else swprintf(title, 256, L"Hyrule Warriors: Definitive Edition (HWDER) - %hs", status.c_str());
    SetWindowTextW(g_hwnd, title);
}

// Record + submit one frame: clear, optionally blit `src`, present.
static std::mutex g_present_queue_lock;

static void present(VkImage src, u32 w, u32 h, const VkClearColorValue& bg, VkSemaphore wait_sem, u64 wait_value) {
    std::lock_guard<std::mutex> pl(g_present_lock);
    if (overlay::swapchain_dirty()) g_resize = true;
    if (g_resize.exchange(false)) {
        std::lock_guard<std::mutex> q(g_dev.queue_lock);
        std::lock_guard<std::mutex> q2(g_present_queue_lock);
        vkDeviceWaitIdle(g_dev.dev);
        create_swapchain();
    }
    if (!g_swapchain || !g_extent.width) return;
    FrameSync& fs = g_frames[g_frame_idx];
    g_frame_idx = (g_frame_idx + 1) % 3;
    if (!fs.fence) {
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(g_dev.dev, &sci, nullptr, &fs.acquired);
        vkCreateSemaphore(g_dev.dev, &sci, nullptr, &fs.rendered);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(g_dev.dev, &fci, nullptr, &fs.fence);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = g_pool;
        ai.commandBufferCount = 1;
        vkAllocateCommandBuffers(g_dev.dev, &ai, &fs.cb);
    }
    LARGE_INTEGER tf, ta, tb, tc;
    QueryPerformanceFrequency(&tf);
    QueryPerformanceCounter(&ta);
    vkWaitForFences(g_dev.dev, 1, &fs.fence, VK_TRUE, UINT64_MAX);
    QueryPerformanceCounter(&tb);
    u32 idx;
    // Bounded: a hidden/minimized window may never hand out an image; the game must not stall on it.
    VkResult r = vkAcquireNextImageKHR(g_dev.dev, g_swapchain, 100000000ull, fs.acquired, VK_NULL_HANDLE, &idx);
    QueryPerformanceCounter(&tc);
    g_trace_fence_ms = (tb.QuadPart - ta.QuadPart) * 1000.0 / tf.QuadPart;
    g_trace_acq_ms = (tc.QuadPart - tb.QuadPart) * 1000.0 / tf.QuadPart;
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        g_resize = true;
        return;
    }
    if (r == VK_TIMEOUT || r == VK_NOT_READY) return;
    vkResetFences(g_dev.dev, 1, &fs.fence);
    vkResetCommandBuffer(fs.cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(fs.cb, &bi);
    VkImage dst = g_images[idx];
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = dst;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(fs.cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    VkImageSubresourceRange rr{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(fs.cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &bg, 1, &rr);
    if (src) {
        float s = std::min((float)g_extent.width / w, (float)g_extent.height / h);
        s32 dw = (s32)(w * s), dh = (s32)(h * s);
        s32 x0 = ((s32)g_extent.width - dw) / 2, y0 = ((s32)g_extent.height - dh) / 2;
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {(s32)w, (s32)h, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[0] = {x0, y0, 0};
        blit.dstOffsets[1] = {x0 + dw, y0 + dh, 1};
        vkCmdBlitImage(fs.cb, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                       VK_FILTER_LINEAR);
    }
    // overlay (F1 menu / frame-time readout) on top of the game image
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(fs.cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &b);
    overlay::render(fs.cb, dst, g_extent);
    b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = 0;
    vkCmdPipelineBarrier(fs.cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &b);
    vkEndCommandBuffer(fs.cb);
    VkPipelineStageFlags ws[2] = {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
    VkSemaphore waits[2] = {fs.acquired, wait_sem};
    u64 wait_values[2] = {0, wait_value};
    VkTimelineSemaphoreSubmitInfo tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    tsi.waitSemaphoreValueCount = wait_sem ? 2 : 1;
    tsi.pWaitSemaphoreValues = wait_values;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, &tsi};
    si.waitSemaphoreCount = wait_sem ? 2 : 1;
    si.pWaitSemaphores = waits;
    si.pWaitDstStageMask = ws;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &fs.cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &fs.rendered;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &fs.rendered;
    pi.swapchainCount = 1;
    pi.pSwapchains = &g_swapchain;
    pi.pImageIndices = &idx;
    bool shared = g_dev.present_queue == g_dev.queue;
    std::unique_lock<std::mutex> q(g_dev.queue_lock, std::defer_lock);
    std::unique_lock<std::mutex> q2(g_present_queue_lock, std::defer_lock);
    if (shared) q.lock();
    else q2.lock();
    VKCHECK(vkQueueSubmit(g_dev.present_queue, 1, &si, fs.fence));
    r = vkQueuePresentKHR(g_dev.present_queue, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) g_resize = true;
}

void present_image(VkImage image, u32 width, u32 height, u32 swap_interval, VkSemaphore wait_sem, u64 wait_value) {
    {
        // Frame pacing statistics: interval between guest presents, logged every 300 frames.
        static LARGE_INTEGER freq, last;
        static double sum = 0, worst = 0;
        static u32 n = 0, slow = 0;
        if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        if (last.QuadPart) {
            double ms = (now.QuadPart - last.QuadPart) * 1000.0 / freq.QuadPart;
            sum += ms;
            worst = std::max(worst, ms);
            if (ms > 20.0) slow++;
            if (ms > 100.0) hw_log("display: slow frame (%.0f ms since the previous present)", ms);
            if (++n == 300) {
                hw_log("display: 300 frames: avg %.2f ms, worst %.1f ms, %u frames over 20 ms | per frame: present %.1f ms, limiter %.1f ms",
                       sum / n, worst, slow, g_present_ms / n, g_limiter_ms / n);
                sum = worst = 0;
                n = slow = 0;
                g_present_ms = g_limiter_ms = 0;
            }
        }
        last = now;
    }
    g_last_guest_frame = now_ms();
    VkClearColorValue black{};
    LARGE_INTEGER f, t0, t1, t2;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    // Wait for the slot first, then present: the caller's work for the next output (frame
    // interpolation replays) overlaps the wait instead of delaying the present.
    frame_limit(swap_interval);
    QueryPerformanceCounter(&t1);
    present(image, width, height, black, wait_sem, wait_value);
    update_title(true);
    QueryPerformanceCounter(&t2);
    g_limiter_ms += (t1.QuadPart - t0.QuadPart) * 1000.0 / f.QuadPart;
    g_present_ms += (t2.QuadPart - t1.QuadPart) * 1000.0 / f.QuadPart;
    // HWDER_PACE_TRACE=1: one line per present (interval since the previous present call, requested swap
    // interval, fence wait / acquire / whole present() / limiter time) for frame pacing analysis.
    static const bool trace = getenv("HWDER_PACE_TRACE") != nullptr;
    if (trace) {
        static LARGE_INTEGER prev;
        double dt = prev.QuadPart ? (t0.QuadPart - prev.QuadPart) * 1000.0 / f.QuadPart : 0;
        prev = t0;
        hw_log("pace: present dt %.2f iv %u fence %.2f acq %.2f present %.2f limit %.2f overshoot %.2f", dt, swap_interval,
               g_trace_fence_ms, g_trace_acq_ms, (t1.QuadPart - t0.QuadPart) * 1000.0 / f.QuadPart,
               (t2.QuadPart - t1.QuadPart) * 1000.0 / f.QuadPart, g_trace_overshoot_ms);
    }
}

void set_status(const std::string& s) {
    std::lock_guard<std::mutex> g(g_status_lock);
    g_status = s;
}

// Boot screen: shown until the game presents its first frame.
static void boot_screen_thread() {
    for (;;) {
        // Only before the first guest frame: afterwards a stall (loading, shader compiles) keeps the
        // last game frame on screen instead of flashing back to the boot colour.
        if (g_last_guest_frame.load() == 0 && now_ms() - g_last_guest_frame > 1000) {
            float t = (float)(now_ms() % 4000) / 4000.0f;
            float pulse = 0.5f + 0.5f * sinf(t * 6.2831853f);
            VkClearColorValue c{};
            c.float32[0] = 0.02f + 0.03f * pulse;
            c.float32[1] = 0.05f + 0.05f * pulse;
            c.float32[2] = 0.10f + 0.08f * pulse;
            c.float32[3] = 1.0f;
            present(VK_NULL_HANDLE, 0, 0, c, VK_NULL_HANDLE, 0);
            update_title(false);
        }
        Sleep(16);
    }
}

void init() {
    // Frame pacing relies on ~1 ms timer accuracy. Request it, and opt out of Windows 11's timer
    // coarsening for processes whose window is not in the foreground / minimized (test runs, or the
    // user alt-tabbing away while the game keeps running).
    timeBeginPeriod(1);
#ifdef PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
    PROCESS_POWER_THROTTLING_STATE pt{};
    pt.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    pt.ControlMask = PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
    pt.StateMask = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof(pt));
#endif
    settings::load();
    std::thread(window_thread).detach();
    while (!g_window_ready) Sleep(1);
    init_vulkan();
    overlay::init(g_hwnd, g_dev.instance, g_dev.phys, g_dev.dev, g_dev.queue_family, g_dev.present_queue, VK_FORMAT_B8G8R8A8_UNORM);
    if (!g_images.empty()) overlay::on_swapchain(g_images, VK_FORMAT_B8G8R8A8_UNORM);
    if (settings::video().display_60hz) PostMessageW(g_hwnd, overlay::kMsgDisplay60, 0, 1);
    if (settings::video().borderless) PostMessageW(g_hwnd, overlay::kMsgBorderless, 0, 1);
    else if (settings::video().window_w && settings::video().window_h)
        PostMessageW(g_hwnd, overlay::kMsgClientSize, 0, (LPARAM)((settings::video().window_w << 16) | settings::video().window_h));
    std::thread(boot_screen_thread).detach();
    hw_log("display: window open");
}

}  // namespace display
