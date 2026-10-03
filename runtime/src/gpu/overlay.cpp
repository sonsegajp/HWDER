#include "gpu/overlay.h"

#include "gpu/settings.h"
#include "hwder/runtime.h"

#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "imgui_impl_win32.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <filesystem>
#include <mutex>
#include <unordered_map>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

namespace overlay {

// Recursive: ImGui's Win32 handler calls ReleaseCapture()/SetCursor(), which re-enter wnd_proc synchronously.
static std::recursive_mutex g_m;
static bool g_inited = false;
static bool g_visible = false;
static bool g_swapchain_dirty = false;
static HWND g_hwnd;
static VkDevice g_dev;
static VkInstance g_instance;
static VkFormat g_fmt;
static std::unordered_map<VkImage, VkImageView> g_views;
static double g_frame_ms[120];
static int g_frame_i = 0;
static LARGE_INTEGER g_last_t;

static void check_vk(VkResult r) {
    if (r != VK_SUCCESS) hw_log("overlay: vulkan error %d", (int)r);
}

void init(HWND hwnd, VkInstance instance, VkPhysicalDevice phys, VkDevice dev, uint32_t queue_family, VkQueue queue,
          VkFormat swapchain_format) {
    std::lock_guard<std::recursive_mutex> l(g_m);
    g_hwnd = hwnd;
    g_dev = dev;
    g_instance = instance;
    g_fmt = swapchain_format;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // window positions are not worth a file
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 6.0f;
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplVulkan_LoadFunctions(
        VK_API_VERSION_1_3,
        [](const char* name, void* ud) { return vkGetInstanceProcAddr((VkInstance)ud, name); }, instance);
    ImGui_ImplVulkan_InitInfo ii{};
    ii.ApiVersion = VK_API_VERSION_1_3;
    ii.Instance = instance;
    ii.PhysicalDevice = phys;
    ii.Device = dev;
    ii.QueueFamily = queue_family;
    ii.Queue = queue;
    ii.DescriptorPoolSize = 16;
    ii.MinImageCount = 2;
    ii.ImageCount = 3;
    ii.UseDynamicRendering = true;
    ii.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &g_fmt;
    ii.CheckVkResultFn = check_vk;
    if (!ImGui_ImplVulkan_Init(&ii)) {
        hw_log("overlay: ImGui Vulkan backend init failed - overlay disabled");
        return;
    }
    g_inited = true;
    g_visible = getenv("HWDER_OVERLAY") != nullptr;
    QueryPerformanceCounter(&g_last_t);
    hw_log("overlay: ready (F1 opens the video options)");
}

bool wndproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (!g_inited) return false;
    std::lock_guard<std::recursive_mutex> l(g_m);
    ImGui_ImplWin32_WndProcHandler(h, msg, wp, lp);
    if (!g_visible) return false;
    switch (msg) {
    case WM_MOUSEMOVE: case WM_LBUTTONDOWN: case WM_LBUTTONUP: case WM_RBUTTONDOWN: case WM_RBUTTONUP:
    case WM_MOUSEWHEEL: case WM_LBUTTONDBLCLK: case WM_CHAR: case WM_KEYUP:
        return true;
    case WM_KEYDOWN:
        return wp != VK_F1 && wp != VK_F11;
    }
    return false;
}

void toggle() {
    std::lock_guard<std::recursive_mutex> l(g_m);
    g_visible = !g_visible;
}
bool visible() { return g_visible; }
bool swapchain_dirty() {
    bool d = g_swapchain_dirty;
    g_swapchain_dirty = false;
    return d;
}

void on_swapchain(const std::vector<VkImage>& images, VkFormat fmt) {
    std::lock_guard<std::recursive_mutex> l(g_m);
    if (!g_dev) return;  // swapchain created before the overlay exists: display re-registers it after init()
    for (auto& [img, view] : g_views) vkDestroyImageView(g_dev, view, nullptr);
    g_views.clear();
    g_fmt = fmt;
    for (VkImage img : images) {
        VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ci.image = img;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = fmt;
        ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView v = VK_NULL_HANDLE;
        if (vkCreateImageView(g_dev, &ci, nullptr, &v) == VK_SUCCESS) g_views[img] = v;
    }
}

static void post_window_change() {
    settings::Video& v = settings::video();
    PostMessageW(g_hwnd, kMsgBorderless, 0, v.borderless ? 1 : 0);
    if (!v.borderless && v.window_w && v.window_h) PostMessageW(g_hwnd, kMsgClientSize, 0, (LPARAM)((v.window_w << 16) | v.window_h));
}

static void build_ui(VkExtent2D extent) {
    settings::Video& v = settings::video();
    bool changed = false;
    ImGuiIO& io = ImGui::GetIO();

    if (v.show_fps) {
        double sum = 0, worst = 0;
        for (double f : g_frame_ms) sum += f, worst = f > worst ? f : worst;
        double avg = sum / 120.0;
        ImGui::SetNextWindowPos(ImVec2(8, 8));
        ImGui::SetNextWindowBgAlpha(0.45f);
        ImGui::Begin("##fps", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs |
                         ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
        ImGui::Text("%.0f fps  %.1f ms (worst %.0f)", avg > 0 ? 1000.0 / avg : 0.0, avg, worst);
        ImGui::End();
    }
    if (!g_visible) return;

    ImGui::SetNextWindowPos(ImVec2(extent.width * 0.5f, extent.height * 0.5f), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(520, 0), ImGuiCond_Appearing);
    ImGui::Begin("HWDER - Video Options   (F1 closes)", &g_visible, ImGuiWindowFlags_NoCollapse);

    ImGui::SeparatorText("Display");
    {
        ImGui::BeginDisabled(true);
        int dummy = 0;
        const char* interp_items[] = {"Off (experimental, disabled)"};
        ImGui::Combo("Frame interpolation", &dummy, interp_items, 1);
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Draw-level frame interpolation exists but is stubbed out: too janky in menus and gameplay. "
                              "The game stays at its native 60 fps.");
    }
    if (ImGui::Checkbox("Switch monitor to 60 Hz while playing", &v.display_60hz)) {
        changed = true;
        PostMessageW(g_hwnd, kMsgDisplay60, 0, v.display_60hz ? 1 : 0);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The game renders at 60 fps and its movies at 30 fps. On a 144 Hz panel each frame is shown for\n"
                          "2 or 3 (movies: 4 or 5) refreshes in turn, which looks like stutter. At 60 Hz every frame is\n"
                          "shown exactly once (movies: twice). Reverts when the game exits.");
    ImGui::TextWrapped("Game logic runs at a fixed 60 fps: the game advances one step per frame, so releasing frames faster "
                       "speeds the whole game up. Presentation pacing is locked to 60 Hz; true 120/144 fps needs a "
                       "game-side timestep change (on the roadmap).");
    if (ImGui::Checkbox("VSync (FIFO)", &v.vsync)) {
        changed = true;
        g_swapchain_dirty = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Off = mailbox presentation (lowest latency).\nOn = classic vsync; use it if you see tearing.");

    ImGui::SeparatorText("Window");
    if (ImGui::Checkbox("Borderless fullscreen (F11)", &v.borderless)) {
        changed = true;
        post_window_change();
    }
    {
        static const int sizes[][2] = {{1280, 720}, {1600, 900}, {1920, 1080}, {2560, 1440}, {3440, 1440}, {3840, 2160}};
        char cur[32];
        snprintf(cur, sizeof(cur), "%ux%u", extent.width, extent.height);
        ImGui::BeginDisabled(v.borderless);
        if (ImGui::BeginCombo("Window size", cur)) {
            for (auto& s : sizes) {
                char item[32];
                snprintf(item, sizeof(item), "%dx%d", s[0], s[1]);
                if (ImGui::Selectable(item, (int)extent.width == s[0] && (int)extent.height == s[1])) {
                    v.window_w = s[0];
                    v.window_h = s[1];
                    changed = true;
                    post_window_change();
                }
            }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
    }

    ImGui::SeparatorText("Rendering");
    {
        static const int levels[] = {0, 2, 4, 8, 16};
        static const char* names[] = {"Game default", "2x anisotropic", "4x anisotropic", "8x anisotropic", "16x anisotropic"};
        int cur = 0;
        for (int i = 0; i < 5; i++)
            if (levels[i] == v.aniso) cur = i;
        if (ImGui::BeginCombo("Texture filtering", names[cur])) {
            for (int i = 0; i < 5; i++)
                if (ImGui::Selectable(names[i], cur == i)) {
                    v.aniso = levels[i];
                    changed = true;
                }
            ImGui::EndCombo();
        }
    }
    {
        static const float steps[] = {0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f, 3.0f};
        int cur = 2;
        for (int i = 0; i < 7; i++)
            if (fabsf(steps[i] - v.res_scale) < 0.01f) cur = i;
        char label[64];
        snprintf(label, sizeof(label), "%.2fx  (%dx%d)", steps[cur], (int)(1920 * steps[cur]), (int)(1080 * steps[cur]));
        if (ImGui::BeginCombo("Internal resolution", label)) {
            for (int i = 0; i < 7; i++) {
                char item[64];
                snprintf(item, sizeof(item), "%.2fx  (%dx%d)", steps[i], (int)(1920 * steps[i]), (int)(1080 * steps[i]));
                if (ImGui::Selectable(item, cur == i)) {
                    v.res_scale = steps[i];
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::Checkbox("Unlocked resolution (render at the window's size)", &v.unlocked_res)) changed = true;
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Scene render targets are allocated at this scale (UI and small post-process targets stay native).\n"
                              "Unlocked = scale chosen from the window height at launch. Aspect ratio stays 16:9 for now.\n"
                              "Applies immediately.");
    }

    ImGui::SeparatorText("Cheats (save file)");
    {
        static std::string status;
        static std::vector<std::pair<size_t, size_t>> clusters;
        auto save_path = []() -> std::string {
            namespace fs = std::filesystem;
            std::error_code ec;
            for (auto& t : fs::directory_iterator("save", ec))
                for (auto& u : fs::directory_iterator(t.path(), ec)) {
                    fs::path p = u.path() / "zmha.bin";
                    if (fs::exists(p, ec)) return p.string();
                }
            return "";
        };
        auto read_all = [](const std::string& p) {
            std::vector<unsigned char> d;
            if (FILE* f = fopen(p.c_str(), "rb")) {
                fseek(f, 0, SEEK_END);
                long n = ftell(f);
                fseek(f, 0, SEEK_SET);
                d.resize(n > 0 ? n : 0);
                if (n > 0) fread(d.data(), 1, n, f);
                fclose(f);
            }
            return d;
        };
        std::string sp = save_path();
        ImGui::TextDisabled("%s", sp.empty() ? "no save found under ./save" : sp.c_str());
        if (ImGui::Button("Snapshot save")) {
            auto d = read_all(sp);
            if (FILE* f = fopen("save_snapshot.bin", "wb")) { fwrite(d.data(), 1, d.size(), f); fclose(f); }
            status = "snapshot taken (" + std::to_string(d.size()) + " bytes)";
        }
        ImGui::SameLine();
        if (ImGui::Button("Diff vs snapshot")) {
            auto a = read_all("save_snapshot.bin"), b = read_all(sp);
            clusters.clear();
            size_t n = std::min(a.size(), b.size()), start = SIZE_MAX, prev = 0, total = 0;
            for (size_t i = 0; i < n; i++) {
                if (a[i] == b[i]) continue;
                total++;
                if (start == SIZE_MAX) start = i;
                else if (i - prev > 64) { clusters.push_back({start, prev}); start = i; }
                prev = i;
            }
            if (start != SIZE_MAX) clusters.push_back({start, prev});
            status = std::to_string(total) + " bytes changed in " + std::to_string(clusters.size()) + " regions";
            for (auto& c : clusters) hw_log("cheats: save changed 0x%zx-0x%zx (%zu bytes)", c.first, c.second, c.second - c.first + 1);
        }
        ImGui::SameLine();
        if (ImGui::Button("Import unlocks from reference")) {
            // Reference = a 100% save at save_reference/zmha.bin. Every byte that is 0 here and 1 there
            // is set (unlock/clear flags); counters and other values are left alone. The game reads its
            // save at boot, so this takes effect on the next launch.
            auto ref = read_all("save_reference/zmha.bin"), cur = read_all(sp);
            size_t n = std::min(ref.size(), cur.size()), changed = 0;
            for (size_t i = 0x40; i < n; i++)
                if (cur[i] == 0 && ref[i] == 1) { cur[i] = 1; changed++; }
            if (!ref.empty() && !cur.empty()) {
                if (FILE* f = fopen((sp + ".bak").c_str(), "wb")) { auto o = read_all(sp); fwrite(o.data(), 1, o.size(), f); fclose(f); }
                if (FILE* f = fopen(sp.c_str(), "wb")) { fwrite(cur.data(), 1, cur.size(), f); fclose(f); }
                status = std::to_string(changed) + " flags set - restart the game (backup: zmha.bin.bak)";
            } else {
                status = ref.empty() ? "no reference save at save_reference/zmha.bin" : "could not read the save";
            }
        }
        if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());
        for (size_t i = 0; i < clusters.size() && i < 8; i++)
            ImGui::TextDisabled("  0x%zx - 0x%zx (%zu bytes)", clusters[i].first, clusters[i].second, clusters[i].second - clusters[i].first + 1);
    }

    ImGui::SeparatorText("Overlay");
    if (ImGui::Checkbox("Show frame time", &v.show_fps)) changed = true;
    ImGui::TextDisabled("Settings are saved to hwder_settings.ini when changed.");
    ImGui::End();
    if (changed) settings::save();
    (void)io;
}

void render(VkCommandBuffer cb, VkImage image, VkExtent2D extent) {
    if (!g_inited) return;
    std::lock_guard<std::recursive_mutex> l(g_m);
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    g_frame_ms[g_frame_i++ % 120] = (t.QuadPart - g_last_t.QuadPart) * 1000.0 / f.QuadPart;
    g_last_t = t;
    settings::Video& v = settings::video();
    if (!g_visible && !v.show_fps) return;
    auto vit = g_views.find(image);
    if (vit == g_views.end()) return;
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    build_ui(extent);
    ImGui::Render();
    ImDrawData* dd = ImGui::GetDrawData();
    if (!dd || dd->TotalVtxCount == 0) return;
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = vit->second;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    vkCmdBeginRendering(cb, &ri);
    ImGui_ImplVulkan_RenderDrawData(dd, cb);
    vkCmdEndRendering(cb);
}

}  // namespace overlay
