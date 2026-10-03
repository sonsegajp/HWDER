#include "gpu/settings.h"

#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace settings {

static Video g_video;
static const char* kFile = "hwder_settings.ini";

Video& video() { return g_video; }

void load() {
    // Default for the 60 Hz monitor switch: on when the desktop refresh is not a multiple of 60. The game
    // runs at 60 fps and its movies at 30; on a 144 Hz panel FIFO presentation shows frames for 2/3 or
    // 5/5/5/4 refreshes, which reads as stutter. An explicit value in the ini always wins.
    {
        DEVMODEW dm{};
        dm.dmSize = sizeof(dm);
        if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &dm) && dm.dmDisplayFrequency > 1)
            g_video.display_60hz = (dm.dmDisplayFrequency % 60) != 0;
    }
    FILE* f = fopen(kFile, "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char* eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        std::string k = line;
        const char* v = eq + 1;
        if (k == "refresh") g_video.refresh = atoi(v);
        else if (k == "vsync") g_video.vsync = atoi(v) != 0;
        else if (k == "aniso") g_video.aniso = atoi(v);
        else if (k == "window_w") g_video.window_w = atoi(v);
        else if (k == "window_h") g_video.window_h = atoi(v);
        else if (k == "borderless") g_video.borderless = atoi(v) != 0;
        else if (k == "res_scale") g_video.res_scale = (float)atof(v);
        else if (k == "unlocked_res") g_video.unlocked_res = atoi(v) != 0;
        else if (k == "show_fps") g_video.show_fps = atoi(v) != 0;
        else if (k == "display_60hz") g_video.display_60hz = atoi(v) != 0;
        else if (k == "interp_hz") g_video.interp_hz = 0;  // frame interpolation is stubbed out (too janky); HWDER_INTERP=<hz> re-enables it for experiments
    }
    fclose(f);
}

void save() {
    FILE* f = fopen(kFile, "w");
    if (!f) return;
    fprintf(f, "refresh=%d\nvsync=%d\naniso=%d\nwindow_w=%d\nwindow_h=%d\nborderless=%d\nres_scale=%.3f\nunlocked_res=%d\nshow_fps=%d\ndisplay_60hz=%d\ninterp_hz=%d\n",
            g_video.refresh, (int)g_video.vsync, g_video.aniso, g_video.window_w, g_video.window_h, (int)g_video.borderless,
            g_video.res_scale, (int)g_video.unlocked_res, (int)g_video.show_fps, (int)g_video.display_60hz, g_video.interp_hz);
    fclose(f);
}

double refresh_hz() {
    if (g_video.refresh > 0) return (double)g_video.refresh;
    DEVMODEW dm{};
    dm.dmSize = sizeof(dm);
    if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &dm) && dm.dmDisplayFrequency > 1) return (double)dm.dmDisplayFrequency;
    return 60.0;
}

}  // namespace settings
