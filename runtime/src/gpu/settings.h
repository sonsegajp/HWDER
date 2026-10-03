// User-facing video settings (edited in the F1 overlay, persisted in hwder_settings.ini next to the exe).
#pragma once

namespace settings {

struct Video {
    int refresh = 60;           // presenter rate in Hz: 60, 120, 144; 0 = match the display
    bool vsync = false;         // FIFO present mode when true, MAILBOX otherwise
    int aniso = 0;              // anisotropic filtering override: 0 = game default, 2/4/8/16
    int window_w = 0;           // requested client size, 0 = leave the window alone
    int window_h = 0;
    bool borderless = false;    // borderless fullscreen on the current monitor
    float res_scale = 1.0f;     // internal resolution scale (renderer support pending)
    bool unlocked_res = false;  // render at the window's native size / aspect (pending)
    bool show_fps = false;      // small frame-time readout in the corner
    int interp_hz = 0;          // frame interpolation output rate: 0 off, -1 match display, else Hz
    bool display_60hz = false;  // switch the monitor to 60 Hz while the game runs (even cadence for 60/30 fps content)
};

Video& video();
void load();
void save();
double refresh_hz();  // `refresh` resolved against the display when it is 0

}  // namespace settings
