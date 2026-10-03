// Host input sources (XInput gamepads + keyboard/mouse on the HWDER window), already translated to
// Switch semantics. Polled by the HID update thread.
#pragma once
#include "hwder/runtime.h"

namespace hid {

// nn::hid::NpadButton bits
enum : u64 {
    kBtnA = 1ull << 0, kBtnB = 1ull << 1, kBtnX = 1ull << 2, kBtnY = 1ull << 3,
    kBtnStickL = 1ull << 4, kBtnStickR = 1ull << 5, kBtnL = 1ull << 6, kBtnR = 1ull << 7,
    kBtnZL = 1ull << 8, kBtnZR = 1ull << 9, kBtnPlus = 1ull << 10, kBtnMinus = 1ull << 11,
    kBtnLeft = 1ull << 12, kBtnUp = 1ull << 13, kBtnRight = 1ull << 14, kBtnDown = 1ull << 15,
    kBtnStickLLeft = 1ull << 16, kBtnStickLUp = 1ull << 17, kBtnStickLRight = 1ull << 18, kBtnStickLDown = 1ull << 19,
    kBtnStickRLeft = 1ull << 20, kBtnStickRUp = 1ull << 21, kBtnStickRRight = 1ull << 22, kBtnStickRDown = 1ull << 23,
};

constexpr s32 kStickMax = 0x7FFF;

struct PadInput {
    u64 buttons = 0;
    s32 lx = 0, ly = 0, rx = 0, ry = 0;  // -32767..32767, +y = up
};

struct TouchInput {
    bool down = false;
    u32 x = 0, y = 0;  // 1280x720 touch panel space
};

struct HostInput {
    PadInput pad;
    TouchInput touch;
};

// Poll every host source (cheap; safe to call at 1 kHz). Not thread safe: call from one thread.
void poll_host_input(HostInput& out);

// Rumble on the active XInput pad. Amplitudes 0..1 (low band -> big motor, high band -> small motor).
void set_rumble(float low, float high);

}  // namespace hid
