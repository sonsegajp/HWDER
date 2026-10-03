// hid services: IHidServer / IAppletResource with the HID shared memory (0x40000) kept up to date by a
// host thread. Layout: switchbrew "HID_Shared_Memory".
#include <windows.h>
#include <timeapi.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>

#include "host_input.h"
#include "service/common.h"

extern "C" u64 hw_cntvct();

namespace ipc {

namespace {

// ------------------------------------------------------------------ shared memory layout
constexpr u64 kShmemSize = 0x40000;
constexpr u32 kLifoEntries = 17;
constexpr u64 kOffDebugPad = 0x0, kOffTouch = 0x400, kOffMouse = 0x3400, kOffKeyboard = 0x3800;
constexpr u64 kOffHome = 0x4C00, kOffSleep = 0x4E00, kOffCapture = 0x5000;
constexpr u64 kOffNpad = 0x9A00, kNpadEntrySize = 0x5000;
constexpr u64 kOffGesture = 0x3BA00;
constexpr u64 kOffConsoleSixAxis = 0x3C200;

// Lifo header: timestamp, total entry count (17), index of latest entry, valid entry count (<=16).
struct LifoHeader {
    s64 timestamp, total, tail, count;
};

// Each entry is {u64 sampling_number<<1, State} where State starts with its own u64 sampling number.
// `fill` writes the state body after the state's sampling number.
template <typename F>
void lifo_push(u8* base, u64 storage_size, u64 sampling, u64 ticks, F fill) {
    auto* h = (LifoHeader*)base;
    s64 next = (h->tail + 1) % kLifoEntries;
    u8* e = base + sizeof(LifoHeader) + next * storage_size;
    memset(e + 8, 0, storage_size - 8);
    *(u64*)(e + 8) = sampling;
    fill(e + 16);
    std::atomic_thread_fence(std::memory_order_release);
    *(volatile u64*)e = sampling << 1;
    std::atomic_thread_fence(std::memory_order_release);
    h->total = kLifoEntries;
    h->tail = next;
    if (h->count < (s64)kLifoEntries - 1) h->count++;
    h->timestamp = (s64)ticks;
}
void lifo_push_empty(u8* base, u64 storage_size, u64 sampling, u64 ticks) {
    lifo_push(base, storage_size, sampling, ticks, [](u8*) {});
}

// NpadStyleSet bits
enum : u32 {
    kStyleFullKey = 1 << 0, kStyleHandheld = 1 << 1, kStyleJoyDual = 1 << 2, kStyleJoyLeft = 1 << 3,
    kStyleJoyRight = 1 << 4,
};
// NpadAttribute bits
enum : u32 {
    kAttrConnected = 1 << 0, kAttrWired = 1 << 1, kAttrLeftConnected = 1 << 2, kAttrLeftWired = 1 << 3,
    kAttrRightConnected = 1 << 4, kAttrRightWired = 1 << 5,
};
constexpr u32 kNpadHandheld = 0x20, kNpadOther = 0x10;

int npad_index(u32 id) {
    if (id < 8) return (int)id;
    if (id == kNpadHandheld) return 8;
    if (id == kNpadOther) return 9;
    return -1;
}

struct NpadGenericState {  // nn::hid::NpadFullKeyState etc. minus the sampling number
    u64 buttons;
    s32 lx, ly, rx, ry;
    u32 attributes;
    u32 reserved;
};

// ------------------------------------------------------------------ global hid state
struct HidState {
    std::once_flag once;
    std::shared_ptr<SharedMemory> shmem;
    std::mutex lock;
    u32 supported_styles = kStyleFullKey | kStyleHandheld | kStyleJoyDual | kStyleJoyLeft | kStyleJoyRight;
    bool id_supported[10] = {true, true, true, true, true, true, true, true, true, true};
    u32 cur_style[10] = {};  // style currently reported per npad entry
    std::shared_ptr<ReadableEvent> style_event[10];
    s64 hold_type = 0;       // 0 vertical, 1 horizontal
    s64 handheld_mode = 0;
    std::atomic<bool> npad_active{false};
    u64 sampling = 0;
};
HidState g_hid;

u8* shm() { return g_hid.shmem->ptr(); }

// Which style the given npad entry exposes now (0 = disconnected).
u32 desired_style(int idx) {
    u32 sup = g_hid.supported_styles;
    if (!g_hid.id_supported[idx]) return 0;
    if (idx == 0) {
        if (sup & kStyleFullKey) return kStyleFullKey;
        if (sup & kStyleJoyDual) return kStyleJoyDual;
        if (sup & kStyleHandheld) return 0;  // handheld-only game: entry 8 carries the input
        return 0;
    }
    if (idx == 8) return (sup & kStyleHandheld) ? kStyleHandheld : 0;
    return 0;
}

void write_npad_header(int idx, u32 style) {
    u8* e = shm() + kOffNpad + (u64)idx * kNpadEntrySize;
    *(u32*)(e + 0x0) = style;
    *(u32*)(e + 0x4) = 0;  // joy assignment mode: dual
    // Full-key colour (attribute, main, sub) at 0x8, joy colour (attribute, left main/sub, right main/sub) at 0x14.
    u32* fc = (u32*)(e + 0x8);
    u32* jc = (u32*)(e + 0x14);
    if (style == kStyleFullKey) {
        fc[0] = 0; fc[1] = 0xFF2D2D2D; fc[2] = 0xFFE6E6E6;
        jc[0] = 2; jc[1] = jc[2] = jc[3] = jc[4] = 0;
    } else if (style) {
        fc[0] = 2; fc[1] = fc[2] = 0;
        jc[0] = 0; jc[1] = 0xFFFF3C28; jc[2] = 0xFF1E0A0A; jc[3] = 0xFF0AB9E6; jc[4] = 0xFF001E1E;
    } else {
        fc[0] = 2; fc[1] = fc[2] = 0;
        jc[0] = 2; jc[1] = jc[2] = jc[3] = jc[4] = 0;
    }
    // After the 7 npad LIFOs (0x28 + 7*0x350) and 6 six-axis LIFOs (6*0x708): device type, system props.
    u8* tail = e + 0x28 + 7 * 0x350 + 6 * 0x708;
    u32 dev = 0;
    if (style == kStyleFullKey) dev = 1 << 0;
    else if (style == kStyleHandheld) dev = (1 << 2) | (1 << 3);
    else if (style == kStyleJoyDual) dev = (1 << 4) | (1 << 5);
    *(u32*)(tail + 0x0) = dev;
    *(u64*)(tail + 0x8) = style ? ((1ull << 3) | (1ull << 11) | (1ull << 12) | (1ull << 13) | (1ull << 14) | (1ull << 15)) : 0;
    *(u32*)(tail + 0x10) = 0;  // system button properties
    *(u32*)(tail + 0x14) = style ? 4 : 0;
    *(u32*)(tail + 0x18) = style ? 4 : 0;
    *(u32*)(tail + 0x1C) = style ? 4 : 0;
}

void update_npad_styles() {
    for (int i = 0; i < 10; i++) {
        u32 want = desired_style(i);
        if (want != g_hid.cur_style[i]) {
            g_hid.cur_style[i] = want;
            write_npad_header(i, want);
            if (g_hid.style_event[i]) g_hid.style_event[i]->signal();
            hw_log("hid: npad %d style 0x%x", i, want);
        }
    }
}

void update_once(const hid::HostInput& in) {
    std::lock_guard<std::mutex> lk(g_hid.lock);
    u64 ticks = hw_cntvct();
    u64 s = ++g_hid.sampling;
    u8* m = shm();
    update_npad_styles();

    lifo_push_empty(m + kOffDebugPad, 0x28, s, ticks);
    lifo_push(m + kOffTouch, 0x298, s, ticks, [&](u8* p) {
        if (!in.touch.down) return;
        *(s32*)(p + 0) = 1;  // entry count
        u8* t = p + 8;       // TouchState: delta time, attributes, finger id, x, y, diameter x/y, angle
        *(u64*)(t + 0x0) = 0;
        *(u32*)(t + 0x8) = 0;
        *(u32*)(t + 0xC) = 0;
        *(u32*)(t + 0x10) = in.touch.x;
        *(u32*)(t + 0x14) = in.touch.y;
        *(u32*)(t + 0x18) = 15;
        *(u32*)(t + 0x1C) = 15;
        *(s32*)(t + 0x20) = 0;
    });
    lifo_push_empty(m + kOffMouse, 0x30, s, ticks);
    lifo_push_empty(m + kOffKeyboard, 0x38, s, ticks);
    lifo_push_empty(m + kOffHome, 0x18, s, ticks);
    lifo_push_empty(m + kOffSleep, 0x18, s, ticks);
    lifo_push_empty(m + kOffCapture, 0x18, s, ticks);
    lifo_push_empty(m + kOffGesture, 0x68, s, ticks);
    *(u64*)(m + kOffConsoleSixAxis) = s;

    for (int i = 0; i < 10; i++) {
        u8* e = m + kOffNpad + (u64)i * kNpadEntrySize;
        u32 style = g_hid.cur_style[i];
        for (int l = 0; l < 7; l++) {
            u32 lstyle = 1u << l;  // lifo order matches style bits: fullkey, handheld, dual, left, right, palma, sysext
            NpadGenericState st{};
            if (style && style == lstyle) {
                st.buttons = in.pad.buttons;
                st.lx = in.pad.lx; st.ly = in.pad.ly;
                st.rx = in.pad.rx; st.ry = in.pad.ry;
                if (style == kStyleFullKey) st.attributes = kAttrConnected;
                else if (style == kStyleHandheld)
                    st.attributes = kAttrConnected | kAttrWired | kAttrLeftConnected | kAttrLeftWired |
                                    kAttrRightConnected | kAttrRightWired;
                else st.attributes = kAttrConnected | kAttrLeftConnected | kAttrRightConnected;
            }
            lifo_push(e + 0x28 + l * 0x350, 0x30, s, ticks, [&](u8* p) { memcpy(p, &st, sizeof(st)); });
        }
        // Six-axis LIFOs: at rest, gravity on -Z, attributes IsConnected when the pad is.
        for (int l = 0; l < 6; l++) {
            lifo_push(e + 0x28 + 7 * 0x350 + l * 0x708, 0x68, s, ticks, [&](u8* p) {
                // SixAxisSensorState: delta_time, sampling, acc, gyro, rotation, direction[3], attr.
                // Its first field is delta_time, so swap the two words lifo_push wrote.
                *(u64*)(p - 8) = 5000000;
                *(u64*)(p + 0) = s;
                float* f = (float*)(p + 8);
                f[2] = -1.0f;  // accel z
                float* dir = (float*)(p + 0x2C);
                dir[0] = 1.0f; dir[4] = 1.0f; dir[8] = 1.0f;
                *(u32*)(p + 0x50) = style ? 1 : 0;
            });
        }
    }
}

void update_thread() {
    timeBeginPeriod(1);
    hid::HostInput in;
    // HWDER_AUTOPLAY=1: tap A every 1.5 s (after 5 s) so unattended runs advance through the menus.
    static const bool autoplay = getenv("HWDER_AUTOPLAY") != nullptr;
    const u64 t0 = GetTickCount64();
    for (;;) {
        hid::poll_host_input(in);
        if (autoplay) {
            u64 t = GetTickCount64() - t0;
            u64 c = t % 4000;
            // Generic advance: A every 1.5 s. Skip cutscenes: + opens "Skip cutscene?", Left selects Yes, A confirms.
            if (t > 5000 && (t % 1500) < 120) in.pad.buttons |= hid::kBtnA;
            // The skip sequence only during the window where the opening cutscenes play (otherwise +
            // interferes with menus such as the mission list).
            bool skip_window = t > 18000 && t < 34000;
            if (skip_window && c >= 700 && c < 900) in.pad.buttons |= hid::kBtnPlus;
            if (skip_window && c >= 1600 && c < 1720) in.pad.buttons |= hid::kBtnLeft;
            if (skip_window && c >= 2100 && c < 2220) in.pad.buttons |= hid::kBtnA;
            if (skip_window && c >= 2600 && c < 2720) in.pad.buttons |= hid::kBtnA;
            // In battle (after ~50 s): close any menu with B now and then, run forward and attack.
            if (t > 50000) {
                // Loading-screen tips wait for Start; a B 1.5 s later closes the pause menu if that is what + opened.
                if ((t % 12000) < 150) in.pad.buttons |= hid::kBtnPlus;
                if ((t % 12000) >= 1500 && (t % 12000) < 1620) in.pad.buttons |= hid::kBtnB;
                if ((t % 800) < 100) in.pad.buttons |= hid::kBtnY;
                in.pad.ly = 20000;
                in.pad.lx = (s32)(((t / 3000) % 2) ? 12000 : -12000);
            }
        }
        update_once(in);
        Sleep(4);
    }
}

void hid_init() {
    std::call_once(g_hid.once, [] {
        g_hid.shmem = kern::create_shared_memory(kShmemSize);
        for (int i = 0; i < 10; i++) g_hid.style_event[i] = make_event(true);
        // Prime every LIFO so the first reads see valid entries.
        hid::HostInput in;
        update_once(in);
        std::thread(update_thread).detach();
    });
}

// ------------------------------------------------------------------ interfaces
class IAppletResource : public SimpleService {
public:
    IAppletResource() : SimpleService("hid:IAppletResource") {
        reg(0, [](Request&, Response& rs) { push_handle(rs, g_hid.shmem); });  // GetSharedMemoryHandle
    }
};

class IActiveVibrationDeviceList : public SimpleService {
public:
    IActiveVibrationDeviceList() : SimpleService("hid:IActiveVibrationDeviceList") { nop({0}); }
};

// Vibration handle: u8 type, u8 npad id, u8 device index (0 left, 1 right), u8 pad.
// Left device drives the low-frequency (big) motor, right device the high-frequency one.
void send_vibration(u32 handle, const float v[4]) {
    u8 npad = (u8)(handle >> 8), dev = (u8)(handle >> 16);
    if (npad != 0 && npad != kNpadHandheld) return;
    static std::mutex m;
    static float amp[2] = {0, 0}, sent[2] = {-1, -1};
    std::lock_guard<std::mutex> l(m);
    amp[dev & 1] = std::max(v[0], v[2]);
    if (amp[0] == sent[0] && amp[1] == sent[1]) return;
    sent[0] = amp[0];
    sent[1] = amp[1];
    hid::set_rumble(amp[0], amp[1]);
}

class IHidServer : public SimpleService {
public:
    IHidServer() : SimpleService("hid") {
        hid_init();
        obj(0, [] { return std::make_shared<IAppletResource>(); });  // CreateAppletResource
        // Activate*/Deactivate*: the shared memory is always live.
        nop({1, 11, 21, 31, 41, 51, 56, 57, 60, 61, 66, 67, 69, 70, 72, 73, 75, 76, 78, 79, 81, 91, 104, 107,
             122, 123, 124, 125, 126, 127, 128, 130, 132, 204, 300, 301, 302, 401, 1000});
        reg(40, [](Request&, Response& rs) { push_handle(rs, make_event()); });  // AcquireXpadIdEventHandle
        ret<s64>(55, 0);                                                          // GetXpadIds
        ret<s64>(59, 0);                                                          // GetJoyXpadIds
        ret<u8>(68, 1);                                                           // IsSixAxisSensorFusionEnabled
        reg(71, [](Request&, Response& rs) { rs.push<float>(0.03f); rs.push<float>(0.4f); });
        reg(74, [](Request&, Response& rs) { rs.push<float>(-0.4f); rs.push<float>(0.4f); });
        ret<u32>(77, 0);
        ret<u32>(80, 1);
        ret<u8>(82, 1);  // IsSixAxisSensorAtRest
        reg(100, [](Request& rq, Response&) {  // SetSupportedNpadStyleSet(u32 styles, aruid)
            u32 styles = rq.pop<u32>();
            std::lock_guard<std::mutex> lk(g_hid.lock);
            g_hid.supported_styles = styles;
            hw_log("hid: SetSupportedNpadStyleSet 0x%x", styles);
            update_npad_styles();
        });
        reg(101, [](Request&, Response& rs) { rs.push<u32>(g_hid.supported_styles); });
        reg(102, [](Request& rq, Response&) {  // SetSupportedNpadIdType(aruid, X buffer u32[])
            Buffer b = rq.in_buffer();
            std::lock_guard<std::mutex> lk(g_hid.lock);
            if (!b.addr) return;
            for (bool& x : g_hid.id_supported) x = false;
            std::string ids;
            for (u64 i = 0; i < b.size / 4; i++) {
                u32 id = ((u32*)b.addr)[i];
                int idx = npad_index(id);
                if (idx >= 0) g_hid.id_supported[idx] = true;
                ids += " " + std::to_string(id);
            }
            hw_log("hid: SetSupportedNpadIdType%s", ids.c_str());
            update_npad_styles();
        });
        reg(103, [](Request&, Response&) {  // ActivateNpad
            g_hid.npad_active = true;
        });
        nop({109});  // ActivateNpadWithRevision
        reg(106, [](Request& rq, Response& rs) {  // AcquireNpadStyleSetUpdateEventHandle(u32 id, aruid, u64)
            u32 id = rq.pop<u32>();
            int idx = npad_index(id);
            push_handle(rs, idx >= 0 ? g_hid.style_event[idx] : make_event(true));
        });
        reg(108, [](Request& rq, Response& rs) {  // GetPlayerLedPattern
            u32 id = rq.pop<u32>();
            static const u64 pat[8] = {1, 3, 7, 0xF, 9, 5, 0xD, 6};
            rs.push<u64>(id < 8 ? pat[id] : 0);
        });
        reg(120, [](Request& rq, Response&) {  // SetNpadJoyHoldType(aruid, s64)
            rq.pop<u64>();
            g_hid.hold_type = rq.pop<s64>();
        });
        reg(121, [](Request&, Response& rs) { rs.push<s64>(g_hid.hold_type); });
        reg(129, [](Request&, Response& rs) { rs.push<s64>(g_hid.handheld_mode); });
        ret<u8>(131, 0);  // IsUnintendedHomeButtonInputProtectionEnabled
        reg(200, [](Request& rq, Response& rs) {  // GetVibrationDeviceInfo(handle)
            u32 h = rq.pop<u32>();
            u8 dev_idx = (u8)(h >> 16);
            rs.push<u32>(1);                    // LinearResonantActuator
            rs.push<u32>(dev_idx == 0 ? 1 : 2);  // left / right
        });
        reg(201, [](Request& rq, Response&) {  // SendVibrationValue(handle, VibrationValue, aruid)
            u32 h = rq.pop<u32>();
            float v[4];
            for (float& f : v) f = rq.pop<float>();
            send_vibration(h, v);
        });
        reg(202, [](Request&, Response& rs) {  // GetActualVibrationValue
            rs.push<float>(0.0f); rs.push<float>(160.0f); rs.push<float>(0.0f); rs.push<float>(320.0f);
        });
        obj(203, [] { return std::make_shared<IActiveVibrationDeviceList>(); });
        ret<u8>(205, 1);  // IsVibrationPermitted
        reg(206, [](Request& rq, Response&) {  // SendVibrationValues(aruid, X handles, X values)
            Buffer hb = rq.in_buffer(0), vb = rq.in_buffer(1);
            if (!hb.addr || !vb.addr) return;
            u64 n = std::min(hb.size / 4, vb.size / 16);
            for (u64 i = 0; i < n; i++) send_vibration(((u32*)hb.addr)[i], (const float*)(vb.addr + i * 16));
        });
        ret<u8>(400, 0);  // IsUsbFullKeyControllerEnabled
        ret<u8>(402, 0);
        ret<u64>(1001, 0);
    }
};

class IHidSystemServer : public SimpleService {
public:
    IHidSystemServer() : SimpleService("hid:sys") { hid_init(); }
};

// irs (IR camera): only the shared memory handle has to be real.
class IIrSensorServer : public SimpleService {
public:
    IIrSensorServer() : SimpleService("irs") {
        nop({302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314});
        reg(304, [](Request&, Response& rs) {  // GetIrsensorSharedMemoryHandle
            static auto mem = kern::create_shared_memory(0x8000);
            push_handle(rs, mem);
        });
        reg(311, [](Request&, Response& rs) { rs.push<u32>(0x000F0000); });  // GetNpadIrCameraHandle
    }
};

// nfp:user / nfc:user (amiibo / NFC): one reader on player 1 that never finds a tag.
class INfcUser : public SimpleService {
public:
    explicit INfcUser(const char* name) : SimpleService(name) {
        reg(0, [this](Request&, Response&) { state_ = 1; });   // Initialize
        reg(1, [this](Request&, Response&) { state_ = 0; });   // Finalize
        reg(2, [](Request& rq, Response& rs) {                  // ListDevices -> count, B buffer of u64 handles
            u64 dev = 0;  // device handle = npad id 0
            write_buffer(rq.out_buffer(), &dev, sizeof(dev));
            rs.push<s32>(1);
        });
        reg(3, [this](Request&, Response&) { dev_state_ = 1; });  // StartDetection: searching
        reg(4, [this](Request&, Response&) { dev_state_ = 0; });  // StopDetection
        auto no_tag = [](Request&, Response& rs) { rs.result = ModuleResult(115, 64); };  // TagNotFound
        for (u32 id : {5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u, 22u, 24u}) reg(id, no_tag);
        reg(17, [this](Request&, Response& rs) { push_handle(rs, activate_); });    // AttachActivateEvent
        reg(18, [this](Request&, Response& rs) { push_handle(rs, deactivate_); });  // AttachDeactivateEvent
        reg(19, [this](Request&, Response& rs) { rs.push<u32>(state_); });          // GetState
        reg(20, [this](Request&, Response& rs) { rs.push<u32>(dev_state_); });      // GetDeviceState
        reg(21, [](Request&, Response& rs) { rs.push<u32>(0); });                   // GetNpadId
        reg(23, [this](Request&, Response& rs) { push_handle(rs, avail_); });       // AttachAvailabilityChangeEvent
    }

private:
    u32 state_ = 0, dev_state_ = 0;
    std::shared_ptr<ReadableEvent> activate_ = make_event(), deactivate_ = make_event(), avail_ = make_event();
};
class NfpUserManager : public SimpleService {
public:
    NfpUserManager() : SimpleService("nfp:user") {
        obj(0, [] { return std::make_shared<INfcUser>("nfp:IUser"); });
    }
};
class NfcUserManager : public SimpleService {
public:
    NfcUserManager() : SimpleService("nfc:user") {
        obj(0, [] { return std::make_shared<INfcUser>("nfc:IUser"); });
    }
};

}  // namespace

void register_hid_services() {
    register_as<NfpUserManager>({"nfp:user", "nfp:sys", "nfp:dbg"});
    register_as<NfcUserManager>({"nfc:user", "nfc:sys"});
    register_as<IHidServer>({"hid"});
    register_as<IHidSystemServer>({"hid:sys", "hid:dbg", "hid:tmp"});
    register_as<IIrSensorServer>({"irs", "irs:sys"});
}

}  // namespace ipc
