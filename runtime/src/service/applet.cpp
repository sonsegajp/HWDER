// am: applet manager (appletOE: the application's view of the system: focus, messages, storage).
#include <mutex>

#include "common.h"

namespace ipc {

namespace {

constexpr Result ResultNoMessage = ModuleResult(128, 3);
constexpr Result ResultNoData = ModuleResult(128, 2);

// System -> application message queue (ICommonStateGetter::ReceiveMessage).
struct MessageQueue {
    std::mutex m;
    std::deque<u32> q;
    std::shared_ptr<ReadableEvent> event = make_event();
    void push(u32 msg) {
        kern::KGuard g(kern::g_kernel_lock);
        std::lock_guard<std::mutex> l(m);
        q.push_back(msg);
        event->signal();
    }
    bool pop(u32* out) {
        kern::KGuard g(kern::g_kernel_lock);
        std::lock_guard<std::mutex> l(m);
        if (q.empty()) {
            event->clear();
            return false;
        }
        *out = q.front();
        q.pop_front();
        if (q.empty()) event->clear();
        return true;
    }
};
MessageQueue& messages() {
    static MessageQueue* mq = [] {
        auto* m = new MessageQueue;
        m->q.push_back(15);  // FocusStateChanged: we start in focus
        m->event->signaled = true;
        return m;
    }();
    return *mq;
}

// IStorage / IStorageAccessor: a byte blob passed between applets.
struct Storage {
    std::vector<u8> data;
};

class IStorageAccessor : public SimpleService {
public:
    explicit IStorageAccessor(std::shared_ptr<Storage> s) : SimpleService("IStorageAccessor"), s_(std::move(s)) {
        reg(0, [this](Request&, Response& rs) { rs.push<u64>(s_->data.size()); });
        reg(10, [this](Request& rq, Response& rs) {  // Write(offset, A buffer)
            u64 off = rq.pop<u64>();
            Buffer b = rq.in_buffer();
            if (off + b.size > s_->data.size()) return void(rs.result = ModuleResult(128, 503));
            memcpy(s_->data.data() + off, (void*)b.addr, b.size);
        });
        reg(11, [this](Request& rq, Response& rs) {  // Read(offset, B buffer)
            u64 off = rq.pop<u64>();
            Buffer b = rq.out_buffer();
            if (off > s_->data.size()) return void(rs.result = ModuleResult(128, 503));
            u64 n = std::min<u64>(b.size, s_->data.size() - off);
            memcpy((void*)b.addr, s_->data.data() + off, n);
        });
    }
    std::shared_ptr<Storage> s_;
};

class IStorage : public SimpleService {
public:
    explicit IStorage(std::shared_ptr<Storage> s) : SimpleService("IStorage"), s_(std::move(s)) {
        reg(0, [this](Request&, Response& rs) { rs.push_object(std::make_shared<IStorageAccessor>(s_)); });
    }
    std::shared_ptr<Storage> s_;
};

class ICommonStateGetter : public SimpleService {
public:
    ICommonStateGetter() : SimpleService("ICommonStateGetter") {
        reg(0, [](Request&, Response& rs) { push_handle(rs, messages().event); });  // GetEventHandle
        reg(1, [](Request&, Response& rs) {                                          // ReceiveMessage
            u32 msg;
            if (!messages().pop(&msg)) return void(rs.result = ResultNoMessage);
            hw_log("am: message %u", msg);
            rs.push<u32>(msg);
        });
        ret<u8>(5, 1);          // GetOperationMode: console (docked)
        ret<u32>(6, 1);         // GetPerformanceMode: boost (docked)
        ret<u8>(8, 0);          // GetBootMode
        ret<u8>(9, 1);          // GetCurrentFocusState: InFocus
        nop({10, 11, 12, 13, 50, 51, 52, 53, 54, 55, 59, 66, 67, 68, 80, 900});
        reg(13, [](Request&, Response& rs) { push_handle(rs, make_event()); });  // GetAcquiredSleepLockEvent
        reg(60, [](Request&, Response& rs) {  // GetDefaultDisplayResolution
            rs.push<s32>(1920);
            rs.push<s32>(1080);
        });
        reg(61, [](Request&, Response& rs) { push_handle(rs, make_event()); });
        ret<u8>(50, 0);  // IsVrModeEnabled
        reg(300, [](Request&, Response& rs) { rs.push<s32>(1); });  // GetSettingsPlatformRegion
    }
};

class ISelfController : public SimpleService {
public:
    ISelfController() : SimpleService("ISelfController") {
        reg(0, [](Request&, Response&) {
            hw_log("am: application requested exit");
            ExitProcess(0);
        });
        nop({1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 41, 42, 44, 45, 46, 50, 51, 61, 62, 64, 65, 68, 72,
             100, 110, 120});
        reg(9, [](Request&, Response& rs) { push_handle(rs, make_event()); });   // GetLibraryAppletLaunchableEvent
        reg(40, [](Request&, Response& rs) { rs.push<u64>(1); });                 // CreateManagedDisplayLayer
        reg(43, [](Request&, Response& rs) { rs.push<u64>(1); rs.push<u64>(2); });  // CreateManagedDisplaySeparableLayer
        ret<u32>(63, 0);   // GetIdleTimeDetectionExtension
        ret<u8>(69, 0);    // IsAutoSleepDisabled
        ret<u64>(90, 0);   // GetAccumulatedSuspendedTickValue
        reg(91, [](Request&, Response& rs) { push_handle(rs, make_event()); });
    }
};

class IWindowController : public SimpleService {
public:
    IWindowController() : SimpleService("IWindowController") {
        ret<u64>(1, 0x7A0);  // GetAppletResourceUserId
        nop({10, 12});
    }
};

class IAudioController : public SimpleService {
public:
    IAudioController() : SimpleService("IAudioController") {
        nop({0, 3, 4});
        ret<float>(1, 1.0f);
        ret<float>(2, 1.0f);
    }
};

class ILibraryAppletAccessor : public SimpleService {
public:
    explicit ILibraryAppletAccessor(u32 id) : SimpleService("ILibraryAppletAccessor"), id_(id) {
        reg(0, [](Request&, Response& rs) { push_handle(rs, make_event(true)); });  // GetAppletStateChangedEvent
        ret<u8>(1, 1);                                                               // IsCompleted
        nop({10, 20, 25, 100, 103, 104, 105});                                       // Start, RequestExit, Push*
        reg(10, [this](Request&, Response&) { hw_log("am: library applet 0x%x started (auto-completes)", id_); });
        reg(101, [](Request&, Response& rs) { rs.result = ResultNoData; });  // PopOutData
        reg(106, [](Request&, Response& rs) { push_handle(rs, make_event()); });
        reg(105, [](Request&, Response& rs) { push_handle(rs, make_event()); });
    }
    u32 id_;
};

class ILibraryAppletCreator : public SimpleService {
public:
    ILibraryAppletCreator() : SimpleService("ILibraryAppletCreator") {
        reg(0, [](Request& rq, Response& rs) {  // CreateLibraryApplet(id, mode)
            u32 id = rq.pop<u32>();
            hw_log("am: CreateLibraryApplet(0x%x)", id);
            rs.push_object(std::make_shared<ILibraryAppletAccessor>(id));
        });
        reg(10, [](Request& rq, Response& rs) {  // CreateStorage(size)
            auto s = std::make_shared<Storage>();
            s->data.resize(rq.pop<u64>());
            rs.push_object(std::make_shared<IStorage>(s));
        });
        reg(11, [](Request& rq, Response& rs) {  // CreateTransferMemoryStorage(writable, size, handle)
            rq.pop<u8>();
            u64 size = rq.pop<u64>();
            auto s = std::make_shared<Storage>();
            s->data.resize(size);
            if (!rq.copy_handles.empty())
                if (auto tm = g_handles.get_as<kern::TransferMemory>(rq.copy_handles[0]))
                    memcpy(s->data.data(), (void*)tm->addr, std::min(size, tm->size));
            rs.push_object(std::make_shared<IStorage>(s));
        });
    }
};

class IApplicationFunctions : public SimpleService {
public:
    IApplicationFunctions() : SimpleService("IApplicationFunctions") {
        reg(1, [](Request& rq, Response& rs) {  // PopLaunchParameter(kind)
            u32 kind = rq.pop<u32>();
            static bool popped_user = false;
            if (kind == 2 && !popped_user) {  // PreselectedUser
                popped_user = true;
                auto s = std::make_shared<Storage>();
                s->data.resize(0x88);
                u32 magic = 0xC79497CA, flag = 1;
                memcpy(&s->data[0], &magic, 4);
                memcpy(&s->data[4], &flag, 4);
                u64 uid[2] = {0x0000000000000001ull, 0x48574445525F5553ull};
                memcpy(&s->data[8], uid, 16);
                rs.push_object(std::make_shared<IStorage>(s));
                return;
            }
            rs.result = ResultNoData;
        });
        reg(20, [](Request&, Response& rs) { rs.push<u64>(0); });  // EnsureSaveData
        reg(21, [](Request&, Response& rs) {                      // GetDesiredLanguage
            u64 lang = 0;
            memcpy(&lang, "en-US", 5);
            rs.push(lang);
        });
        nop({22, 30, 31, 32, 60, 65, 66, 67, 68, 90, 100, 101, 102, 111, 120, 121, 122, 123, 124, 150, 151});
        reg(23, [](Request&, Response& rs) {  // GetDisplayVersion
            char v[16] = "1.0.0";
            rs.data.insert(rs.data.end(), v, v + 16);
        });
        reg(25, [](Request&, Response& rs) { rs.push<u64>(0); });  // ExtendSaveData
        reg(26, [](Request&, Response& rs) {                      // GetSaveDataSize
            rs.push<u64>(0x10000000);
            rs.push<u64>(0x10000000);
        });
        ret<u8>(40, 0);  // NotifyRunning
        reg(50, [](Request&, Response& rs) {  // GetPseudoDeviceId
            rs.push<u64>(0x4857444552000001ull);
            rs.push<u64>(0x0123456789ABCDEFull);
        });
        reg(110, [](Request&, Response& rs) { rs.push<u32>(0); });
        reg(130, [](Request&, Response& rs) { push_handle(rs, make_event()); });  // GetGpuErrorDetectedSystemEvent
        reg(140, [](Request&, Response& rs) { push_handle(rs, make_event()); });  // GetFriendInvitationStorageChannelEvent
        reg(141, [](Request&, Response& rs) { rs.result = ResultNoData; });
        reg(150, [](Request&, Response& rs) { rs.result = ResultNoData; });
        reg(160, [](Request&, Response& rs) { push_handle(rs, make_event()); });
    }
};

class IDisplayController : public SimpleService {
public:
    IDisplayController() : SimpleService("IDisplayController") {}
};
class IDebugFunctions : public SimpleService {
public:
    IDebugFunctions() : SimpleService("IDebugFunctions") {}
};
class IProcessWindingController : public SimpleService {
public:
    IProcessWindingController() : SimpleService("IProcessWindingController") {
        ret<u32>(0, 0);  // GetLaunchReason
    }
};

class IApplicationProxy : public SimpleService {
public:
    IApplicationProxy() : SimpleService("IApplicationProxy") {
        obj(0, [] { return std::make_shared<ICommonStateGetter>(); });
        obj(1, [] { return std::make_shared<ISelfController>(); });
        obj(2, [] { return std::make_shared<IWindowController>(); });
        obj(3, [] { return std::make_shared<IAudioController>(); });
        obj(4, [] { return std::make_shared<IDisplayController>(); });
        obj(10, [] { return std::make_shared<IProcessWindingController>(); });
        obj(11, [] { return std::make_shared<ILibraryAppletCreator>(); });
        obj(20, [] { return std::make_shared<IApplicationFunctions>(); });
        obj(1000, [] { return std::make_shared<IDebugFunctions>(); });
    }
};

class AppletOE : public SimpleService {
public:
    AppletOE() : SimpleService("appletOE") {
        obj(0, [] {
            display::set_status("applet proxy open");
            return std::make_shared<IApplicationProxy>();
        });
    }
};

}  // namespace

void register_applet_services() { register_as<AppletOE>({"appletOE"}); }

}  // namespace ipc
