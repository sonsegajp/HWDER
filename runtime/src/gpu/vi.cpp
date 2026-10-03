// vi: display service. IHOSBinderDriver speaks android's IGraphicBufferProducer protocol over
// parcels; queued buffers are composited (presented) by our compositor thread at the swap interval.
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "gpu.h"
#include "nvdrv_device.h"
#include "service/common.h"

namespace ipc {
namespace {

// ------------------------------------------------------------------ parcels
class ParcelReader {
public:
    ParcelReader(const u8* buf, u64 size) {
        if (!buf || size < 16) return;
        u32 data_size = *(const u32*)buf, data_off = *(const u32*)(buf + 4);
        if (data_off + data_size > size) data_size = (u32)(size - data_off);
        p_ = buf + data_off;
        end_ = p_ + data_size;
    }
    template <typename T>
    T read() {
        T v{};
        if (p_ + sizeof(T) <= end_) memcpy(&v, p_, sizeof(T));
        p_ += (sizeof(T) + 3) & ~3ull;
        return v;
    }
    void read_bytes(void* dst, u64 n) {
        if (p_ + n <= end_) memcpy(dst, p_, n);
        p_ += (n + 3) & ~3ull;
    }
    void skip_token() {  // u32 strict mode policy, then String16 interface name
        read<u32>();
        u32 len = read<u32>();
        if (len != 0xFFFFFFFF) p_ += ((len + 1) * 2 + 3) & ~3ull;
    }
    // Flattened object: u32 size, u32 fd count, then the object.
    template <typename T>
    T read_flattened() {
        u32 len = read<u32>();
        read<u32>();
        T v{};
        if (p_ + len <= end_) memcpy(&v, p_, std::min<u64>(len, sizeof(T)));
        p_ += (len + 3) & ~3ull;
        return v;
    }

private:
    const u8* p_ = nullptr;
    const u8* end_ = nullptr;
};

class ParcelWriter {
public:
    template <typename T>
    void write(const T& v) {
        const u8* b = (const u8*)&v;
        data_.insert(data_.end(), b, b + sizeof(T));
        while (data_.size() & 3) data_.push_back(0);
    }
    void write_bytes(const void* p, u64 n) {
        const u8* b = (const u8*)p;
        data_.insert(data_.end(), b, b + n);
        while (data_.size() & 3) data_.push_back(0);
    }
    template <typename T>
    void write_flattened(const T& v) {
        write<u32>(sizeof(T));
        write<u32>(0);
        write(v);
    }
    // A binder object: data plus an entry in the object offset table.
    template <typename T>
    void write_interface(const T& v) {
        objects_.push_back((u32)data_.size());
        write(v);
    }
    // Serialize into the guest buffer: header {data_size, data_offset, objects_size, objects_offset}.
    u64 serialize(u8* out, u64 cap) const {
        u32 obj_size = (u32)objects_.size() * 4;
        u32 hdr[4] = {(u32)data_.size(), 0x10, obj_size, 0x10 + (u32)data_.size()};
        u64 total = 0x10 + data_.size() + obj_size;
        if (out && cap >= total) {
            memcpy(out, hdr, 0x10);
            memcpy(out + 0x10, data_.data(), data_.size());
            memcpy(out + 0x10 + data_.size(), objects_.data(), obj_size);
        } else if (out) {
            hw_log("vi: parcel buffer too small (%llu < %llu)", (unsigned long long)cap, (unsigned long long)total);
        }
        return total;
    }

private:
    std::vector<u8> data_;
    std::vector<u32> objects_;
};

struct NvFence {
    u32 id, value;
};
struct NvMultiFence {
    u32 num_fences;
    NvFence fences[4];
};
static_assert(sizeof(NvMultiFence) == 0x24);

#pragma pack(push, 4)
struct QueueBufferInput {
    s64 timestamp;
    s32 is_auto_timestamp;
    s32 crop_left, crop_top, crop_right, crop_bottom;
    s32 scaling_mode;
    u32 transform;
    u32 sticky_transform;
    u32 pad;
    u32 swap_interval;
    NvMultiFence fence;
};
#pragma pack(pop)
static_assert(sizeof(QueueBufferInput) == 0x54);

struct QueueBufferOutput {
    u32 width, height, transform_hint, num_pending_buffers;
};

// android::NvGraphicBuffer (flattened as an object)
struct NvGraphicBuffer {
    u32 magic;
    s32 width, height, stride, format, usage;
    u32 pad0;
    u32 index;
    u32 pad1[3];
    u32 buffer_id;
    u32 pad2[6];
    u32 external_format;
    u32 pad3[10];
    u32 nvmap_handle;
    u32 offset;
    u32 pad4[60];
};
static_assert(sizeof(NvGraphicBuffer) == 0x16C);

// ------------------------------------------------------------------ buffer queue + compositor
constexpr int kSlots = 64;
enum class SlotState { Free, Dequeued, Queued, Acquired };

struct Slot {
    SlotState state = SlotState::Free;
    bool has_buffer = false;
    NvGraphicBuffer buffer{};
    u64 frame = 0;
};

struct Queued {
    int slot;
    QueueBufferInput in;
};

class BufferQueue {
public:
    BufferQueue() {
        thread_ = std::thread([this] { compositor(); });
        thread_.detach();
    }

    std::shared_ptr<kern::ReadableEvent> buffer_event = make_event(true);

    void transact(u32 code, ParcelReader& in, ParcelWriter& out) {
        in.skip_token();
        s32 status = 0;
        switch (code) {
        case 1: {  // RequestBuffer(slot)
            s32 slot = in.read<s32>();
            std::lock_guard<std::mutex> l(m_);
            bool ok = slot >= 0 && slot < kSlots && slots_[slot].has_buffer;
            out.write<s32>(ok ? 1 : 0);
            if (ok) out.write_flattened(slots_[slot].buffer);
            break;
        }
        case 2: {  // SetBufferCount
            in.read<s32>();
            break;
        }
        case 3: {  // DequeueBuffer(async, w, h, format, usage)
            in.read<u32>();
            u32 w = in.read<u32>(), h = in.read<u32>();
            (void)w, (void)h;
            int slot = dequeue();
            out.write<s32>(slot);
            NvMultiFence f{};
            out.write_flattened(f);
            if (slot < 0) status = -11;  // WOULD_BLOCK
            break;
        }
        case 4: {  // DetachBuffer(slot)
            s32 slot = in.read<s32>();
            std::lock_guard<std::mutex> l(m_);
            if (slot >= 0 && slot < kSlots) slots_[slot] = Slot{};
            break;
        }
        case 7: {  // QueueBuffer(slot, QueueBufferInput)
            s32 slot = in.read<s32>();
            QueueBufferInput qbi = in.read_flattened<QueueBufferInput>();
            {
                std::lock_guard<std::mutex> l(m_);
                if (slot >= 0 && slot < kSlots) {
                    slots_[slot].state = SlotState::Queued;
                    queue_.push_back({slot, qbi});
                }
            }
            cv_.notify_all();
            QueueBufferOutput qbo{width_, height_, 0, (u32)pending()};
            if (pace_trace()) {
                static LARGE_INTEGER freq, prev;
                if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
                LARGE_INTEGER now;
                QueryPerformanceCounter(&now);
                hw_log("pace: queue slot %d dt %.2f produce %.2f iv %u fence %u:%u (value %u) pending %u", slot,
                       prev.QuadPart ? (now.QuadPart - prev.QuadPart) * 1000.0 / freq.QuadPart : 0.0,
                       (now.QuadPart - last_dequeue_.QuadPart) * 1000.0 / freq.QuadPart, qbi.swap_interval,
                       qbi.fence.num_fences ? qbi.fence.fences[0].id : 0, qbi.fence.num_fences ? qbi.fence.fences[0].value : 0,
                       qbi.fence.num_fences ? nv::syncpoint_read(qbi.fence.fences[0].id) : 0, qbo.num_pending_buffers);
                prev = now;
            }
            out.write(qbo);
            break;
        }
        case 8: {  // CancelBuffer(slot, fence)
            s32 slot = in.read<s32>();
            {
                std::lock_guard<std::mutex> l(m_);
                if (slot >= 0 && slot < kSlots) slots_[slot].state = SlotState::Free;
            }
            cv_.notify_all();
            buffer_event->signal();
            break;
        }
        case 9: {  // Query(what)
            s32 what = in.read<s32>();
            s32 v = 0;
            switch (what) {
            case 0: v = (s32)width_; break;
            case 1: v = (s32)height_; break;
            case 2: v = 1; break;           // RGBA_8888
            case 3: v = 1; break;           // MIN_UNDEQUEUED_BUFFERS
            case 4: v = 1; break;           // QUEUES_TO_WINDOW_COMPOSER
            default: v = 0; break;
            }
            out.write<s32>(v);
            break;
        }
        case 10: {  // Connect(listener?, api, controlled_by_app)
            QueueBufferOutput qbo{width_, height_, 0, 0};
            out.write(qbo);
            break;
        }
        case 11:  // Disconnect
            break;
        case 14: {  // SetPreallocatedBuffer(slot, has_buffer?, buffer)
            s32 slot = in.read<s32>();
            s32 has = in.read<s32>();
            if (slot >= 0 && slot < kSlots) {
                std::lock_guard<std::mutex> l(m_);
                if (has) {
                    NvGraphicBuffer gb = in.read_flattened<NvGraphicBuffer>();
                    slots_[slot].buffer = gb;
                    slots_[slot].has_buffer = true;
                    slots_[slot].state = SlotState::Free;
                    width_ = (u32)gb.width;
                    height_ = (u32)gb.height;
                    hw_log("vi: buffer slot %d = %dx%d stride %d format %d nvmap %u+0x%x", slot, gb.width, gb.height,
                           gb.stride, gb.format, gb.nvmap_handle, gb.offset);
                } else {
                    slots_[slot] = Slot{};
                }
            }
            buffer_event->signal();
            break;
        }
        default:
            hw_log("vi: IGraphicBufferProducer transaction %u not implemented", code);
            break;
        }
        out.write<s32>(status);
    }

private:
    int pending() {
        std::lock_guard<std::mutex> l(m_);
        return (int)queue_.size();
    }

    static bool pace_trace() {
        static const bool t = getenv("HWDER_PACE_TRACE") != nullptr;
        return t;
    }

    int dequeue() {
        LARGE_INTEGER t0;
        QueryPerformanceCounter(&t0);
        std::unique_lock<std::mutex> l(m_);
        for (;;) {
            // Oldest-presented free slot first (round robin keeps triple buffering in order).
            int best = -1;
            for (int i = 0; i < kSlots; i++)
                if (slots_[i].has_buffer && slots_[i].state == SlotState::Free &&
                    (best < 0 || slots_[i].frame < slots_[best].frame))
                    best = i;
            if (best >= 0) {
                slots_[best].state = SlotState::Dequeued;
                if (!any_free_locked()) buffer_event->clear();
                if (pace_trace()) {
                    LARGE_INTEGER freq, t1;
                    QueryPerformanceFrequency(&freq);
                    QueryPerformanceCounter(&t1);
                    last_dequeue_ = t1;
                    hw_log("pace: dequeue slot %d waited %.2f", best, (t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart);
                }
                return best;
            }
            if (!cv_.wait_for(l, std::chrono::seconds(2), [&] { return any_free_locked(); })) {
                hw_log("vi: DequeueBuffer timed out (no free buffer)");
                return -1;
            }
        }
    }
    bool any_free_locked() const {
        for (const Slot& s : slots_)
            if (s.has_buffer && s.state == SlotState::Free) return true;
        return false;
    }

    void compositor() {
        SetThreadDescription(GetCurrentThread(), L"HWDER compositor");
        int presented = -1;
        u64 frame = 0;
        for (;;) {
            Queued q;
            NvGraphicBuffer gb;
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return !queue_.empty(); });
                q = queue_.front();
                queue_.pop_front();
                slots_[q.slot].state = SlotState::Acquired;
                gb = slots_[q.slot].buffer;
            }
            // Wait for rendering into the buffer to finish. Never present an unfinished buffer: when the GPU
            // thread stalls for more than a second (shader builds, first-time ASTC decodes, frame dumps) a
            // bounded wait would show half-drawn frames (UI layers without their text, clear-colour flashes).
            for (u32 i = 0; i < std::min<u32>(q.in.fence.num_fences, 4); i++) {
                int waited = 0;
                while (!nv::syncpoint_wait(q.in.fence.fences[i].id, q.in.fence.fences[i].value, 1000000000ll)) {
                    if (++waited == 1 || waited % 10 == 0)
                        hw_log("vi: frame fence %u:%u not reached after %d s (value %u) - still waiting", q.in.fence.fences[i].id,
                               q.in.fence.fences[i].value, waited, nv::syncpoint_read(q.in.fence.fences[i].id));
                    if (waited >= 30) break;  // escape hatch for a wedged GPU thread
                }
            }
            gpu::FramebufferInfo fb;
            fb.address = nv::nvmap_address(gb.nvmap_handle) + gb.offset;
            fb.width = (u32)gb.width;
            fb.height = (u32)gb.height;
            fb.stride = (u32)gb.stride;
            fb.format = (u32)gb.format;
            u32 gobs = ((u32)gb.height + 7) / 8, bh = 0;
            while (bh < 4 && (1u << bh) < gobs) bh++;
            fb.block_height_log2 = bh;
            fb.block_linear = true;
            fb.transform_flags = q.in.transform;
            fb.crop_left = q.in.crop_left;
            fb.crop_top = q.in.crop_top;
            fb.crop_right = q.in.crop_right;
            fb.crop_bottom = q.in.crop_bottom;
            fb.swap_interval = q.in.swap_interval ? q.in.swap_interval : 1;
            if (fb.address) gpu::renderer()->present(fb);
            {
                std::lock_guard<std::mutex> l(m_);
                if (presented >= 0 && presented != q.slot && slots_[presented].state == SlotState::Acquired)
                    slots_[presented].state = SlotState::Free;
                slots_[q.slot].frame = ++frame;
                presented = q.slot;
            }
            cv_.notify_all();
            buffer_event->signal();
            if (frame == 1) {
                hw_log("vi: first frame presented (%ux%u)", fb.width, fb.height);
                display::set_status("game running");
            }
        }
    }

    std::mutex m_;
    std::condition_variable cv_;
    Slot slots_[kSlots];
    std::deque<Queued> queue_;
    std::thread thread_;
    u32 width_ = 1280, height_ = 720;
    LARGE_INTEGER last_dequeue_{};  // HWDER_PACE_TRACE
};

BufferQueue& layer_queue() {
    static BufferQueue* q = new BufferQueue;
    return *q;
}

// ------------------------------------------------------------------ vsync
std::shared_ptr<ReadableEvent> vsync_event() {
    static std::shared_ptr<ReadableEvent> ev = [] {
        auto e = make_event();
        hw_log("vi: vsync event requested by the game");
        std::thread([e] {
            SetThreadDescription(GetCurrentThread(), L"HWDER vsync");
            // The game's frame clock: its ticks must not be delayed by busy worker threads.
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
            HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
            LARGE_INTEGER freq, next, now;
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&next);
            const s64 period = freq.QuadPart / 60;
            for (;;) {
                next.QuadPart += period;
                QueryPerformanceCounter(&now);
                s64 wait = next.QuadPart - now.QuadPart;
                if (wait > 0) {
                    LARGE_INTEGER due;
                    due.QuadPart = -(wait * 10000000 / freq.QuadPart);
                    SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0);
                    WaitForSingleObject(timer, INFINITE);
                } else if (wait < -period * 4) {
                    next = now;
                }
                e->signal();
            }
        }).detach();
        return e;
    }();
    return ev;
}

// ------------------------------------------------------------------ services
constexpr s32 kBinderId = 1;

class IHOSBinderDriver : public SimpleService {
public:
    IHOSBinderDriver() : SimpleService("IHOSBinderDriver") {
        auto transact = [](Request& rq, Response&, Buffer in, Buffer out) {
            s32 id = rq.pop<s32>();
            u32 code = rq.pop<u32>();
            ParcelReader pr((const u8*)in.addr, in.size);
            ParcelWriter pw;
            static const bool trace = getenv("HWDER_VI_TRACE") != nullptr;
            if (trace) hw_log("vi: binder %d transact %u", id, code);
            layer_queue().transact(code, pr, pw);
            if (out.addr) pw.serialize((u8*)out.addr, out.size);
        };
        reg(0, [transact](Request& rq, Response& rs) {  // TransactParcel (A in, B out)
            Buffer in = rq.a.empty() ? Buffer{} : rq.a[0];
            Buffer out = rq.b.empty() ? Buffer{} : rq.b[0];
            transact(rq, rs, in, out);
        });
        reg(3, [transact](Request& rq, Response& rs) {  // TransactParcelAuto (X/A in, C/B out)
            transact(rq, rs, rq.in_buffer(0), rq.out_buffer(0));
        });
        nop({1});  // AdjustRefcount
        reg(2, [](Request&, Response& rs) { push_handle(rs, layer_queue().buffer_event); });  // GetNativeHandle
    }
};

class ISystemDisplayService : public SimpleService {
public:
    ISystemDisplayService() : SimpleService("ISystemDisplayService") {
        ret<u64>(1200, 0);    // GetZOrderCountMin
        ret<u64>(1202, 255);  // GetZOrderCountMax
        reg(1203, [](Request&, Response& rs) {  // GetDisplayLogicalResolution
            rs.push<s32>(1920);
            rs.push<s32>(1080);
        });
        nop({2201, 2203, 2205, 2207, 2312});
        ret<u64>(2204, 0);
        reg(3200, [](Request&, Response& rs) {  // GetDisplayMode
            rs.push<u32>(1920);
            rs.push<u32>(1080);
            rs.push<float>(60.0f);
            rs.push<u32>(0);
        });
    }
};

class IManagerDisplayService : public SimpleService {
public:
    IManagerDisplayService() : SimpleService("IManagerDisplayService") {
        ret<u64>(2010, 1);  // CreateManagedLayer
        nop({2011, 2012, 6000, 6001, 6002, 6003, 6004, 7000});
    }
};

u64 write_native_window(Request& rq) {
    ParcelWriter pw;
    struct {
        u32 magic, process_id, id, pad[3];
        char dispdrv[8];
        u32 pad2[2];
    } nw{2, 1, (u32)kBinderId, {}, "dispdrv", {}};
    pw.write_interface(nw);
    Buffer out = rq.out_buffer(0);
    return pw.serialize((u8*)out.addr, out.size);
}

class IApplicationDisplayService : public SimpleService {
public:
    IApplicationDisplayService() : SimpleService("IApplicationDisplayService") {
        obj(100, [] { return std::make_shared<IHOSBinderDriver>(); });
        obj(101, [] { return std::make_shared<ISystemDisplayService>(); });
        obj(102, [] { return std::make_shared<IManagerDisplayService>(); });
        obj(103, [] { return std::make_shared<IHOSBinderDriver>(); });
        reg(1000, [](Request& rq, Response& rs) {  // ListDisplays
            struct {
                char name[0x40];
                u8 has_limited_layers;
                u8 pad[7];
                u64 max_layers, width, height;
            } info{"Default", 1, {}, 1, 1920, 1080};
            write_buffer(rq.out_buffer(), &info, sizeof(info));
            rs.push<u64>(1);
        });
        reg(1010, [](Request&, Response& rs) { rs.push<u64>(0); });  // OpenDisplay
        reg(1011, [](Request&, Response& rs) { rs.push<u64>(0); });  // OpenDefaultDisplay
        nop({1020, 1101, 2021, 2031, 2101});
        reg(1102, [](Request&, Response& rs) {  // GetDisplayResolution
            rs.push<u64>(1920);
            rs.push<u64>(1080);
        });
        reg(2020, [](Request& rq, Response& rs) {  // OpenLayer -> native window parcel
            display::set_status("display layer open");
            rs.push<u64>(write_native_window(rq));
        });
        reg(2030, [](Request& rq, Response& rs) {  // CreateStrayLayer -> layer id, parcel size
            rs.push<u64>(1);
            rs.push<u64>(write_native_window(rq));
        });
        reg(2102, [](Request& rq, Response& rs) { rs.push<u64>(rq.pop<u32>()); });  // ConvertScalingMode
        reg(5202, [](Request&, Response& rs) { push_handle(rs, vsync_event()); });  // GetDisplayVsyncEvent
        reg(5203, [](Request&, Response& rs) { push_handle(rs, vsync_event()); });
    }
};

class ViRoot : public SimpleService {
public:
    explicit ViRoot(const char* name) : SimpleService(name) {
        for (u32 id : {0u, 1u, 2u, 3u}) obj(id, [] { return std::make_shared<IApplicationDisplayService>(); });
    }
};

}  // namespace

void register_vi_services() {
    for (const char* n : {"vi:m", "vi:s", "vi:u"}) register_service(n, [n] { return std::make_shared<ViRoot>(n); });
}

}  // namespace ipc
