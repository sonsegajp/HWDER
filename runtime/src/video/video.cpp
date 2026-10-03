// video: nvhost-nvdec / nvhost-vic / nvhost-nvjpg host1x channels, and the mm:u service.
//
// The channels accept command buffers, track their syncpoint increments and complete them
// synchronously. Decoding itself is done by host1x.cpp (methods are routed there per class).
#include <mutex>

#include "gpu/nvdrv_device.h"
#include "host1x.h"
#include "service/common.h"

namespace video {

namespace {

using namespace nv;

// Pinned buffers get a 32-bit device address ("IOVA") so command streams can refer to them.
struct IovaMap {
    std::mutex m;
    std::map<u32, u32> handle_to_iova;
    std::map<u32, std::pair<u64, u64>> iova_to_mem;  // iova -> (guest address, size)
    u32 next = 0x10000000;
} g_iova;

u32 pin(u32 handle) {
    std::lock_guard<std::mutex> l(g_iova.m);
    auto it = g_iova.handle_to_iova.find(handle);
    if (it != g_iova.handle_to_iova.end()) return it->second;
    u64 addr = nvmap_address(handle), size = nvmap_size(handle);
    if (!addr) return 0;
    u32 iova = g_iova.next;
    g_iova.next += (u32)((size + 0xFFFF) & ~0xFFFFull) + 0x10000;
    g_iova.handle_to_iova[handle] = iova;
    g_iova.iova_to_mem[iova] = {addr, size};
    return iova;
}

class Host1xChannel : public Device {
public:
    explicit Host1xChannel(const char* name, u32 class_id) : name_(name), class_id_(class_id) {
        syncpt_ = syncpoint_allocate();
    }

    NvResult ioctl(u32 fd, u32 cmd, IoctlBuffers& b) override {
        u8* p = b.out;
        u32 group = (cmd >> 8) & 0xFF, nr = cmd & 0xFF;
        if (group == 'H' && nr == 0x01) return Success;  // SetNvmapFd
        if (group != 0) return unknown(cmd);
        switch (nr) {
        case 0x01: return submit(p, b.out_size);
        case 0x02:  // GetSyncpoint(param) -> id
            *(u32*)(p + 4) = syncpt_;
            return Success;
        case 0x03:  // GetWaitbase
            *(u32*)(p + 4) = 0;
            return Success;
        case 0x07:  // SetSubmitTimeout
            return Success;
        case 0x09: {  // MapBuffer {num, reserved, attach} + {handle, address}[]
            u32 n = *(u32*)p;
            for (u32 i = 0; i < n && 0xC + (i + 1) * 8 <= b.out_size; i++) {
                u32* e = (u32*)(p + 0xC + i * 8);
                e[1] = pin(e[0]);
            }
            return Success;
        }
        case 0x0A: {  // UnmapBuffer: keep the IOVA (handles are reused across frames)
            u32 n = *(u32*)p;
            for (u32 i = 0; i < n && 0xC + (i + 1) * 8 <= b.out_size; i++) ((u32*)(p + 0xC + i * 8))[1] = 0;
            return Success;
        }
        case 0x23:  // GetClkRate {rate, module}
            *(u32*)p = 192000000;
            return Success;
        default:
            return unknown(cmd);
        }
    }

private:
    NvResult unknown(u32 cmd) {
        hw_log("%s: unknown ioctl 0x%08x", name_, cmd);
        return Success;
    }

    // Submit {cmdbuf_count, reloc_count, syncpt_count, fence_count} followed by
    // cmdbufs{mem, offset, words}[], relocs{4 words}[], reloc_shifts u32[], syncpt_incrs{id, n, 3 unk}[], fences u32[]
    NvResult submit(u8* p, u64 size) {
        u32* h = (u32*)p;
        u32 ncmd = h[0], nreloc = h[1], nsync = h[2], nfence = h[3];
        u64 off = 0x10;
        auto need = [&](u64 n) { return off + n <= size; };
        struct Cmd { u32 mem, offset, words; };
        std::vector<Cmd> cmds;
        for (u32 i = 0; i < ncmd && need(12); i++, off += 12) cmds.push_back(*(Cmd*)(p + off));
        off += (u64)nreloc * 16 + (u64)nreloc * 4;
        u8* incrs = p + off;
        off += (u64)nsync * 20;
        u32* fences = (u32*)(p + off);
        if (!need((u64)nfence * 4)) nfence = 0;

        std::lock_guard<std::mutex> l(m_);
        for (const Cmd& c : cmds) {
            u64 base = nvmap_address(c.mem);
            if (!base) continue;
            host1x::process(class_id_, (const u32*)(base + c.offset), c.words, &state_);
        }
        for (u32 i = 0; i < nsync; i++) {
            u32 id = *(u32*)(incrs + i * 20), n = *(u32*)(incrs + i * 20 + 4);
            u32 thresh = syncpoint_reserve(id, n);
            for (u32 k = 0; k < n; k++) syncpoint_increment(id);
            if (i < nfence) fences[i] = thresh;
        }
        return Success;
    }

    const char* name_;
    u32 class_id_;
    u32 syncpt_ = 0;
    std::mutex m_;
    host1x::ChannelState state_;
};

}  // namespace

bool iova_to_guest(u32 iova, u64* addr, u64* size) {
    std::lock_guard<std::mutex> l(g_iova.m);
    auto it = g_iova.iova_to_mem.upper_bound(iova);
    if (it == g_iova.iova_to_mem.begin()) return false;
    --it;
    if (iova >= it->first + it->second.second) return false;
    *addr = it->second.first + (iova - it->first);
    *size = it->second.second - (iova - it->first);
    return true;
}

}  // namespace video

namespace video {
void prewarm_decoder();
}
namespace ipc {

namespace {

// mm:u - multimedia clock/resource manager. Requests are granted immediately.
class MmU : public SimpleService {
public:
    MmU() : SimpleService("mm:u") {
        nop({0, 1, 2, 5, 6});
        ret<u32>(3, 0);  // GetOld
        reg(4, [this](Request&, Response& rs) { rs.push<u32>(next_id_++); });  // Initialize -> id
        reg(7, [](Request&, Response& rs) { rs.push<u32>(0); });              // Get(id) -> setting
    }

private:
    u32 next_id_ = 1;
};

}  // namespace

void register_video_services() {
    video::prewarm_decoder();  // create the Media Foundation decoder now, not 200 ms into the first movie
    nv::register_device("/dev/nvhost-nvdec", [] { return std::make_shared<video::Host1xChannel>("nvdec", 0xF0); });
    nv::register_device("/dev/nvhost-vic", [] { return std::make_shared<video::Host1xChannel>("vic", 0x5D); });
    nv::register_device("/dev/nvhost-nvjpg", [] { return std::make_shared<video::Host1xChannel>("nvjpg", 0xC0); });
    register_as<MmU>({"mm:u"});
}

}  // namespace ipc
