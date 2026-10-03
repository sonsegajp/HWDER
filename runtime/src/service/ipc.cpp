// HIPC/CMIF message handling. Layout reference: switchbrew "IPC_Marshalling".
#include <windows.h>

#include "ipc.h"

#include <cstring>

extern "C" void hw_log_backtrace(const char* tag);

namespace ipc {

using namespace kern;

constexpr u32 kSfci = 0x49434653;  // 'SFCI'
constexpr u32 kSfco = 0x4F434653;  // 'SFCO'

// ------------------------------------------------------------------ sessions
// Server-side domain state. Clones of a domain session share it (same object ids).
struct Domain {
    bool active = false;
    std::map<u32, ServicePtr> objects;  // object id -> service
    u32 next_object = 1;
    u32 add(ServicePtr s) {
        u32 id = next_object++;
        objects[id] = std::move(s);
        return id;
    }
};

struct Session : Object {
    const char* type_name() const override { return "Session"; }
    ServicePtr service;
    std::shared_ptr<Domain> dom = std::make_shared<Domain>();
};

Handle new_session(ServicePtr s) {
    auto sess = std::make_shared<Session>();
    sess->service = std::move(s);
    return g_handles.add(sess);
}
static Handle clone_session(const Session& from) {
    auto sess = std::make_shared<Session>();
    sess->service = from.service;
    sess->dom = from.dom;
    return g_handles.add(sess);
}

// ------------------------------------------------------------------ services
static std::map<std::string, std::function<ServicePtr()>>& registry() {
    static std::map<std::string, std::function<ServicePtr()>> r;
    return r;
}
void register_service(const std::string& name, std::function<ServicePtr()> factory) { registry()[name] = std::move(factory); }
ServicePtr open_service(const std::string& name) {
    auto it = registry().find(name);
    if (it != registry().end()) return it->second();
    hw_log("ipc: service '%s' not implemented - using stub", name.c_str());
    return std::make_shared<StubService>(name);
}

void Service::dispatch(Request& rq, Response& rs) {
    auto it = commands_.find(rq.command);
    if (it == commands_.end()) {
        hw_log("ipc: %s command %u not implemented (stubbed)", name_.c_str(), rq.command);
        return;
    }
    it->second(rq, rs);
}

void StubService::dispatch(Request& rq, Response& rs) {
    hw_log("ipc: [stub %s] command %u (args %u bytes, A%zu B%zu X%zu C%zu)", name_.c_str(), rq.command, rq.args_size,
           rq.a.size(), rq.b.size(), rq.x.size(), rq.c.size());
}

Buffer Request::in_buffer(size_t i) const {
    if (i < a.size() && a[i].addr) return a[i];
    if (i < x.size()) return x[i];
    return {};
}
Buffer Request::out_buffer(size_t i) const {
    if (i < b.size() && b[i].addr) return b[i];
    if (i < c.size()) return c[i];
    return {};
}
std::string Request::in_string(size_t i) const {
    Buffer bf = in_buffer(i);
    if (!bf.addr) return {};
    const char* s = (const char*)bf.addr;
    return std::string(s, strnlen(s, (size_t)bf.size));
}

// ------------------------------------------------------------------ message parsing
struct Parsed {
    Request rq;
    bool domain_close = false;
    u32 domain_object = 0;
    u32 domain_cmd = 0;
};

static void parse(const u8* msg, Parsed& p, bool is_domain) {
    const u32* w = (const u32*)msg;
    u32 w0 = w[0], w1 = w[1];
    Request& rq = p.rq;
    rq.type = w0 & 0xFFFF;
    u32 nx = (w0 >> 16) & 0xF, na = (w0 >> 20) & 0xF, nb = (w0 >> 24) & 0xF, nw = (w0 >> 28) & 0xF;
    u32 raw_words = w1 & 0x3FF;
    u32 c_flags = (w1 >> 10) & 0xF;
    bool has_handles = (w1 >> 31) & 1;
    const u32* cur = w + 2;
    if (has_handles) {
        u32 hd = *cur++;
        rq.send_pid = hd & 1;
        u32 ncopy = (hd >> 1) & 0xF, nmove = (hd >> 5) & 0xF;
        if (rq.send_pid) {
            rq.pid = *(const u64*)cur;
            cur += 2;
        }
        for (u32 i = 0; i < ncopy; i++) rq.copy_handles.push_back(*cur++);
        for (u32 i = 0; i < nmove; i++) rq.move_handles.push_back(*cur++);
    }
    for (u32 i = 0; i < nx; i++) {
        u32 d0 = cur[0], d1 = cur[1];
        cur += 2;
        u64 addr = d1 | ((u64)((d0 >> 12) & 0xF) << 32) | ((u64)((d0 >> 6) & 0x7) << 36);
        rq.x.push_back({addr, d0 >> 16});
        rq.x_index.push_back((d0 & 0x3F) | ((d0 >> 3) & 0xE00));
    }
    auto abw = [&](std::vector<Buffer>& v, u32 n) {
        for (u32 i = 0; i < n; i++) {
            u32 s0 = cur[0], a0 = cur[1], d2 = cur[2];
            cur += 3;
            u64 size = s0 | ((u64)((d2 >> 24) & 0xF) << 32);
            u64 addr = a0 | ((u64)((d2 >> 28) & 0xF) << 32) | ((u64)((d2 >> 2) & 0x7) << 36);
            v.push_back({addr, size});
        }
    };
    abw(rq.a, na);
    abw(rq.b, nb);
    abw(rq.w, nw);
    const u8* raw = (const u8*)cur;
    const u8* raw_end = raw + raw_words * 4;
    // CMIF payload is 16-byte aligned relative to the message.
    size_t off = (size_t)(raw - msg);
    const u8* body = msg + ((off + 15) & ~(size_t)15);
    if (is_domain && (rq.type == 4 || rq.type == 6)) {
        u8 dcmd = body[0];
        u8 nobj = body[1];
        u16 len = *(const u16*)(body + 2);
        p.domain_cmd = dcmd;
        p.domain_object = *(const u32*)(body + 4);
        p.domain_close = dcmd == 2;
        body += 16;
        const u32* objs = (const u32*)(body + len);
        for (u32 i = 0; i < nobj; i++) rq.in_objects.push_back(objs[i]);
    }
    if (*(const u32*)body == kSfci || rq.type == 4 || rq.type == 5 || rq.type == 6 || rq.type == 7) {
        rq.command = *(const u32*)(body + 8);
        rq.args = body + 16;
        rq.args_size = (u32)(raw_end > rq.args ? raw_end - rq.args : 0);
    }
    // C descriptors follow the raw data.
    const u32* cd = (const u32*)raw_end;
    u32 nc = c_flags == 0 || c_flags == 1 ? 0 : (c_flags == 2 ? 1 : c_flags - 2);
    for (u32 i = 0; i < nc; i++) {
        u64 addr = cd[0] | ((u64)(cd[1] & 0xFFFF) << 32);
        rq.c.push_back({addr, cd[1] >> 16});
        cd += 2;
    }
}

static void write_response(u8* msg, const Response& rs, bool domain, const std::vector<u32>& domain_objs,
                           const std::vector<Handle>& extra_moves) {
    std::vector<Handle> moves = rs.move_handles;
    moves.insert(moves.end(), extra_moves.begin(), extra_moves.end());
    bool has_handles = !rs.copy_handles.empty() || !moves.empty();
    u32* w = (u32*)msg;
    u32* cur = w + 2;
    if (has_handles) {
        *cur++ = ((u32)rs.copy_handles.size() << 1) | ((u32)moves.size() << 5);
        for (Handle h : rs.copy_handles) *cur++ = h;
        for (Handle h : moves) *cur++ = h;
    }
    size_t off = (size_t)((u8*)cur - msg);
    u8* body = msg + ((off + 15) & ~(size_t)15);
    u8* p = body;
    if (domain) {
        memset(p, 0, 16);
        *(u32*)p = (u32)domain_objs.size();
        p += 16;
    }
    u32* hdr = (u32*)p;
    hdr[0] = kSfco;
    hdr[1] = 0;
    hdr[2] = rs.result;
    hdr[3] = 0;
    p += 16;
    memcpy(p, rs.data.data(), rs.data.size());
    p += (rs.data.size() + 3) & ~(size_t)3;
    for (u32 id : domain_objs) {
        *(u32*)p = id;
        p += 4;
    }
    u32 raw_words = (u32)(((p - (u8*)cur) + 3) / 4) + 4;  // include alignment padding slack
    w[0] = 0;
    w[1] = (raw_words & 0x3FF) | (has_handles ? 0x80000000u : 0);
}

// ------------------------------------------------------------------ entry points
Result connect_to_named_port(const char* name, Handle* out) {
    std::string n(name, strnlen(name, 12));
    if (n != "sm:") hw_log("ipc: ConnectToNamedPort(%s)", n.c_str());
    *out = new_session(open_service(n));
    return ResultSuccess;
}

static void dump_msg(const char* tag, const u8* msg) {
    char line[0x100 * 3 + 1];
    int n = 0;
    for (int i = 0; i < 0x80; i++) n += snprintf(line + n, sizeof(line) - n, "%02x%s", msg[i], (i & 3) == 3 ? " " : "");
    hw_log("ipc %s: %s", tag, line);
}

Result send_sync_request(Handle h, u8* msg) {
    static const char* dump_filter = getenv("HWDER_IPC_DUMP");
    auto sess = g_handles.get_as<Session>(h);
    if (!sess) {
        hw_log("ipc: SendSyncRequest on invalid handle 0x%x", h);
        return ResultInvalidHandle;
    }
    Parsed p;
    parse(msg, p, sess->dom->active);
    Request& rq = p.rq;
    Response rs;
    std::vector<u32> domain_out;
    std::vector<Handle> extra_moves;

    if (rq.type == 2) {  // Close
        g_handles.close(h);
        return ResultSuccess;
    }
    if (rq.type == 5 || rq.type == 7) {  // Control
        static const bool trace = getenv("HWDER_IPC_TRACE") != nullptr;
        if (trace) hw_log("ipc: control #%u on handle 0x%x (%s, domain=%d)", rq.command, h, sess->service->name().c_str(), sess->dom->active);
        switch (rq.command) {
        case 0:  // ConvertCurrentObjectToDomain
            sess->dom->active = true;
            rs.push<u32>(sess->dom->add(sess->service));
            break;
        case 1: {  // CopyFromCurrentDomain(object id) -> move handle
            u32 id = rq.pop<u32>();
            auto it = sess->dom->objects.find(id);
            if (it == sess->dom->objects.end()) rs.result = ResultInvalidHandle;
            else extra_moves.push_back(new_session(it->second));
            break;
        }
        case 2: case 4:  // CloneCurrentObject(Ex): the clone shares the domain
            extra_moves.push_back(clone_session(*sess));
            break;
        case 3:  // QueryPointerBufferSize
            rs.push<u16>(0x8000);
            break;
        default:
            hw_log("ipc: control command %u", rq.command);
        }
        write_response(msg, rs, false, {}, extra_moves);
        return ResultSuccess;
    }
    ServicePtr target = sess->service;
    if (sess->dom->active) {
        auto it = sess->dom->objects.find(p.domain_object);
        if (it == sess->dom->objects.end()) {
            hw_log("ipc: domain object %u not found", p.domain_object);
            rs.result = ResultInvalidHandle;
            write_response(msg, rs, true, {}, {});
            return ResultSuccess;
        }
        if (p.domain_close) {
            sess->dom->objects.erase(it);
            write_response(msg, rs, true, {}, {});
            return ResultSuccess;
        }
        target = it->second;
    }
    bool dump = dump_filter && target->name().find(dump_filter) != std::string::npos;
    if (dump) dump_msg("req ", msg);
    {
        // A service call that takes long blocks the guest thread that issued it: report it.
        LARGE_INTEGER d0, d1, df;
        QueryPerformanceCounter(&d0);
        target->dispatch(rq, rs);
        QueryPerformanceCounter(&d1);
        QueryPerformanceFrequency(&df);
        double ms = (d1.QuadPart - d0.QuadPart) * 1000.0 / df.QuadPart;
        if (ms > 10.0) hw_log("ipc: slow: %s command %u took %.0f ms", target->name().c_str(), rq.command, ms);
    }
    static const bool trace = getenv("HWDER_IPC_TRACE") != nullptr;
    if (trace) {
        char hex[3 * 16 + 1] = {};
        for (size_t i = 0; i < rs.data.size() && i < 16; i++) snprintf(hex + i * 3, 4, " %02x", rs.data[i]);
        hw_log("ipc: %s #%u -> 0x%x (%zu bytes, %zu objs, %zu handles)%s", target->name().c_str(), rq.command, rs.result,
               rs.data.size(), rs.out_objects.size(), rs.copy_handles.size() + rs.move_handles.size(), hex);
    }
    static const char* bt_filter = getenv("HWDER_IPC_BT");  // e.g. "IFile#4"
    if (bt_filter) {
        std::string key = target->name() + "#" + std::to_string(rq.command);
        if (key == bt_filter) hw_log_backtrace(key.c_str());
    }
    for (auto& obj : rs.out_objects) {
        if (sess->dom->active) domain_out.push_back(sess->dom->add(obj));
        else extra_moves.push_back(new_session(obj));
    }
    write_response(msg, rs, sess->dom->active, domain_out, extra_moves);
    if (dump) dump_msg("resp", msg);
    return ResultSuccess;
}

}  // namespace ipc
