// HOS IPC: HIPC message framing + CMIF command protocol (incl. domains), served in-process.
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../kernel/kernel.h"

namespace ipc {

using kern::Handle;
using kern::Result;

struct Buffer {
    u64 addr = 0, size = 0;
};

class Service;
using ServicePtr = std::shared_ptr<Service>;

// A parsed incoming request.
struct Request {
    u32 type = 0;      // 4 Request, 5 Control, 2 Close, 6/7 with context
    u32 command = 0;
    bool send_pid = false;
    u64 pid = 0;
    std::vector<Handle> copy_handles, move_handles;
    std::vector<Buffer> x, a, b, w, c;
    std::vector<u32> x_index;
    const u8* args = nullptr;  // CMIF payload after the header
    u32 args_size = 0;
    u32 cursor = 0;
    std::vector<u32> in_objects;  // domain input object ids

    template <typename T>
    T pop() {
        cursor = (cursor + (u32)alignof(T) - 1) & ~((u32)alignof(T) - 1);
        T v{};
        if (cursor + sizeof(T) <= args_size) memcpy(&v, args + cursor, sizeof(T));
        cursor += sizeof(T);
        return v;
    }
    // Input buffer N of "type 0x5 / 0x21 / 0x9" kinds: prefer A, fall back to X.
    Buffer in_buffer(size_t i = 0) const;
    Buffer out_buffer(size_t i = 0) const;  // B, fall back to C
    std::string in_string(size_t i = 0) const;
};

// The reply being built.
struct Response {
    Result result = 0;
    std::vector<u8> data;
    std::vector<Handle> copy_handles, move_handles;
    std::vector<ServicePtr> out_objects;  // returned interfaces

    template <typename T>
    void push(const T& v) {
        size_t a = alignof(T) < 4 ? alignof(T) : (alignof(T) > 8 ? 8 : alignof(T));
        while (data.size() % a) data.push_back(0);
        const u8* p = (const u8*)&v;
        data.insert(data.end(), p, p + sizeof(T));
    }
    void push_object(ServicePtr s) { out_objects.push_back(std::move(s)); }
};

using CommandFn = std::function<void(Request&, Response&)>;

class Service : public std::enable_shared_from_this<Service> {
public:
    explicit Service(std::string name) : name_(std::move(name)) {}
    virtual ~Service() = default;
    const std::string& name() const { return name_; }
    virtual void dispatch(Request& rq, Response& rs);

protected:
    void reg(u32 id, CommandFn fn) { commands_[id] = std::move(fn); }
    std::map<u32, CommandFn> commands_;
    std::string name_;
};

// Service that logs every command and replies success with no data (used for anything we have not
// implemented yet, so the game tells us what it needs).
class StubService : public Service {
public:
    using Service::Service;
    void dispatch(Request& rq, Response& rs) override;
};

// Registry for sm:GetService
void register_service(const std::string& name, std::function<ServicePtr()> factory);
ServicePtr open_service(const std::string& name);
void register_all_services();

Result connect_to_named_port(const char* name, Handle* out);
Result send_sync_request(Handle h, u8* message);
Handle new_session(ServicePtr s);

}  // namespace ipc
