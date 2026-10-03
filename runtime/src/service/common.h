// Helpers shared by service implementations.
#pragma once
#include <cstring>
#include <deque>

#include "../gpu/display.h"
#include "ipc.h"

namespace ipc {

using kern::g_handles;
using kern::ReadableEvent;
using kern::SharedMemory;

constexpr Result ModuleResult(u32 module, u32 desc) { return module | (desc << 9); }

inline std::shared_ptr<ReadableEvent> make_event(bool signaled = false) {
    auto e = std::make_shared<ReadableEvent>();
    e->signaled = signaled;
    return e;
}
// Copy-handle semantics: every call hands out a new handle to the same kernel object.
inline void push_handle(Response& rs, kern::ObjectPtr obj) { rs.copy_handles.push_back(g_handles.add(std::move(obj))); }

inline void write_buffer(const Buffer& b, const void* src, size_t n) {
    if (!b.addr) return;
    memcpy((void*)b.addr, src, n < b.size ? n : (size_t)b.size);
}

// Service whose commands that return an interface are declared with obj().
class SimpleService : public Service {
public:
    using Service::Service;

protected:
    template <typename F>
    void obj(u32 id, F factory) {
        reg(id, [factory](Request&, Response& rs) { rs.push_object(factory()); });
    }
    void nop(std::initializer_list<u32> ids) {
        for (u32 id : ids) reg(id, [](Request&, Response&) {});
    }
    template <typename T>
    void ret(u32 id, T v) {
        reg(id, [v](Request&, Response& rs) { rs.push(v); });
    }
};

// Register a service factory under several names.
template <typename T>
void register_as(std::initializer_list<const char*> names) {
    for (const char* n : names) register_service(n, [] { return std::make_shared<T>(); });
}

void register_applet_services();
void register_misc_services();
void register_hid_services();
void register_fs_services();
void register_audio_services();
void register_gpu_services();
void register_video_services();  // before gpu: registers nvdrv devices

}  // namespace ipc
