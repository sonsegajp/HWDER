// sm: — the service manager. GetServiceHandle opens our in-process implementation.
#include "common.h"
#include "../gpu/display.h"

namespace ipc {

class SmService : public Service {
public:
    SmService() : Service("sm:") {
        reg(0, [](Request&, Response&) {});  // RegisterClient(pid)
        reg(1, [](Request& rq, Response& rs) {  // GetServiceHandle(name) -> session
            u64 raw = rq.pop<u64>();
            char name[9] = {};
            memcpy(name, &raw, 8);
            hw_log("sm: GetService(%s)", name);
            display::set_status(std::string("service ") + name);
            rs.push_object(open_service(name));
        });
        reg(2, [](Request&, Response&) {});  // RegisterService
        reg(3, [](Request&, Response&) {});  // UnregisterService
        reg(4, [](Request&, Response&) {});  // DetachClient
    }
};

void register_all_services() {
    register_service("sm:", [] { return std::make_shared<SmService>(); });
    register_applet_services();
    register_misc_services();
    register_hid_services();
    register_fs_services();
    register_audio_services();
    register_video_services();
    register_gpu_services();
}

}  // namespace ipc
