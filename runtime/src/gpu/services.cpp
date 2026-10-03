// GPU-side services: nvdrv (driver) and vi (display).
#include "service/common.h"

namespace ipc {

void register_nvdrv_services();
void register_vi_services();

void register_gpu_services() {
    register_nvdrv_services();
    register_vi_services();
}

}  // namespace ipc
