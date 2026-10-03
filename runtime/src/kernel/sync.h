// Kernel waiting primitives (implemented in kernel.cpp).
#pragma once
#include "kernel.h"

namespace kern {
Result arbitrate_lock(Handle owner, u64 addr, Handle self);
Result arbitrate_unlock(u64 addr);
Result wait_process_wide_key(u64 mutex_addr, u64 key, Handle self, s64 timeout_ns);
void signal_process_wide_key(u64 key, s32 count);
Result wait_for_address(u64 addr, u32 type, s32 value, s64 timeout_ns);
Result signal_to_address(u64 addr, u32 type, s32 value, s32 count);
Result wait_synchronization(const Handle* handles, s32 count, s64 timeout_ns, s32* out_index);
}  // namespace kern
