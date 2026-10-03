// Dear ImGui configuration for HWDER: route asserts to the log instead of the CRT abort dialog.
#pragma once
#include "hwder/runtime.h"
#define IM_ASSERT(_EXPR)                                                                   \
    do {                                                                                   \
        if (!(_EXPR)) hw_log("imgui: assertion failed: %s (%s:%d)", #_EXPR, __FILE__, __LINE__); \
    } while (0)
