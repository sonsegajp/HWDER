# Owned by the shader translator. Paths relative to the repo root.
list(APPEND RUNTIME_SOURCES
    runtime/src/shader/decode.cpp
    runtime/src/shader/fallback.cpp
    runtime/src/shader/translate.cpp
    runtime/src/shader/program.cpp
    runtime/src/shader/emit_alu.cpp
    runtime/src/shader/emit_mem.cpp
)
