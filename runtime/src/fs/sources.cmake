# Owned by the fs subsystem. Paths relative to the repo root.
list(APPEND RUNTIME_SOURCES
    runtime/src/fs/fs.cpp
    runtime/src/fs/romfs.cpp
    runtime/src/fs/hostfs.cpp
    runtime/src/fs/nsp.cpp
)
