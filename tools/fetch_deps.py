"""Fetch third-party sources into third_party/ (see third_party/README.md).

In a git checkout prefer `git submodule update --init --recursive`; this script is for source
archives without submodule metadata.
"""
import os
import subprocess

TP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")

DEPS = [
    ("imgui", "https://github.com/ocornut/imgui.git"),
    ("spirv-headers", "https://github.com/KhronosGroup/SPIRV-Headers.git"),
    ("vulkan-headers", "https://github.com/KhronosGroup/Vulkan-Headers.git"),
]


def main():
    os.makedirs(TP, exist_ok=True)
    for name, url in DEPS:
        dst = os.path.join(TP, name)
        if os.path.isdir(dst) and os.listdir(dst):
            continue
        subprocess.run(["git", "clone", "--depth", "1", url, name], cwd=TP, check=True)


if __name__ == "__main__":
    main()
