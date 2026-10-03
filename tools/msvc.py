"""Run commands inside the Visual Studio x64 developer environment.

Paths are passed through environment variables (HW_*) so that non-ASCII characters
in the project path survive cmd.exe.
"""
import os
import subprocess
import tempfile

VCVARS = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
_env_cache = None


def vs_env():
    global _env_cache
    if _env_cache is None:
        out = subprocess.run(f'cmd /c ""{VCVARS}" >nul && set"', capture_output=True, text=True,
                             shell=False).stdout
        env = {}
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
        if "INCLUDE" not in env:
            raise RuntimeError("failed to load Visual Studio environment")
        _env_cache = env
    return dict(_env_cache)


def run(args, cwd=None, **kw):
    """Run an argv list (no shell) with the VS environment."""
    env = vs_env()
    exe = args[0]
    for d in env["PATH"].split(";"):
        cand = os.path.join(d, exe if exe.endswith(".exe") else exe + ".exe")
        if os.path.exists(cand):
            args = [cand] + list(args[1:])
            break
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, **kw)
