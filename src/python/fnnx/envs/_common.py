import os
import platform
import subprocess
from typing import Any


def select_pip_deps(
    raw_deps: list[dict[str, Any]], accelerator: str | None
) -> list[dict[str, Any]]:
    sys_os = platform.system().lower()  # 'linux', 'darwin', 'windows'
    machine = platform.machine().lower()  # 'x86_64', 'arm64', ...
    accel = ("cpu" if accelerator is None else accelerator).lower()

    out: list[dict[str, Any]] = []
    for d in raw_deps:
        cond = d.get("condition") or {}
        ok = True
        if cond.get("os"):
            ok = ok and sys_os in [o.lower() for o in cond["os"]]
        if cond.get("platform"):
            ok = ok and machine in [p.lower() for p in cond["platform"]]
        if cond.get("accelerator"):
            ok = ok and accel in [a.lower() for a in cond["accelerator"]]
        if ok:
            out.append(
                {"package": d["package"], "extra_pip_args": d.get("extra_pip_args")}
            )
    return out


def run_cmd(
    cmd: list[str], *, env: dict[str, str] | None = None, capture: bool = True
) -> str:
    if capture:
        proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )
        return proc.stdout
    else:
        proc = subprocess.run(cmd, env=env, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        return ""


def which(exe: str) -> str | None:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(path, exe)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None
