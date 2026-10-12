"""Windows process and window operations, restricted to verified worker PIDs."""

import os
from pathlib import Path

GIB = 1024**3


def memory_status():
    if os.name != "nt":
        return None
    import ctypes as c
    from ctypes import wintypes as w

    class Memory(c.Structure):
        _fields_ = [("length", w.DWORD), ("load", w.DWORD)] + [
            (n, c.c_ulonglong)
            for n in (
                "total",
                "available",
                "commit_limit",
                "commit_available",
                "virtual_total",
                "virtual_available",
                "extended",
            )
        ]

    api = c.WinDLL("kernel32", use_last_error=True)
    api.GlobalMemoryStatusEx.argtypes = [c.POINTER(Memory)]
    api.GlobalMemoryStatusEx.restype = w.BOOL
    value = Memory()
    value.length = c.sizeof(value)
    if not api.GlobalMemoryStatusEx(c.byref(value)):
        return None
    return dict(
        total=value.total,
        available=value.available,
        commit_available=value.commit_available,
    )


def process_info(pid):
    if os.name != "nt":
        return None
    import ctypes as c
    from ctypes import wintypes as w

    api = c.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    api.OpenProcess.restype = w.HANDLE
    api.CloseHandle.argtypes = [w.HANDLE]
    api.CloseHandle.restype = w.BOOL
    api.GetExitCodeProcess.argtypes = [w.HANDLE, c.POINTER(w.DWORD)]
    api.GetExitCodeProcess.restype = w.BOOL
    api.QueryFullProcessImageNameW.argtypes = [
        w.HANDLE,
        w.DWORD,
        w.LPWSTR,
        c.POINTER(w.DWORD),
    ]
    api.QueryFullProcessImageNameW.restype = w.BOOL
    api.GetProcessTimes.argtypes = [w.HANDLE] + [c.POINTER(w.FILETIME)] * 4
    api.GetProcessTimes.restype = w.BOOL

    class Counters(c.Structure):
        _fields_ = [("size", w.DWORD), ("faults", w.DWORD)] + [
            (n, c.c_size_t)
            for n in (
                "peak_working",
                "working",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
                "private",
            )
        ]

    api.K32GetProcessMemoryInfo.argtypes = [w.HANDLE, c.POINTER(Counters), w.DWORD]
    api.K32GetProcessMemoryInfo.restype = w.BOOL
    handle = api.OpenProcess(0x410, False, int(pid)) or api.OpenProcess(
        0x1000, False, int(pid)
    )
    if not handle:
        return dict(
            alive=False, error=f"Cannot inspect PID {pid}: {c.get_last_error()}"
        )
    try:
        code = w.DWORD()
        alive = bool(
            api.GetExitCodeProcess(handle, c.byref(code)) and code.value == 259
        )
        name = c.create_unicode_buffer(32768)
        length = w.DWORD(len(name))
        path = (
            name.value
            if api.QueryFullProcessImageNameW(handle, 0, name, c.byref(length))
            else ""
        )
        times = [w.FILETIME() for _ in range(4)]
        created = None
        if api.GetProcessTimes(handle, *(c.byref(t) for t in times)):
            created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        m = Counters()
        m.size = c.sizeof(m)
        measured = api.K32GetProcessMemoryInfo(handle, c.byref(m), m.size)
        return dict(
            alive=alive,
            path=path,
            created=created,
            working_bytes=m.working if measured else 0,
            private_bytes=m.private if measured else 0,
        )
    finally:
        api.CloseHandle(handle)


class WindowAPI:
    def __init__(self):
        if os.name != "nt":
            raise OSError("Game-window controls require Windows.")
        import ctypes as c
        from ctypes import wintypes as w

        self.c, self.w = c, w
        self.api = c.WinDLL("user32", use_last_error=True)
        self.callback = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        declarations = {
            "EnumWindows": ([self.callback, w.LPARAM], w.BOOL),
            "GetWindowThreadProcessId": ([w.HWND, c.POINTER(w.DWORD)], w.DWORD),
            "GetWindowTextW": ([w.HWND, w.LPWSTR, c.c_int], c.c_int),
            "GetClassNameW": ([w.HWND, w.LPWSTR, c.c_int], c.c_int),
            "IsWindowVisible": ([w.HWND], w.BOOL),
            "IsIconic": ([w.HWND], w.BOOL),
            "ShowWindowAsync": ([w.HWND, c.c_int], w.BOOL),
            "SetForegroundWindow": ([w.HWND], w.BOOL),
            "SetWindowTextW": ([w.HWND, w.LPCWSTR], w.BOOL),
            "SetWindowPos": (
                [w.HWND, w.HWND, c.c_int, c.c_int, c.c_int, c.c_int, w.UINT],
                w.BOOL,
            ),
        }
        for name, (args, result) in declarations.items():
            f = getattr(self.api, name)
            f.argtypes = args
            f.restype = result

    def enumerate(self, pid):
        result = []

        @self.callback
        def visit(hwnd, _):
            owner = self.w.DWORD()
            self.api.GetWindowThreadProcessId(hwnd, self.c.byref(owner))
            if owner.value == int(pid):
                title = self.c.create_unicode_buffer(512)
                kind = self.c.create_unicode_buffer(256)
                self.api.GetWindowTextW(hwnd, title, len(title))
                self.api.GetClassNameW(hwnd, kind, len(kind))
                # SDL creates helper/message windows too. Touch only game windows.
                if (
                    "celeste" in title.value.lower()
                    or "fly brain" in title.value.lower()
                    or kind.value == "SDL_app"
                ):
                    result.append(
                        dict(
                            handle=hwnd,
                            visible=bool(self.api.IsWindowVisible(hwnd)),
                            minimized=bool(self.api.IsIconic(hwnd)),
                        )
                    )
            return True

        if not self.api.EnumWindows(visit, 0):
            raise OSError("Windows could not enumerate game windows.")
        return result

    def restore(self, handle, foreground=False):
        if not self.api.ShowWindowAsync(handle, 9):
            raise OSError("Windows rejected restore request.")
        # Bring a restored window onto the visible desktop without resizing it.
        self.api.SetWindowPos(
            handle, 0, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010 | 0x0040 | 0x4000
        )
        if foreground:
            self.api.SetForegroundWindow(handle)

    def hide(self, handle):
        if not self.api.ShowWindowAsync(handle, 0):
            raise OSError("Windows rejected hide request.")

    def title(self, handle, worker):
        self.api.SetWindowTextW(handle, f"Fly Brain - Worker {worker:02d} - Celeste")


def apply_windows(pid, show, worker=0, foreground=False, force=False, api=None):
    api = api or WindowAPI()
    windows = api.enumerate(pid)
    for window in windows:
        handle = window["handle"]
        if show and (force or not window["visible"] or window["minimized"]):
            api.restore(handle, foreground)
        elif not show and window["visible"]:
            api.hide(handle)
        if worker:
            api.title(handle, worker)
    return dict(
        windows=len(windows),
        visible=sum(w["visible"] and not w["minimized"] for w in windows),
        message=(
            ("Restore requested" if show else "Hide requested")
            if windows
            else "No game window found yet"
        ),
    )


def owned_process(row):
    pid = row.get("pid")
    info = process_info(pid) if pid else None
    if not info or not info.get("alive"):
        return False
    if (
        row.get("process_created") is None
        or info.get("created") != row["process_created"]
    ):
        return False
    try:
        return (
            Path(info["path"])
            .resolve()
            .is_relative_to(Path(row["worker_path"]).resolve())
        )
    except (KeyError, ValueError, OSError):
        return False
