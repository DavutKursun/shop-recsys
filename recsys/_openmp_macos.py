"""macOS: one OpenMP runtime for torch, faiss-cpu and lightgbm.

Each of the three wheels ships its own libomp (torch/lib/, faiss/.dylibs/, LightGBM via @rpath → Homebrew).
Two copies in one process are not isolated: dyld coalesces their C++ template code (weak symbols) while each
keeps its own global state, so the first parallel region of the second library aborts ("OMP: Error #15"),
hangs in a barrier, or segfaults in `__kmp_suspend_initialize_thread`. KMP_DUPLICATE_LIB_OK=TRUE only
silences the abort; the hang and the segfault stay. The fix is to load ONE image: faiss and scikit-learn's
bundled copies become symlinks to torch's libomp, and LightGBM's `@rpath/libomp.dylib` load command is
rewritten to torch's copy (then re-signed ad hoc). `python scripts/fix_openmp_macos.py` does this once per
venv; it is idempotent, and a pip reinstall or upgrade of any of the three undoes it (run it again).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import struct
import subprocess
import sys
import warnings
from pathlib import Path

FIX_COMMAND = "python scripts/fix_openmp_macos.py"
MH_MAGIC_64 = 0xFEEDFACF
LC_LOAD_DYLIB_CMDS = {0xC, 0x80000018, 0x8000001F, 0x80000023}  # LOAD, LOAD_WEAK, REEXPORT, LOAD_UPWARD
LC_RPATH = 0x8000001C
BUNDLED = ("faiss", "sklearn")  # wheels that carry <pkg>/.dylibs/libomp.dylib (delocate)
PROBE = """
import numpy as np, torch, faiss, lightgbm as lgb
(torch.randn(256, 256) @ torch.randn(256, 256)).sum().item()
x = np.random.default_rng(0).random((2000, 32), dtype=np.float32)
ix = faiss.IndexFlatIP(32); ix.add(x); ix.search(x[:5], 10)
X = np.random.default_rng(0).random((500, 5)); y = (X[:, 0] > 0.5).astype(int)
lgb.train({"objective": "binary", "verbose": -1, "num_threads": 4}, lgb.Dataset(X, y), 5)
print("OPENMP_PROBE_OK")
"""


# --- Mach-O load commands (what `otool -L` and `otool -l | grep RPATH` print), without importing anything ------------
def _load_commands(binary: Path) -> list[tuple[int, bytes]]:
    with open(binary, "rb") as f:
        head = f.read(32)
        if len(head) < 32 or struct.unpack("<I", head[:4])[0] != MH_MAGIC_64:
            return []  # the wheels ship thin 64-bit binaries; anything else is left alone
        ncmds, sizeofcmds = struct.unpack("<II", head[16:24])
        cmds = f.read(sizeofcmds)
    out, off = [], 0
    for _ in range(ncmds):
        if off + 8 > len(cmds):
            break
        cmd, size = struct.unpack("<II", cmds[off : off + 8])
        out.append((cmd, cmds[off : off + size]))
        off += size
    return out


def _lc_str(payload: bytes) -> str:
    offset = struct.unpack("<I", payload[8:12])[0]
    return payload[offset:].split(b"\0", 1)[0].decode(errors="replace")


def dylib_dependencies(binary: Path) -> list[str]:
    return [_lc_str(p) for cmd, p in _load_commands(binary) if cmd in LC_LOAD_DYLIB_CMDS]


def rpaths(binary: Path) -> list[str]:
    return [_lc_str(p) for cmd, p in _load_commands(binary) if cmd == LC_RPATH]


def resolve(binary: Path, dep: str) -> Path | None:
    """Where dyld would find `dep` when loading `binary` (None if it cannot be resolved statically)."""
    if dep.startswith("@loader_path/"):
        return binary.parent / dep[len("@loader_path/") :]
    if dep.startswith("@rpath/"):
        leaf = dep[len("@rpath/") :]
        for rp in rpaths(binary):
            cand = Path(rp.replace("@loader_path", str(binary.parent))) / leaf
            if cand.exists():
                return cand
        return None
    if dep.startswith("@"):
        return None
    return Path(dep)


# --- the packages ---------------------------------------------------------------------------------------------------
def _package_dir(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)  # does not import the package
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def torch_libomp() -> Path | None:
    d = _package_dir("torch")
    p = d / "lib" / "libomp.dylib" if d else None
    return p if p is not None and p.exists() else None


def _bundled_copies() -> list[Path]:
    out = []
    for pkg in BUNDLED:
        d = _package_dir(pkg)
        if d and ((d / ".dylibs" / "libomp.dylib").is_symlink() or (d / ".dylibs" / "libomp.dylib").exists()):
            out.append(d / ".dylibs" / "libomp.dylib")
    return out


def _omp_binaries() -> list[Path]:
    out = []
    d = _package_dir("lightgbm")
    if d:
        out += sorted((d / "lib").glob("lib_lightgbm*.dylib"))
    d = _package_dir("faiss")
    if d:
        out += sorted(d.glob("libfaiss*.dylib")) + sorted(d.glob("_swigfaiss*.so"))
    return out


def status() -> dict:
    """Which libomp images torch, faiss, lightgbm and scikit-learn would load into one process."""
    st = {"applicable": sys.platform == "darwin", "runtime": None, "images": {}, "extra": [], "needs_fix": False}
    if not st["applicable"]:
        return st
    ref = torch_libomp()
    if ref is None:
        return st
    st["runtime"] = str(ref)
    ref_real = os.path.realpath(ref)
    for copy in _bundled_copies():
        st["images"][str(copy)] = os.path.realpath(copy)
    for binary in _omp_binaries():
        for dep in dylib_dependencies(binary):
            if "libomp" in dep:
                target = resolve(binary, dep)
                st["images"][str(binary)] = os.path.realpath(target) if target else f"unresolved {dep}"
    st["extra"] = sorted({loaded for loaded in st["images"].values() if loaded != ref_real})
    st["needs_fix"] = bool(st["extra"])
    return st


def warn_if_two_runtimes() -> None:
    """Called from recsys/__init__ on macOS: a warning now beats a segfault with no traceback later."""
    try:
        st = status()
    except (OSError, ValueError, ImportError, struct.error):  # a diagnostic must never break an import
        return
    if st["needs_fix"]:
        msg = (f"{len(st['extra']) + 1} OpenMP runtimes would load into this process (torch, faiss-cpu and lightgbm each bundle one); "
               f"on macOS that aborts, hangs or segfaults Python. Run once: {FIX_COMMAND}")
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


# --- the fix ----------------------------------------------------------------------------------------------------------
def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise SystemExit(f"{cmd[0]} not found; install the Xcode command line tools: xcode-select --install") from e
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"{' '.join(cmd)} failed: {e.stderr.strip()}") from e


def apply() -> list[str]:
    """Point every libomp reference at torch's copy. Returns one line per change."""
    ref = torch_libomp()
    if ref is None:
        raise SystemExit("torch is not installed in this interpreter; pip install torch first")
    ref_real, changed = os.path.realpath(ref), []
    for copy in _bundled_copies():
        if os.path.realpath(copy) != ref_real:
            rel = os.path.relpath(ref, copy.parent)
            copy.unlink()
            copy.symlink_to(rel)
            changed.append(f"{copy} -> {rel}")
    for binary in _omp_binaries():
        for dep in dylib_dependencies(binary):
            target = resolve(binary, dep) if "libomp" in dep else None
            if "libomp" in dep and (target is None or os.path.realpath(target) != ref_real):
                new = "@loader_path/" + os.path.relpath(ref, binary.parent)
                _run(["install_name_tool", "-change", dep, new, str(binary)])
                _run(["codesign", "--force", "--sign", "-", str(binary)])  # the edit invalidates the ad-hoc signature
                changed.append(f"{binary}: {dep} -> {new}")
    return changed


def verify(timeout: int = 180) -> dict:
    """torch, faiss and LightGBM doing real work in one fresh process; counts the libomp images dyld loaded."""
    env = {**os.environ, "DYLD_PRINT_LIBRARIES": "1"}
    try:
        p = subprocess.run([sys.executable, "-c", PROBE], check=False, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": f"hung for {timeout}s (two OpenMP runtimes deadlock in a barrier)", "images": []}
    images = sorted({line.split()[-1] for line in p.stderr.splitlines() if "libomp" in line})
    ok = p.returncode == 0 and "OPENMP_PROBE_OK" in p.stdout
    detail = "ok" if ok else (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1]
    return {"ok": ok, "detail": detail, "images": images}


def _short(path: str) -> str:
    return path.split("site-packages/")[-1] if "site-packages/" in path else path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="macOS: make torch, faiss-cpu and lightgbm share one OpenMP runtime (no-op elsewhere)")
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if more than one runtime would load")
    ap.add_argument("--no-verify", action="store_true", help="skip the end-to-end probe in a fresh process")
    a = ap.parse_args(argv)
    st = status()
    if not st["applicable"]:
        print(f"{sys.platform}: not macOS, nothing to do")
        return 0
    if st["runtime"] is None:
        print("torch is not installed in this interpreter; install it first")
        return 1
    print(f"runtime: {_short(st['runtime'])}")
    for who, loaded in st["images"].items():
        print(f"  {_short(who):48s} -> {_short(loaded)}{'' if loaded == os.path.realpath(st['runtime']) else '   <- a second runtime'}")
    if a.check:
        print("one runtime: OK" if not st["needs_fix"] else f"{len(st['extra']) + 1} runtimes would load; run: {FIX_COMMAND}")
        return 1 if st["needs_fix"] else 0
    if st["needs_fix"]:
        for line in apply():
            print("  changed " + _short(line))
        st = status()
        if st["needs_fix"]:
            print("still more than one runtime: " + ", ".join(_short(x) for x in st["extra"]))
            return 1
        print("now one runtime")
    else:
        print("already one runtime; nothing changed")
    if a.no_verify:
        return 0
    v = verify()
    outcome = "OK" if v["ok"] else "FAILED: " + v["detail"]
    print(f"probe (torch + faiss + lightgbm in one process): {outcome}; libomp images loaded: {len(v['images'])}")
    return 0 if v["ok"] else 1
