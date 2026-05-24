import os
import shutil
import stat

_original_rmtree = shutil.rmtree

def force_delete(target_path):
    if os.path.isdir(target_path):
        for root_dir, dirs, files in os.walk(target_path, topdown=False):
            for f in files:
                fp = os.path.join(root_dir, f)
                try:
                    os.chmod(fp, stat.S_IWRITE)
                    os.unlink(fp)
                except Exception:
                    pass
            for d in dirs:
                dp = os.path.join(root_dir, d)
                try:
                    os.chmod(dp, stat.S_IWRITE)
                    os.rmdir(dp)
                except Exception:
                    pass
        try:
            os.chmod(target_path, stat.S_IWRITE)
            os.rmdir(target_path)
        except Exception:
            pass
    else:
        try:
            os.chmod(target_path, stat.S_IWRITE)
            os.unlink(target_path)
        except Exception:
            pass

def _robust_rmtree(path, *args, **kwargs):
    # For Python 3.12+ (which uses onexc instead of onerror)
    if "onexc" in kwargs:
        orig = kwargs["onexc"]
        def wrapped_onexc(func, ep, err):
            try:
                force_delete(ep)
            except Exception:
                if orig:
                    orig(func, ep, err)
        kwargs["onexc"] = wrapped_onexc
    else:
        orig = kwargs.get("onerror")
        def wrapped_onerror(func, ep, ei):
            try:
                force_delete(ep)
            except Exception:
                if orig:
                    orig(func, ep, ei)
        kwargs["onerror"] = wrapped_onerror
        
    return _original_rmtree(path, *args, **kwargs)

shutil.rmtree = _robust_rmtree

from codex.types import CodexConfig
from codex.core import CodexSession

import sys
import inspect
stack_str = "".join(frame[1] for frame in inspect.stack() if frame[1] is not None)
is_running_cli = "codex/cli.py" in stack_str or "codex.cli" in stack_str or (sys.argv and sys.argv[0] == "-m")

cli = None
if not is_running_cli:
    import codex.cli as cli

__all__ = ["CodexConfig", "CodexSession", "cli"]
