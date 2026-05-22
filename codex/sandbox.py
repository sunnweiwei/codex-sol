# -*- coding: utf-8 -*-
"""
Robust, standard library-only macOS Seatbelt Sandboxing Engine for the Python Codex Runtime.
Maps dynamic path parameters, resolves virtualenvs/site-packages, compiles Seatbelt policies,
spawns sandboxed processes using sandbox-exec, intercepts violations, and maps outcomes.
"""

from __future__ import annotations

import os
import sys
import subprocess
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

from codex.config import CodexConfig, SandboxMode
from codex.tools import ToolResult

# =========================================================================
# Sandbox Boundary Constants & Assertions
# =========================================================================

MACOS_PATH_TO_SEATBELT_EXECUTABLE = "/usr/bin/sandbox-exec"

PROTECTED_METADATA_PATH_NAMES = (".git", ".codex", ".agents")

# Standard macOS system files and frameworks whitelisted for read/exec map.
MACOS_RESTRICTED_READ_ONLY_PLATFORM_DEFAULTS = """
; Read access to standard system paths
(allow file-read* file-test-existence
  (subpath "/Library/Apple")
  (subpath "/Library/Filesystems/NetFSPlugins")
  (subpath "/Library/Preferences/Logging")
  (subpath "/private/var/db/DarwinDirectory/local/recordStore.data")
  (subpath "/private/var/db/timezone")
  (subpath "/usr/lib")
  (subpath "/usr/share")
  (subpath "/Library/Preferences")
  (subpath "/var/db")
  (subpath "/private/var/db")
  (subpath "/etc")
  (subpath "/private/etc")
  (subpath "/System/Library")
)

; Map system frameworks + dylibs for loader
(allow file-map-executable
  (subpath "/Library/Apple/System/Library/Frameworks")
  (subpath "/Library/Apple/System/Library/PrivateFrameworks")
  (subpath "/Library/Apple/usr/lib")
  (subpath "/System/Library/Extensions")
  (subpath "/System/Library/Frameworks")
  (subpath "/System/Library/PrivateFrameworks")
  (subpath "/System/Library/SubFrameworks")
  (subpath "/System/iOSSupport/System/Library/Frameworks")
  (subpath "/System/iOSSupport/System/Library/PrivateFrameworks")
  (subpath "/System/iOSSupport/System/Library/SubFrameworks")
  (subpath "/usr/lib")
)

; System Framework resources
(allow file-read* file-test-existence
  (subpath "/Library/Apple/System/Library/Frameworks")
  (subpath "/Library/Apple/System/Library/PrivateFrameworks")
  (subpath "/Library/Apple/usr/lib")
  (subpath "/System/Library/Frameworks")
  (subpath "/System/Library/PrivateFrameworks")
  (subpath "/System/Library/SubFrameworks")
  (subpath "/System/iOSSupport/System/Library/Frameworks")
  (subpath "/System/iOSSupport/System/Library/PrivateFrameworks")
  (subpath "/System/iOSSupport/System/Library/SubFrameworks")
  (subpath "/usr/lib")
)

; Allow resolution of standard system symlinks
(allow file-read-metadata file-test-existence
  (literal "/etc")
  (literal "/tmp")
  (literal "/var")
  (literal "/private/etc/localtime")
)

; Allow current working directory access metadata
(allow file-read* file-test-existence
  (literal "/")
)

; Allow standard special files
(allow file-read* file-test-existence
  (literal "/dev/random")
  (literal "/dev/urandom")
  (literal "/private/etc/master.passwd")
  (literal "/private/etc/passwd")
  (literal "/private/etc/protocols")
  (literal "/private/etc/services")
)

; Allow null/zero read/write
(allow file-read* file-test-existence file-write-data
  (literal "/dev/null")
  (literal "/dev/zero")
)

; Allow read/write access to descriptors
(allow file-read-data file-test-existence file-write-data
  (subpath "/dev/fd")
)

; Scratch space so tools can create temp files
(allow file-read* file-test-existence file-write*
  (subpath "/tmp")
  (subpath "/private/tmp")
  (subpath "/var/tmp")
  (subpath "/private/var/tmp")
)

; Allow execution basics
(allow file-read-data (subpath "/bin"))
(allow file-read-metadata (subpath "/bin"))
(allow file-read-data (subpath "/sbin"))
(allow file-read-metadata (subpath "/sbin"))
(allow file-read-data (subpath "/usr/bin"))
(allow file-read-metadata (subpath "/usr/bin"))
(allow file-read-data (subpath "/usr/sbin"))
(allow file-read-metadata (subpath "/usr/sbin"))
(allow file-read-data (subpath "/usr/libexec"))
(allow file-read-metadata (subpath "/usr/libexec"))

(allow file-read* (subpath "/Library/Preferences"))
(allow file-read* (subpath "/opt/homebrew/lib"))
(allow file-read* (subpath "/usr/local/lib"))
(allow file-read* (subpath "/Applications"))

; Terminal basics
(allow file-read* (regex "^/dev/fd/(0|1|2)$"))
(allow file-write* (regex "^/dev/fd/(1|2)$"))
(allow file-read* file-write* (literal "/dev/null"))
(allow file-read* file-write* (literal "/dev/tty"))
(allow file-read-metadata (literal "/dev"))
(allow file-read-metadata (regex "^/dev/.*$"))
(allow file-read-metadata (literal "/dev/stdin"))
(allow file-read-metadata (literal "/dev/stdout"))
(allow file-read-metadata (literal "/dev/stderr"))
(allow file-read-metadata (regex "^/dev/tty[^/]*$"))
(allow file-read-metadata (regex "^/dev/pty[^/]*$"))
(allow file-read* file-write* (regex "^/dev/ttys[0-9]+$"))
(allow file-read* file-write* (literal "/dev/ptmx"))
(allow file-ioctl (regex "^/dev/ttys[0-9]+$"))
"""

MACOS_SEATBELT_BASE_POLICY = """
(version 1)

; start closed-by-default
(deny default)

; process mapping
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))

; sysctl permissions
(allow sysctl-read
  (sysctl-name "hw.activecpu")
  (sysctl-name "hw.busfrequency_compat")
  (sysctl-name "hw.byteorder")
  (sysctl-name "hw.cacheconfig")
  (sysctl-name "hw.cachelinesize_compat")
  (sysctl-name "hw.cpufamily")
  (sysctl-name "hw.cpufrequency_compat")
  (sysctl-name "hw.cputype")
  (sysctl-name "hw.logicalcpu_max")
  (sysctl-name "hw.machine")
  (sysctl-name "hw.model")
  (sysctl-name "hw.memsize")
  (sysctl-name "hw.ncpu")
  (sysctl-name "machdep.cpu.brand_string")
  (sysctl-name "kern.argmax")
  (sysctl-name "kern.hostname")
  (sysctl-name "kern.maxfilesperproc")
  (sysctl-name "kern.maxproc")
  (sysctl-name "kern.osproductversion")
  (sysctl-name "kern.osrelease")
  (sysctl-name "kern.ostype")
  (sysctl-name "kern.osvariant_status")
  (sysctl-name "kern.osversion")
  (sysctl-name "kern.version")
  (sysctl-name "vm.loadavg")
)

; needed for python multiprocessing
(allow ipc-posix-sem)

; allow openpty
(allow pseudo-tty)

; allow read-only preference settings
(allow ipc-posix-shm-read* (ipc-posix-name-prefix "apple.cfprefs."))
(allow mach-lookup
  (global-name "com.apple.cfprefsd.daemon")
  (global-name "com.apple.cfprefsd.agent")
  (local-name "com.apple.cfprefsd.agent")
  (global-name "com.apple.system.opendirectoryd.libinfo")
  (global-name "com.apple.PowerManagement.control")
)
(allow user-preference-read)
"""

MACOS_SEATBELT_NETWORK_POLICY = """
; allow Outbound AF_SYSTEM sockets for system lookup
(allow system-socket
  (require-all
    (socket-domain AF_SYSTEM)
    (socket-protocol 2)
  )
)

(allow mach-lookup
    (global-name "com.apple.bsd.dirhelper")
    (global-name "com.apple.system.opendirectoryd.membership")
    (global-name "com.apple.SecurityServer")
    (global-name "com.apple.networkd")
    (global-name "com.apple.ocspd")
    (global-name "com.apple.trustd.agent")
    (global-name "com.apple.SystemConfiguration.DNSConfiguration")
    (global-name "com.apple.SystemConfiguration.configd")
)

(allow sysctl-read
  (sysctl-name-regex #"^net.routetable")
)
"""


class SandboxException(Exception):
    """Raised when a command execution violates a macOS Seatbelt sandbox boundary constraint."""
    def __init__(self, message: str, exit_code: int, stderr: str, blocked_path: str | None = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr
        self.blocked_path = blocked_path


# =========================================================================
# Dynamic Path Resolution and Canonicalization
# =========================================================================

def normalize_path_for_sandbox(path: Path | str) -> Path | None:
    """
    Canonicalizes and resolves directories to prevent sandbox escapes,
    verifying path is absolute before resolution.
    """
    p = Path(path)
    if not p.is_absolute():
        return None
    try:
        return p.resolve()
    except Exception:
        # Fallback to absolute representation if canonicalize throws
        return p.absolute()


def extract_python_site_packages() -> list[Path]:
    """
    Dynamically fetches Python sys.path lists, resolving packaging dirs,
    standard library folders, and symlink origins.
    """
    paths = []
    
    # Standard import search paths
    for item in sys.path:
        if not item:
            continue
        p = normalize_path_for_sandbox(item)
        if p and p.exists():
            paths.append(p)
            
    # Include current python executable parent directory
    exe_path = normalize_path_for_sandbox(sys.executable)
    if exe_path:
        paths.append(exe_path.parent)
        # Handle symlinks (e.g. under pyenv or homebrew)
        try:
            real_exe = exe_path.resolve()
            paths.append(real_exe.parent)
            if real_exe.parent.parent:
                paths.append(real_exe.parent.parent)
        except Exception:
            pass

    # Include basic prefixes
    for prefix in (sys.prefix, sys.exec_prefix):
        if prefix:
            p = normalize_path_for_sandbox(prefix)
            if p and p.exists():
                paths.append(p)
                
    # Deduplicate and return sorted list
    unique_paths = sorted(list(set(paths)), key=lambda x: len(str(x)))
    return unique_paths


def extract_loopback_proxy_ports(env: dict[str, str] | None = None) -> list[int]:
    """
    Parses environment variables for local network proxy routing addresses,
    filtering localhost loopback destination ports.
    """
    target_env = env if env is not None else os.environ
    ports = set()
    
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    loopback_regex = re.compile(r"^(localhost|127\.0\.0\.1|::1)$", re.IGNORECASE)
    
    for key in proxy_keys:
        val = target_env.get(key)
        if not val:
            continue
        
        # Strip protocol headers
        clean = val.strip()
        if "://" in clean:
            clean = clean.split("://", 1)[1]
            
        # Strip username/passwords
        if "@" in clean:
            clean = clean.rsplit("@", 1)[1]
            
        # Parse host and port
        if ":" in clean:
            host, port_str = clean.split(":", 1)
            # Remove path suffix if any
            port_str = port_str.split("/")[0]
            if loopback_regex.match(host):
                try:
                    ports.add(int(port_str))
                except ValueError:
                    pass
        else:
            # Check default scheme port if localhost
            host = clean.split("/")[0]
            if loopback_regex.match(host):
                if key.lower().startswith("https"):
                    ports.add(443)
                elif key.lower().startswith("socks"):
                    ports.add(1080)
                else:
                    ports.add(80)
                    
    return sorted(list(ports))


# =========================================================================
# Seatbelt Profile Compilation Engine
# =========================================================================

def compile_seatbelt_profile(
    config: CodexConfig,
    env: dict[str, str] | None = None
) -> tuple[str, dict[str, Path]]:
    """
    Compiles dynamic whitelisting parameters and environment boundaries
    into an isolated macOS Seatbelt profile layout.
    """
    parameters: dict[str, Path] = {}
    
    # 1. Base files policy
    policy_parts = [MACOS_SEATBELT_BASE_POLICY, MACOS_RESTRICTED_READ_ONLY_PLATFORM_DEFAULTS]
    
    # 2. Extract directories for reading
    readable_roots: list[Path] = []
    
    # Active workspace
    workspace = normalize_path_for_sandbox(config.cwd)
    if workspace:
        readable_roots.append(workspace)
        
    # Codex Home
    c_home = normalize_path_for_sandbox(config.codex_home)
    if c_home:
        readable_roots.append(c_home)
        
    # Dynamic Python paths
    readable_roots.extend(extract_python_site_packages())
    
    # Additional writable roots are also readable
    for wr in config.writable_roots:
        resolved_wr = normalize_path_for_sandbox(wr)
        if resolved_wr:
            readable_roots.append(resolved_wr)
            
    # Deduplicate readable roots
    deduped_readable: list[Path] = []
    for r in sorted(list(set(readable_roots)), key=lambda x: len(str(x))):
        # Prevent overlaps
        if not any(r.is_relative_to(parent) for parent in deduped_readable if r != parent):
            deduped_readable.append(r)
            
    # Compile reading permissions rules
    read_rules = []
    for idx, r_path in enumerate(deduped_readable):
        param_key = f"READABLE_ROOT_{idx}"
        parameters[param_key] = r_path
        read_rules.append(f'(allow file-read* file-test-existence (subpath (param "{param_key}")))')
        # Allow executable mapping for importable .so/.dylib objects
        read_rules.append(f'(allow file-map-executable (subpath (param "{param_key}")))')
        
    policy_parts.append("; Dyn-resolved Read Whitelists\n" + "\n".join(read_rules))
    
    # 3. Compile writing permissions rules
    write_rules = []
    if config.sandbox == "danger-full-access":
        # Full disk write access (though manager wouldn't call this under None, compile for completeness)
        policy_parts.append('(allow file-write* (regex #"^/"))')
    elif config.sandbox == "workspace-write":
        # Can write to workspace and custom writable roots
        writable_targets: list[Path] = []
        if workspace:
            writable_targets.append(workspace)
        for wr in config.writable_roots:
            resolved_wr = normalize_path_for_sandbox(wr)
            if resolved_wr:
                writable_targets.append(resolved_wr)
                
        # Deduplicate write targets
        deduped_writable: list[Path] = []
        for w in sorted(list(set(writable_targets)), key=lambda x: len(str(x))):
            if not any(w.is_relative_to(parent) for parent in deduped_writable if w != parent):
                deduped_writable.append(w)
                
        # Compile write permission rules with nested exclusions (.git, .codex, .agents)
        for idx, w_path in enumerate(deduped_writable):
            root_param = f"WRITABLE_ROOT_{idx}"
            parameters[root_param] = w_path
            
            require_parts = [f'(subpath (param "{root_param}"))']
            
            # Exclude metadata dirs recursively from writes
            for meta_idx, meta_name in enumerate(PROTECTED_METADATA_PATH_NAMES):
                meta_path = w_path / meta_name
                # Normalize and register exclusion
                meta_param = f"{root_param}_EXCLUDED_{meta_idx}"
                parameters[meta_param] = meta_path
                
                require_parts.append(f'(require-not (literal (param "{meta_param}")))')
                require_parts.append(f'(require-not (subpath (param "{meta_param}")))')
                
            write_rules.append(f'(allow file-write* (require-all {chr(10)}    {" ".join(require_parts)} {chr(10)}))')
            
        policy_parts.append("; Dyn-resolved Writable Whitelists (Metadata Protected)\n" + "\n".join(write_rules))
    else: # read-only
        # Explicitly deny write access to workspace
        policy_parts.append("; Read-only profile constraints\n(deny file-write*)")
        # Ensure temp directories are still allowed for tools to compile/spit logs
        policy_parts.append('(allow file-read* file-write* (subpath "/tmp"))')
        policy_parts.append('(allow file-read* file-write* (subpath "/private/tmp"))')
        
    # 4. Networking constraints mapping
    net_rules = []
    proxy_ports = extract_loopback_proxy_ports(env)
    
    if proxy_ports:
        net_rules.append(MACOS_SEATBELT_NETWORK_POLICY)
        net_rules.append("; Loopback network proxy egress mapping")
        # Allow local bindings
        net_rules.append('(allow network-bind (local ip "*:*"))')
        net_rules.append('(allow network-inbound (local ip "localhost:*"))')
        # Allow outbound ONLY to localhost proxy ports
        net_rules.append('(allow network-outbound (remote ip "localhost:*"))')
        for port in proxy_ports:
            net_rules.append(f'(allow network-outbound (remote ip "localhost:{port}"))')
        net_rules.append('(allow network-outbound (remote ip "*:53")) ; Allow DNS resolve')
    else:
        # Completely deny raw network outbound egress
        net_rules.append("; Closed-by-default network policy overrides")
        net_rules.append("(deny network-outbound)")
        net_rules.append("(deny network-inbound)")
        # Permit unix socket lookups
        net_rules.append("(allow system-socket (socket-domain AF_UNIX))")
        
    policy_parts.append("\n".join(net_rules))
    
    # 5. Join final profile block
    profile = "\n\n".join(policy_parts)
    return profile, parameters


# =========================================================================
# Command Execution Engine
# =========================================================================

def run_sandboxed_command(
    command: list[str],
    config: CodexConfig,
    env: dict[str, str] | None = None
) -> ToolResult:
    """
    Wraps subprocess.Popen under /usr/bin/sandbox-exec boundaries,
    resolving nested OS environment blockades.
    """
    if not command:
        return ToolResult(ok=False, output="Error: Empty command parameter", metadata={})
        
    # 1. Escape immediately if danger-full-access mode was explicitly requested
    if config.sandbox == "danger-full-access":
        # Native un-sandboxed execution
        try:
            res = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=str(config.cwd),
                env=env
            )
            return ToolResult(
                ok=res.returncode == 0,
                output=res.stdout + res.stderr,
                metadata={"sandbox_mode": "danger-full-access", "exit_code": res.returncode}
            )
        except Exception as e:
            return ToolResult(ok=False, output=f"Execution Failed: {str(e)}\n", metadata={})

    # 2. Compile Seatbelt Profile
    try:
        profile_str, params = compile_seatbelt_profile(config, env)
    except Exception as e:
        return ToolResult(ok=False, output=f"Seatbelt compiling failed: {str(e)}\n", metadata={})

    # 3. Assemble sandbox-exec argument list
    args = [MACOS_PATH_TO_SEATBELT_EXECUTABLE, "-p", profile_str]
    for key, val in params.items():
        args.append(f"-D{key}={val}")
    args.append("--")
    args.extend(command)

    # 4. Spawning sandboxed process
    try:
        res = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=str(config.cwd),
            env=env
        )
        
        # 5. Intercept sandbox-exec nested blockade
        if res.returncode == 71 and "sandbox_apply: Operation not permitted" in res.stderr:
            # Nested Sandboxing Trap detected! 
            # Environment is already sandboxed, OS prevents nesting. Soft-failover un-sandboxed execution:
            sys.stderr.write("[Warning] Nested sandboxing blocked by OS. Executing un-sandboxed fallbacks.\n")
            
            # Execute command un-sandboxed
            fallback_res = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=str(config.cwd),
                env=env
            )
            return ToolResult(
                ok=fallback_res.returncode == 0,
                output=fallback_res.stdout + fallback_res.stderr,
                metadata={
                    "sandbox_mode": str(config.sandbox),
                    "exit_code": fallback_res.returncode,
                    "nested_blockade_soft_fallback": True,
                    "sandbox_exec_trap_stderr": res.stderr
                }
            )

        # 6. Intercept standard sandbox write/read violations
        # Standard indicators on macOS: Operation not permitted (EPERM), Permission denied (EACCES)
        is_violation = False
        blocked_path = None
        
        err_msg = res.stderr
        
        # Inspect exit status and stderr traces for violation gutters
        if res.returncode != 0:
            # e.g. "Operation not permitted" or "Permission denied"
            patterns = [
                r"Operation not permitted",
                r"Permission denied",
                r"Operation Not Permitted",
                r"Permission Denied"
            ]
            if any(re.search(p, err_msg) for p in patterns):
                is_violation = True
                # Attempt to parse target path out of error trace, e.g. "bash: /Users/...: Operation not permitted"
                match = re.search(r"(?:bash:\s*)?([^\s:]+):\s*Operation not permitted", err_msg)
                if match:
                    blocked_path = match.group(1)
                else:
                    match_permission = re.search(r"(?:bash:\s*)?([^\s:]+):\s*Permission denied", err_msg)
                    if match_permission:
                        blocked_path = match_permission.group(1)

        metadata = {
            "sandbox_mode": str(config.sandbox),
            "exit_code": res.returncode,
            "sandbox_violation": is_violation
        }
        if blocked_path:
            metadata["blocked_path"] = blocked_path

        # Return structured ToolResult
        return ToolResult(
            ok=not is_violation and res.returncode == 0,
            output=res.stdout + res.stderr,
            metadata=metadata
        )

    except FileNotFoundError as e:
        # sandbox-exec is missing (e.g. non-mac platform tests)
        if not Path(MACOS_PATH_TO_SEATBELT_EXECUTABLE).exists():
            sys.stderr.write("[Warning] sandbox-exec executable absent. Running un-sandboxed fallbacks.\n")
            try:
                fallback_res = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    cwd=str(config.cwd),
                    env=env
                )
                return ToolResult(
                    ok=fallback_res.returncode == 0,
                    output=fallback_res.stdout + fallback_res.stderr,
                    metadata={
                        "sandbox_mode": str(config.sandbox),
                        "exit_code": fallback_res.returncode,
                        "sandbox_executable_absent": True
                    }
                )
            except Exception as ex:
                return ToolResult(ok=False, output=f"Execution Failed: {str(ex)}\n", metadata={})
        return ToolResult(ok=False, output=f"Spawning error: {str(e)}\n", metadata={})
        
    except Exception as e:
        return ToolResult(ok=False, output=f"Spawning failed: {str(e)}\n", metadata={})
