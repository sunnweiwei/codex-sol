"""Runtime limits, timeouts, wait constraints, and configuration boundaries."""

from __future__ import annotations

# Minimum wait timeout (in milliseconds) allowed for agent wait tools to prevent tight loops burning CPU.
# Maps exactly to upstream multiagent MIN_WAIT_TIMEOUT_MS / DEFAULT_MULTI_AGENT_V2_MIN_WAIT_TIMEOUT_MS.
_MIN_AGENT_WAIT_TIMEOUT_MS: int = 10000

# Maximum wait timeout (in milliseconds) allowed for agent wait tools (1 hour).
# Maps exactly to upstream multiagent MAX_WAIT_TIMEOUT_MS / MAX_MULTI_AGENT_V2_WAIT_TIMEOUT_MS.
_MAX_AGENT_WAIT_TIMEOUT_MS: int = 3600000
