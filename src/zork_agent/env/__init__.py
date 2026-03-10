"""Environment adapters for Jericho and offline replay.

TODO: add richer live-vs-stub capability reporting once Jericho branching lands.
"""

from zork_agent.env.jericho_env import JerichoEnv
from zork_agent.env.replay import ReplaySession

__all__ = ["JerichoEnv", "ReplaySession"]

