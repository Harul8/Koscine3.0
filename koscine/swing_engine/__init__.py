"""Independent swing-trading engine.

This package intentionally does not import the legacy production, tiered,
expansion, rescue, or overlay stacks. It shares data files with Koscine, but
owns its contract, labels, selection, and evaluation path.
"""

from .contract import SWING_ENGINE_CONTRACT_VERSION, SwingContract, load_universe

__all__ = ["SWING_ENGINE_CONTRACT_VERSION", "SwingContract", "load_universe"]
