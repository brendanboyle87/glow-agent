"""Helpers for deterministic experiment seeds.

TODO: expand this helper if the implementation adopts libraries with separate RNG state.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed Python and any optional libraries that are available."""

    # TODO: include torch/jax seeding only when those dependencies actually exist here.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore[import-not-found]

        np.random.seed(seed)
    except ImportError:
        pass
