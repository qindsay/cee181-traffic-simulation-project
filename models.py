from dataclasses import dataclass
import math
from typing import Optional
import numpy as np


@dataclass
class StatParam:
    """Defines a parameter distribution with mean, std, and hard physical bounds."""
    mean: float
    std: float
    min_val: float
    max_val: float

    def sample(self, rng: np.random.Generator, z_override: Optional[float] = None) -> float:
        z = z_override if z_override is not None else rng.standard_normal()
        val = self.mean + z * self.std
        return float(np.clip(val, self.min_val, self.max_val))

    @staticmethod
    def z_to_percentile(z: float) -> float:
        """Exact standard-normal CDF percentile (0-100) from a Z-score."""
        return 50.0 * (1.0 + math.erf(z / math.sqrt(2.0)))


