from dataclasses import dataclass
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
        """Approximates normal CDF percentile (0-100) from Z-score."""
        return float(0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3))) * 100.0)


