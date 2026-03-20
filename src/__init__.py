"""PhaseForensics: Training-Free Synthetic Image Detection via Local Phase Coherence Violation."""

import numpy as np

# Compatibility shims: dtcwt uses functions removed in NumPy 2.0
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
if not hasattr(np, "issubsctype"):
    np.issubsctype = np.issubdtype

__version__ = "0.1.0"

from .detector import PhaseForensicsDetector

__all__ = ["PhaseForensicsDetector", "__version__"]
