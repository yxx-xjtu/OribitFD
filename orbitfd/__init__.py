"""OrbitFD: Finite-Support Prototype Orbits for Open-Set Fault Diagnosis."""
from .bank import OrbitBank, empirical_quantile, fit_bank
from .encoders import SourceClassifier, WaveEncoder
from .metrics import benchmark_metrics, open_set_metrics
from .online import OnlineConfig, OrbitOnline, huber_vector_loss, robust_location
from .stream import evaluate_stream

__all__ = [
    "OrbitBank",
    "fit_bank",
    "empirical_quantile",
    "OnlineConfig",
    "OrbitOnline",
    "huber_vector_loss",
    "robust_location",
    "WaveEncoder",
    "SourceClassifier",
    "open_set_metrics",
    "benchmark_metrics",
    "evaluate_stream",
]
