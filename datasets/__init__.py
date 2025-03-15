#helps importing and loading datasets

from .nvidia import load_nvidia_data
from .maxtemperatures import load_maxtemperature_data
from .rainfall import load_rainfall_data

__all__ = [
    "load_nvidia_data",
    "load_maxtemperature_data",
    "load_rainfall_data"
    ]