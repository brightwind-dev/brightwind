from .load.load import *
from .load.station import MeasurementStation
from .analyse import correlation as Correl
from .analyse.shear import *
from .analyse.analyse import *
from .analyse.plot import *
from .transform.transform import *
from .transform.scale import *
from .export.export import *
from . import demo_datasets
from .utils.gis import *
# from .utils.utils import *

import sys
import warnings
from packaging.version import Version
import pandas as pd

if sys.version_info < (3, 11):
    warnings.warn(
        "Support for Python 3.10 is deprecated and will be removed in v3.0.0. "
        "Please upgrade to Python 3.11 or newer.",
        FutureWarning,
        stacklevel=2,
    )

if Version(pd.__version__) <= Version("2.2"):
    warnings.warn(
        "Support for pandas ≤ 2.2 is deprecated and will be removed in v3.0.0. "
        "Please upgrade to pandas 2.3 or newer.",
        FutureWarning,
        stacklevel=2,
    )

__all__ = ['analyse', 'transform', 'export', 'load', 'demo_datasets']

__version__ = '2.6.0-dev'
