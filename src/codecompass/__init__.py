__version__ = "0.2.2"

import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*",
    category=UserWarning,
)