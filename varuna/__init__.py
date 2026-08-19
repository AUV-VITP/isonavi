"""VARUNA: autonomous underwater reconnaissance and assessment for disaster response.

Core simulation and autonomy library. The modules are deliberately free of any
ROS dependency so they can be unit tested and benchmarked directly; the ROS 2
packages under ros2_ws/ are thin wrappers around this code.
"""

__version__ = "0.1.0"

from . import geometry, acoustics  # noqa: F401
