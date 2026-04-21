"""
API routes package.
Contains route definitions for all endpoints.
"""

from . import analysis
from . import comparison
from . import health

__all__ = ["analysis", "comparison", "health"]
