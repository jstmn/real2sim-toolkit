"""Shim: `r2st.segment_anything` re-exports the inner SAM package."""

from .segment_anything import *
from .segment_anything import SamPredictor, sam_model_registry  # noqa: F401
