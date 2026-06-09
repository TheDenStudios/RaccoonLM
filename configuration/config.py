"""
Configuration — backward-compatible re-export.

All canonical config lives in raccoonlm/config.py (root level).
This file exists so imports from configuration/* submodules still work
if they accidentally reference the old path.
"""
from raccoonlm.config import settings, get_default_model  # noqa: F401
