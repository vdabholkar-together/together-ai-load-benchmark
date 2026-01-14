"""
Utility package for dashboard feature groupings.

These helpers keep page-specific utilities grouped without changing runtime
behavior. Each module exposes a register_* function that can be called from
app.py to keep imports explicit while we gradually refactor.
"""

