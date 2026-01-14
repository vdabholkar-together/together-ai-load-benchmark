"""
Helpers placeholder for concurrency page.

Currently, logic lives in app.py; this module provides a no-op registration
hook so app.py can import and call it without changing behavior. We can move
the concurrency-specific functions here later.
"""

def register_concurrency_utils(app, socketio):
    """No-op registration placeholder for future refactors."""
    return None

