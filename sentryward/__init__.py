"""Verify a PTX kernel, optionally against a mathematical reference."""
__all__ = ["verify"]

def __getattr__(name):
    if name == "verify":
        from .api import verify
        return verify
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
