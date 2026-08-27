"""Local, GPU-backed Babel L0/L2 drafting service."""

from .app import app, create_app

__all__ = ["app", "create_app"]
