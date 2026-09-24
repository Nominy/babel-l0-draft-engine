"""Local, GPU-backed Babel L0/L2 drafting service."""

__all__ = ["app", "create_app"]


def __getattr__(name: str):
    if name in __all__:
        from importlib import import_module

        engine = import_module(".app", __name__)
        globals().update(app=engine.app, create_app=engine.create_app)
        return getattr(engine, name)
    raise AttributeError(name)
