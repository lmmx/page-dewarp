"""Import the `snoop` tracer decorator or make a no-op wrapper if unavailable."""

__all__ = ("snoop",)

try:
    from pysnooper import snoop
except ImportError:
    # Development dependencies not installed: don't debug, use no-op decorator
    def snoop(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
