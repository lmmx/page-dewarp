"""Import the `snoop` tracer decorator or define a no-op if unavailable.

This file attempts to import `pysnooper.snoop`. If it's not installed,
we fall back to a dummy function that does nothing, so debug instrumentation
won't break in environments lacking pysnooper.
"""

__all__ = ("snoop",)

try:
    from pysnooper import snoop
except ImportError:
    # Development dependencies not installed: don't debug, use a no-op decorator
    def snoop(*args, **kwargs):
        """Return a decorator that leaves the function unchanged.

        This no-op is used if pysnooper isn't installed.
        """

        def decorator(func):
            return func

        return decorator
