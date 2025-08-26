"""Cache factory for creating cache instances."""

from .cache import AdaptiveCache


def adaptive_cache(max_size: int = 1000):
    """Create an adaptive cache instance."""
    return AdaptiveCache(max_size=max_size)


# Backward compatibility alias
DistributedCache = AdaptiveCache