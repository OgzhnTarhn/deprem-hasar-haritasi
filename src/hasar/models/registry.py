"""Name-based registry for model factories.

Models are registered under a short name (``"changemamba"``, ``"damageformer"``, ...) and built
from a config by name, so the training code never imports a concrete model class directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_FACTORIES: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that registers ``factory`` under ``name``.

    The factory is returned unchanged. Registering the same name twice raises
    :class:`ValueError`.
    """

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        if name in _FACTORIES:
            raise ValueError(f"'{name}' adı zaten kayıtlı.")
        _FACTORIES[name] = factory
        return factory

    return decorator


def build(name: str, **kwargs: Any) -> Any:
    """Build the model registered under ``name``, passing ``kwargs`` to its factory."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        known = ", ".join(available()) or "(kayıt yok)"
        raise KeyError(f"'{name}' adlı model kayıtlı değil. Kayıtlı adlar: {known}") from None
    return factory(**kwargs)


def available() -> list[str]:
    """Return the registered model names, sorted."""
    return sorted(_FACTORIES)
