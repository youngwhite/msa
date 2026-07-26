"""Model registry: name -> class, so the CLI and trainer never import models directly."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # importing models here at runtime would be circular:
    from .config import DatasetSpec  # models import this module to register themselves.
    from .models.base import MSAModel

_MODELS: dict[str, type] = {}


def register_model(name: str) -> Callable[[type], type]:
    key = name.lower()

    def decorator(cls: type) -> type:
        from .models.base import MSAModel  # deferred: see the note above

        if key in _MODELS and _MODELS[key] is not cls:
            raise KeyError(f"model {key!r} is already registered to {_MODELS[key].__name__}")
        if not issubclass(cls, MSAModel):
            raise TypeError(f"{cls.__name__} must subclass MSAModel to be registered")
        cls.name = key
        _MODELS[key] = cls
        return cls

    return decorator


def _ensure_models_imported() -> None:
    """Registration happens as a side effect of importing the models package."""
    import msa.models  # noqa: F401


def available_models() -> list[str]:
    _ensure_models_imported()
    return sorted(_MODELS)


def get_model_class(name: str) -> type[MSAModel]:
    _ensure_models_imported()
    key = name.lower()
    if key not in _MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_MODELS)}")
    return _MODELS[key]


def build_model(name: str, spec: DatasetSpec, **kwargs) -> MSAModel:
    return get_model_class(name).build(spec, **kwargs)
