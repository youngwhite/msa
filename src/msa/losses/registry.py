"""Loss registry: name -> class, mirroring `msa.registry` for models.

Contrastive objectives here are *auxiliary* terms, never replacements for the
task loss. That distinction is the whole reason this package exists as a
separate thing from `compute_loss`: an InfoNCE value and an L1 value are not
comparable quantities and cannot be screened against each other. What a
candidate is judged on is what it does to validation MAE/Corr (regression) or
accuracy/F1 (classification) when added to the task loss — see
docs/roadmap.md, phase 2.
"""

from __future__ import annotations

from collections.abc import Callable

_LOSSES: dict[str, type] = {}


def register_loss(name: str) -> Callable[[type], type]:
    key = name.lower()

    def decorator(cls: type) -> type:
        from .contrastive import ContrastiveLoss  # deferred: it imports this module

        if key in _LOSSES and _LOSSES[key] is not cls:
            raise KeyError(f"loss {key!r} is already registered to {_LOSSES[key].__name__}")
        if not issubclass(cls, ContrastiveLoss):
            raise TypeError(f"{cls.__name__} must subclass ContrastiveLoss to be registered")
        cls.name = key
        _LOSSES[key] = cls
        return cls

    return decorator


def _ensure_losses_imported() -> None:
    import msa.losses  # noqa: F401


def available_losses() -> list[str]:
    _ensure_losses_imported()
    return sorted(_LOSSES)


def get_loss_class(name: str) -> type:
    _ensure_losses_imported()
    key = name.lower()
    if key not in _LOSSES:
        raise KeyError(f"unknown loss {name!r}; available: {sorted(_LOSSES)}")
    return _LOSSES[key]


def build_loss(name: str, **kwargs):
    return get_loss_class(name)(**kwargs)
