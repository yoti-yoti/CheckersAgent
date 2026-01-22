# networks/registry.py

from typing import Type, Dict
import torch.nn as nn

_NETWORK_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_network(name: str):
    """
    Decorator to register a network class under a string name.
    """
    def decorator(cls: Type[nn.Module]):
        if name in _NETWORK_REGISTRY:
            raise ValueError(f"Network '{name}' already registered")
        _NETWORK_REGISTRY[name] = cls
        return cls
    return decorator


def get_network_class(name: str) -> Type[nn.Module]:
    if name not in _NETWORK_REGISTRY:
        raise KeyError(
            f"Unknown network '{name}'. "
            f"Available: {list(_NETWORK_REGISTRY.keys())}"
        )
    return _NETWORK_REGISTRY[name]


def list_networks():
    return list(_NETWORK_REGISTRY.keys())
