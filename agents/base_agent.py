# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

class BaseAgent(ABC):
    @abstractmethod
    def act(self, obs) -> tuple[Any, Tensor, Any]:
        pass

    @abstractmethod
    def update(self, transition):
        pass

    @abstractmethod
    def finish_rollout(self):
        pass

    @abstractmethod
    def learn_from_rollout(self, rollout):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
