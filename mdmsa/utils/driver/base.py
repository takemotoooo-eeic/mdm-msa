import subprocess
import warnings
from abc import ABC, abstractmethod

from omegaconf import DictConfig


class BaseDriver(ABC):
    """Base class of drivers (trainer and evaluator)."""

    def set_commit_hash(self) -> None:
        try:
            completed_process = subprocess.run("git rev-parse HEAD", shell=True, check=True, capture_output=True)
            commit_hash = completed_process.stdout.decode().strip()
        except subprocess.CalledProcessError:
            commit_hash = None
            warnings.warn("The system is not managed by git.", UserWarning)

        self.commit_hash = commit_hash

    @classmethod
    @abstractmethod
    def build_from_config(cls, config: DictConfig) -> "BaseDriver":
        pass


class Driver(BaseDriver):
    pass
