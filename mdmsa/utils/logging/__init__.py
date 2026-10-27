import os
from logging import Logger, getLogger
from typing import Optional

try:
    from torch.distributed.elastic.utils.logging import _derive_module_name
except ImportError:

    def _derive_module_name(*args, **kwargs) -> str:
        return "Logger"


__all__ = ["get_logger"]


def get_logger(name: Optional[str] = None) -> Logger:
    """Get logger by name.

    Args:
        name (str, optional): Name of the logger. If no name provided, the name will
              be derived from the call stack.

    Returns:
        Logger: Logger to record process.

    """
    if name is None:
        name = _derive_module_name(depth=2)

    logger = _setup_logger(name)

    return logger


def _setup_logger(name: Optional[str] = None):
    logger = getLogger(name)
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    return logger


class DummyLogger(Logger):
    def __init__(self, name: str, level: int = 0) -> None:
        super().__init__(name, level=level)

    def info(self, *args, **kwargs) -> None:
        pass
