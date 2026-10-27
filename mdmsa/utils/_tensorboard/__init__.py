from typing import Optional

from torch.utils.tensorboard import SummaryWriter

__all__ = [
    "get_writer",
]


def get_writer(log_dir: Optional[str] = None) -> SummaryWriter:
    """Get TensorBoard writer.

    Args:
        log_dir (str, optional): Directory to save logs.

    Returns:
        SummaryWriter: Writer to record training.

    """
    return SummaryWriter(log_dir)


class DummySummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __getattribute__(self, name: str) -> None:
        if hasattr(super(), name):

            def _no_ops(*args, **kwargs) -> None:
                pass

            return _no_ops
        else:
            raise AttributeError(f"SummaryWriter has no attribute {name}.")
