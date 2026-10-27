import torch


def convert_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if dtype is None:
        dtype = torch.float32
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "float32":
        dtype = torch.float32
    elif dtype == "float64":
        dtype = torch.float64

    return dtype
