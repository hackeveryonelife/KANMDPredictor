import os
import torch


def get_device(prefer: str = "cuda") -> torch.device:
    if os.environ.get("FORCE_CPU", "0") == "1":
        return torch.device("cpu")

    if prefer == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)} "
              f"(CUDA {torch.version.cuda})")
        return device

    print("[Device] CUDA not available / forced off -> using CPU")
    return torch.device("cpu")


def ensure_sparse_coalesced(x):
    if x.is_sparse:
        x = x.coalesce()
    return x


def to_device(obj, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return obj.to(device)
