from typing import Tuple
import torch
from torch.utils.data import Dataset


class RGBDepthClassificationDataset(Dataset):
    """
    Replace this with NYU Depth V2 (or other) loader.
    Must return: rgb (3,H,W), depth (1,H,W), label (int).
    """
    def __init__(self, split: str = "train"):
        super().__init__()
        self.split = split
        self.items = []  # Fill with paths/records

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # TODO: load rgb, depth, label
        raise NotImplementedError("Implement dataset loading here.")