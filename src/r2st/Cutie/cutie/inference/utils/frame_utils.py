
import torch

from inference.object_info import ObjectInfo


class FrameInfo:
    def __init__(self, image: torch.Tensor, mask: torch.Tensor, segments_info: list[ObjectInfo], ti: int, info: dict):
        self.image = image
        self.mask = mask
        self.segments_info = segments_info
        self.ti = ti
        self.info = info

    @property
    def name(self) -> str:
        return self.info["frame"]

    @property
    def shape(self) -> tuple(int):
        return self.info["shape"]

    @property
    def need_save(self) -> bool:
        return self.info["save"]
