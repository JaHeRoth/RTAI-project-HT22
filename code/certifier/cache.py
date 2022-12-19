from typing import Tuple, Dict

from torch import Tensor
from torch.nn import Conv2d, BatchNorm2d


class ConvToAffineCache:
    _cache: Dict[str, Tuple[Tensor, Tensor]] = {}

    @staticmethod
    def _to_key(conv_layer: Conv2d, in_height: int, in_width: int, bn_layer: BatchNorm2d):
        # TODO: Unit test this on all networks for hash collitions between different layers
        return f"{hash(conv_layer)}_{in_height}_{in_width}_{hash(bn_layer)}"

    def set(self, conv_layer: Conv2d, in_height: int, in_width: int, bn_layer: BatchNorm2d, value: Tuple[Tensor, Tensor]):
        key = self._to_key(conv_layer, in_height, in_width, bn_layer)
        self._cache[key] = value

    def get(self, conv_layer: Conv2d, in_height: int, in_width: int, bn_layer: BatchNorm2d):
        key = self._to_key(conv_layer, in_height, in_width, bn_layer)
        return self._cache[key] if key in self._cache.keys() else None
