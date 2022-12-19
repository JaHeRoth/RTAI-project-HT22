from typing import Tuple, List, Union, Dict, Optional

from torch import Tensor

Bound = Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]
Bounds = List[Union[Bound, Dict[str, List[Bound]]]]
Alpha = Union[str, Dict[Union[str, int], Tensor]]

DEBUG = True
