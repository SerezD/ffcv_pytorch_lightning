from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State
from dataclasses import replace

from typing import Tuple, Optional, Callable

import torch


class DivideImage255(Operation):

    def __init__(self):
        super().__init__()

    def generate_code(self) -> Callable:
        def divide(image, dst):
            dst = image.to(torch.float32) / 255.
            return dst

        divide.is_parallel = True

        return divide

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:

        return replace(previous_state, dtype=torch.float32), None
