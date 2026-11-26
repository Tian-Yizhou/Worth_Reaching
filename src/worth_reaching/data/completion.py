"""Public inputs and hidden training targets for masked trajectory windows."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CompletionInput:
    observations: NDArray[np.float32]
    actions: NDArray[np.int32]
    goal: str
    remaining_actions: int
    query_steps: NDArray[np.int32]
    terminal_example: NDArray[np.float32] | None = None


@dataclass(frozen=True, slots=True)
class CompletionTarget:
    observations: NDArray[np.float32]
    actions: NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class CompletionExample:
    world_id: str
    episode_id: str
    start_step: int
    inputs: CompletionInput
    target: CompletionTarget
