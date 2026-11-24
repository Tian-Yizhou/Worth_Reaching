"""Sample masked windows and expose their public inputs and training targets."""

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

SCHEMA = "completion_windows_v1"
CompletionSplit = Literal["train", "dev"]


@dataclass(frozen=True, slots=True)
class WindowConfig:
    seed: int = 0
    windows_per_episode: int = 8
    min_span: int = 1
    max_span: int | None = None

    def __post_init__(self) -> None:
        if self.windows_per_episode < 1:
            raise ValueError("Use at least one window per episode")
        if self.min_span < 1 or (
            self.max_span is not None and self.max_span < self.min_span
        ):
            raise ValueError("Use positive mask spans with max-span >= min-span")

    def sample(self, world_id: str, budget: int) -> list[tuple[int, int]]:
        """One natural reset plus distinct later starts, with an interior mask each."""
        if self.min_span >= budget:
            raise ValueError("min-span must leave room for a separate terminal step")
        rng = random.Random(f"{SCHEMA}:{self.seed}:{world_id}")
        latest_start = budget - self.min_span - 1
        starts = [
            0,
            *rng.sample(
                range(1, latest_start + 1),
                min(latest_start, self.windows_per_episode - 1),
            ),
        ]
        windows = []
        for start in starts:
            span_limit = budget - start - 1
            if self.max_span is not None:
                span_limit = min(span_limit, self.max_span)
            windows.append((start, rng.randint(self.min_span, span_limit)))
        return windows


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


@dataclass(frozen=True, slots=True)
class CompletionCorpus:
    directory: Path
    goal: str
    budget: int

    def iter_examples(
        self, split: CompletionSplit, *, with_terminal_example: bool = False
    ) -> Iterator[CompletionExample]:
        """Yield independent model inputs and hidden targets from one split."""
        if split not in ("train", "dev"):
            raise ValueError("Completion windows are available for train and dev")
        if with_terminal_example and split != "train":
            raise ValueError("Recorded terminal examples are available for train only")
        for path in sorted((self.directory / split).glob("*.npz")):
            with np.load(path, allow_pickle=False) as episode:
                observations = episode["observations"]
                actions = episode["actions"]
                windows = episode["windows"].tolist()
                world_id = episode["world_id"].item()
                episode_id = episode["episode_id"].item()
                endpoint = (
                    episode["terminal_example"] if with_terminal_example else None
                )
            for start, span in windows:
                end = start + span
                yield CompletionExample(
                    world_id=world_id,
                    episode_id=episode_id,
                    start_step=start,
                    inputs=CompletionInput(
                        observations=observations[: start + 1].copy(),
                        actions=actions[:start].copy(),
                        goal=self.goal,
                        remaining_actions=self.budget - start,
                        query_steps=np.arange(1, span + 1, dtype=np.int32),
                        terminal_example=None if endpoint is None else endpoint.copy(),
                    ),
                    target=CompletionTarget(
                        observations=observations[start + 1 : end + 1].copy(),
                        actions=actions[start:end].copy(),
                    ),
                )
