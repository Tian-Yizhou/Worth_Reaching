"""Resolve model recipes from task defaults, TOML and CLI flags."""

import argparse
import tomllib
from dataclasses import fields, replace
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from worth_reaching.models.completion import TrainingConfig
    from worth_reaching.models.continuous_completion import CompletionConfig

Recipe = TypeVar("Recipe", bound="TrainingConfig | CompletionConfig")


def completion_recipe(defaults: Recipe, args: argparse.Namespace) -> Recipe:
    """Apply explicit CLI settings last. The runner owns horizon and input schema."""
    names = {field.name for field in fields(defaults)} - {
        "max_horizon",
        "max_action_budget",
        "predictor_stopping",
    }
    stopping = defaults.predictor_stopping
    values = {name: getattr(defaults, name) for name in names}
    values.update(
        predictor_val_every=stopping.val_every, predictor_patience=stopping.patience
    )
    if args.config is not None:
        with args.config.open("rb") as stream:
            recipe = tomllib.load(stream)
        if set(recipe) != {"model"} or not isinstance(recipe["model"], dict):
            raise ValueError("Expected a [model] table in the completion recipe")
        for name, value in recipe["model"].items():
            if name not in values:
                raise ValueError(f"Unknown model setting: {name}")
            expected = type(values[name])
            if expected is float and type(value) is int:
                value = float(value)
            if type(value) is not expected:
                raise ValueError(f"model.{name} requires {expected.__name__}")
            values[name] = value
    values.update(
        {
            name: vars(args)[name]
            for name in values
            if getattr(args, name, None) is not None
        }
    )
    stopping = replace(
        stopping,
        val_every=values.pop("predictor_val_every"),
        patience=values.pop("predictor_patience"),
    )
    return replace(defaults, **values, predictor_stopping=stopping)


def add_option_arguments(parser: argparse.ArgumentParser) -> None:
    """Options for the value, the online rounds and the planner."""
    parser.add_argument("--value-updates", type=int, default=200_000)
    parser.add_argument("--value-options", type=int, default=1_000_000)
    parser.add_argument("--value-heads", type=int, default=4)
    parser.add_argument(
        "--time-costs",
        type=float,
        nargs="+",
        default=[0.0, 0.0005, 0.001, 0.002],
        help="Per-action costs the value is trained on",
    )
    parser.add_argument(
        "--eval-costs",
        type=float,
        nargs="+",
        default=[0.0, 0.001],
        help="Time costs evaluated; ablations use the last",
    )
    parser.add_argument(
        "--online-rounds",
        type=int,
        default=3,
        help="Rounds of executor attempts on training resets that correct the value",
    )
    parser.add_argument("--online-episodes", type=int, default=256)
    parser.add_argument("--online-updates", type=int, default=50_000)
    parser.add_argument(
        "--explore",
        type=float,
        default=0.2,
        help="Probability of pursuing a random candidate at a decision while collecting",
    )
    parser.add_argument(
        "--goal-attempts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also pursue the terminal goal uninterrupted on every training reset",
    )
    parser.add_argument(
        "--fast-budgets",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25],
        help="Budget fractions under which the fast arms also propose routes",
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--risk",
        type=float,
        default=0.0,
        help="Rank candidates by mean + risk x spread across value heads",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Value change, in spreads, that triggers a switch or new candidates",
    )
