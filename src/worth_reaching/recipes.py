"""Resolve a completion model recipe from task defaults, TOML and CLI flags."""

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
