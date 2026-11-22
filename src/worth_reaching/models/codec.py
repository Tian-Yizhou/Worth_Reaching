"""Observation encoder and decoder shared by the completion models."""

from torch import nn


def linear(input_size: int, output_size: int) -> nn.Linear:
    layer = nn.Linear(input_size, output_size)
    # Match the variance of the Flax truncated-normal dense initializer.
    std = input_size**-0.5 / 0.8796256610342398
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(input_size, width, output_size, *, layers, activation=None):
    modules = []
    for _ in range(layers):
        modules.extend(
            (
                linear(input_size, width),
                activation() if activation else nn.GELU(approximate="tanh"),
            )
        )
        input_size = width
    modules.append(linear(input_size, output_size))
    return nn.Sequential(*modules)


class ObservationCodec(nn.Module):
    def __init__(self, width: int, latent_size: int, observation_size: int):
        super().__init__()
        self.encoder = mlp(observation_size, width, latent_size, layers=1)
        self.decoder = mlp(latent_size, width, observation_size, layers=1)

    def encode(self, observations):
        return self.encoder(observations)

    def decode(self, latents):
        return self.decoder(latents)

    def forward(self, observations):
        return self.decode(self.encode(observations))


def reconstruction(model, batch):
    (observations,) = batch
    return ((model(observations) - observations, None),)


def legacy_dense(params: dict, prefix: str, *layer_names: str) -> dict:
    values = {}
    for index, name in enumerate(layer_names):
        layer = params[name]
        values[f"{prefix}.{2 * index}.weight"] = layer["kernel"].T
        values[f"{prefix}.{2 * index}.bias"] = layer["bias"]
    return values
