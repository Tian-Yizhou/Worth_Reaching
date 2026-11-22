"""Score proposed latent prefixes under a frozen completion model's samples."""

import numpy as np


def prefix_energy(
    proposals: np.ndarray, reference: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Return relative negative log smoothed density per latent coordinate.

    Both arrays contain only the queried positions that will be attempted. The
    reference draws are independent of the proposals. Gaussian smoothing keeps
    finite-sample estimates finite and preserves multiple modes in the bank.
    The Gaussian normalization is constant across proposals and is omitted.
    """
    if (
        proposals.ndim != 3
        or reference.ndim != 3
        or not len(proposals)
        or len(reference) < 2
        or proposals.shape[1:] != reference.shape[1:]
        or not all(proposals.shape[1:])
        or scale.shape != (proposals.shape[-1],)
        or not (np.isfinite(scale) & (scale > 0)).all()
        or not np.isfinite(proposals).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Expected finite latent prefixes and a positive latent scale")

    dimensions = proposals.shape[1] * proposals.shape[2]
    bandwidth = len(reference) ** (-1.0 / (dimensions + 4))
    delta = (proposals[:, None].astype(np.float64) - reference[None]) / scale
    log_kernel = -np.square(delta).sum(axis=(2, 3)) / (2 * bandwidth**2)
    peak = log_kernel.max(axis=1, keepdims=True)
    log_mean = peak[:, 0] + np.log(np.exp(log_kernel - peak).mean(axis=1))
    return -log_mean / dimensions
