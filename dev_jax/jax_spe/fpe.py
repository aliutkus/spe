import functools
import math
from typing import Tuple, Union

from flax import nn
import jax
import jax.numpy as jnp
import numpy as np


class SineFPE(nn.Module):
    """Sinusoidal stochastic positional encoding.

    Args:
        rng_key: A PRNGKey.
        key_shape: The shape of keys and queries.
        num_realizations: The number of realizations of the stochastic
            process (R).
        num_sines: The number of sin and cos components (K).
    """

    def apply(
        self,
        rng_key,
        key_shape,
        num_sines: int = 10,
    ):
        length = key_shape[1]
        in_features = key_shape[-1]
        num_heads = key_shape[-2]
        params_shape = (num_heads, in_features, num_sines)
        # bias initial frequencies to low values for long term range
        freqs = self.param(
            'freqs', params_shape,
            lambda *args: jax.random.normal(*args) - 4.)
        offsets = self.param('offsets', params_shape, jax.random.normal)
        def init_gains(rng_key, shape):
            gains = jax.random.normal(rng_key, shape)
            return gains / (jnp.sqrt(jnp.linalg.norm(gains, axis=-1, keepdims=True)) / 2)
        gains = self.param('gains', params_shape, init_gains)

        # build omega_q and omega_k,
        # with shape (num_heads, keys_dim, length, 2*num_sines)
        indices = jnp.linspace(0, length-1, length)

        # making sure the frequencies are in [0, 0.5]
        freqs = jax.nn.sigmoid(freqs[None, :, :, :]) / 2.

        phases_q = (
            2 * math.pi
            * freqs * indices[:, None, None, None]
            + offsets[None, :, :, :]
        )
        omega_q = jnp.stack([jnp.cos(phases_q), jnp.sin(phases_q)], axis=-1).reshape(
            length, num_heads, in_features, 2*num_sines
        )

        phases_k = (
            2 * math.pi
            * freqs * indices[:, None, None, None]
        )
        omega_k = jnp.stack([jnp.cos(phases_k), jnp.sin(phases_k)], axis=-1).reshape(
            length, num_heads, in_features, 2*num_sines
        )

        # gains is (num_heads, keys_dim, num_sines). Making them nonnegative with softplus
        gains = jax.nn.softplus(gains)

        # # now upsample it to (num_heads, keys_dim, 2*num_sines)
        gains = jnp.stack(
            [gains, gains], axis=-1
        ).reshape(num_heads, in_features, 2 * num_sines)

        # scale each of the 2*num_sines by the appropriate gain
        qbar = omega_q * gains[None, ...]
        kbar = omega_k * gains[None, ...]

        scale = jnp.sqrt(jnp.sqrt(jnp.reciprocal(2 * num_sines * in_features)))
        return scale * qbar, scale * kbar


class FPEGate(nn.Module):

    def apply(self, rng_key, fpe_code):
        qbar, kbar = fpe_code

        gate = self.param(
            'gate',
            kbar.shape[-3:-1],
            jax.random.normal)

        # incorporate the constant bias for Pd if required. First draw noise
        # such that noise noise^T = 1, for each head, feature, realization.
        in_features = kbar.shape[-2]
        num_realizations = kbar.shape[-1]
        noise = jax.random.normal(rng_key, kbar.shape[-3:])
        noise = noise / jnp.sqrt(jnp.sqrt(in_features * num_realizations))
        # constrain the gate parameter to be in [0 1]
        gate = jax.nn.sigmoid(gate[..., None])
        # add to queries and keys.
        pe_coef, noise_coef = jnp.sqrt(gate), jnp.sqrt(1. - gate)
        qbar = pe_coef * qbar + noise_coef * noise
        kbar = pe_coef * kbar + noise_coef * noise

        return qbar, kbar


def apply_fpe(keys, spe):
    return (keys[..., None] * spe).reshape((*keys.shape[:-1], -1))
