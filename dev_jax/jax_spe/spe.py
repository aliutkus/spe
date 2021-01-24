import math
from typing import Tuple, Union

from flax import nn
import jax
import jax.numpy as jnp
import numpy as np


class SineSPE(nn.Module):
    """Sinusoidal stochastic positional encoding.

    Args:
        num_heads: The number of attention heads.
        in_features: The number of input features per attention head.
            If the actual key/query dimension is greater, only the
            first `in_features` will be used and the rest will be
            copied to the output unchanged. This is useful for keeping
            some features non-positional.
        num_realizations: The number of realizations of the stochastic
            process (R).
        num_sines: The number of sin and cos components (K).
    """

    def apply(
        self,
        rng_key,
        key_shape,
        num_realizations: int = 64,
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
        gains = self.param('gains', params_shape, jax.random.normal)

        # build omega_q and omega_k,
        # with shape (num_heads, keys_dim, length, 2*num_sines)
        indices = jnp.linspace(0, length-1, length)

        # making sure the frequencies are in [0, 0.5]
        freqs = jax.nn.sigmoid(freqs[:, :, None, :]) / 2.

        phases_q = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
            + offsets[:, :, None, :]
        )
        omega_q = jnp.stack([jnp.cos(phases_q), jnp.sin(phases_q)], axis=-1).reshape(
            num_heads, in_features, length, 2*num_sines
        )

        phases_k = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
        )
        omega_k = jnp.stack([jnp.cos(phases_k), jnp.sin(phases_k)], axis=-1).reshape(
            num_heads, in_features, length, 2*num_sines
        )

        # gains is (num_heads, keys_dim, 2*num_sines). Making them nonnegative with softplus
        gains = jax.nn.softplus(gains).repeat(2, axis=2)

        # draw noise of appropriate shape
        z = jax.random.normal(
            rng_key,
            (1, num_heads, in_features, 2 * num_sines, num_realizations),
        ) / jnp.sqrt(num_sines * 2)

        # scale each of the 2*num_sines by the appropriate gain
        # z is still (1, num_heads, keys_dim, 2*num_sines, num_realizations)
        z = z * gains[None, ..., None]

        # computing the sum over the sines.
        # gets (1, num_heads, keys_dim, length, num_realizations)
        qbar = jnp.matmul(omega_q[None], z)
        kbar = jnp.matmul(omega_k[None], z)

        # permuting them to be (1, length, num_heads, keys_dim, num_realizations)
        qbar = jnp.transpose(qbar, (0, 3, 1, 2, 4))
        kbar = jnp.transpose(kbar, (0, 3, 1, 2, 4))

        scale = jnp.sqrt(jnp.sqrt(jnp.reciprocal(num_realizations * in_features)))
        return scale * qbar, scale * kbar


class SPEGate(nn.Module):

    def apply(self, rng_key, spe_code):
        qbar, kbar = spe_code

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


def apply_spe(keys, spe):
    # sum over the keys_dim after multiplying by queries and keys
    # spe is (1, max_len, ...), truncating and broadcasting over the batch
    return (spe[:, :keys.shape[1]] * keys[..., None]).sum(axis=-2)

