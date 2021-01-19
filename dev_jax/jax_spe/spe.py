import math
from typing import Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class SineSPE(nn.Module):
    num_heads: int = 1
    in_features: Union[int, Tuple[int]] = 64
    num_realizations: int = 64
    num_sines: int = 10

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

    def setup(self):
        params_shape = (self.num_heads, self.in_features, self.num_sines)
        # bias initial frequencies to low values for long term range
        self.freqs = self.param('freqs', lambda *args: jax.random.normal(*args) - 5.,
                                params_shape)
        self.offsets = self.param('offsets', jax.random.normal, params_shape)
        self.gains = self.param('gains', jax.random.normal, params_shape)

    def __call__(self, rng_key, length):
        """
        Generate sinusoidal SPEs.
        """
        # build omega_q and omega_k,
        # with shape (num_heads, keys_dim, length, 2*num_sines)
        indices = jnp.linspace(0, length-1, length)

        # making sure the frequencies are in [0, 0.5]
        freqs = jax.nn.sigmoid(self.freqs[:, :, None, :]) / 2.

        phases_q = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
            + self.offsets[:, :, None, :]
        )
        omega_q = jnp.stack([jnp.cos(phases_q), jnp.sin(phases_q)], axis=-1).reshape(
            self.num_heads, self.in_features, length, 2*self.num_sines
        )

        phases_k = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
        )
        omega_k = jnp.stack([jnp.cos(phases_k), jnp.sin(phases_k)], axis=-1).reshape(
            self.num_heads, self.in_features, length, 2*self.num_sines
        )

        # gains is (num_heads, keys_dim, 2*num_sines). Making them nonnegative with softplus
        gains = jax.nn.softplus(self.gains).repeat(2, axis=2)

        # draw noise of appropriate shape
        z = jax.random.normal(
            rng_key,
            (1, self.num_heads, self.in_features, 2 * self.num_sines, self.num_realizations),
        ) / math.sqrt(self.num_realizations * self.in_features)

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

        return qbar, kbar


def apply_spe(keys, spe):
    # split off the non-positional part
    keys, keys_rest = jnp.split(keys, [spe.shape[-2]], axis=-1)

    # sum over the keys_dim after multiplying by queries and keys
    # spe is (1, max_len, ...), truncating and broadcasting over the batch
    keys = (spe[:, :keys.shape[1]] * keys[..., None]).sum(axis=-2)

    # add the non-positional part back
    return jnp.concatenate([keys, keys_rest], axis=-1)
