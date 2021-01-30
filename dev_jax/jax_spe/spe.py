import functools
import math
from typing import Tuple, Union

from flax import nn
import jax
import jax.numpy as jnp
import numpy as np


class SineSPE(nn.Module):
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
        def init_gains(rng_key, shape):
            gains = jax.random.normal(rng_key, shape)
            return gains / (jnp.sqrt(jnp.linalg.norm(gains, axis=-1, keepdims=True)) / 2)
        gains = self.param('gains', params_shape, init_gains)

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

        # gains is (num_heads, keys_dim, num_sines). Making them nonnegative with softplus
        gains = jax.nn.softplus(gains)

        # now upsample it to 2 * num_sines
        gains = jnp.stack(
            [gains, gains], axis=-1
        ).reshape(num_heads, in_features, 2 * num_sines)

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


class ConvSPE(nn.Module):
    """
    Convolutive stochastic positional encoding.

    Args:
        rng_key: A PRNGKey.
        key_shape: The shape of keys and queries.
        num_realizations: The number of realizations of the stochastic
            process (R).
        kernel_size: The size of the convolution kernel.
    """

    def apply(
        self,
        rng_key,
        key_shape,
        num_realizations: int = 256,
        kernel_size: Union[int, Tuple[int, ...]] = 200,
    ):
        batchsize = 1
        original_shape = key_shape[1:-2]
        in_features = key_shape[-1]
        num_heads = key_shape[-2]
        ndim = len(original_shape)

        # making kernel_size a list of length dimension in any case
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim

        # create the two convolution layers
        kernel_init = functools.partial(
            jax.random.uniform,
            minval=0.,
            maxval=1 / jnp.sqrt(jnp.prod(jnp.array(kernel_size)) / 2))
        conv_q = nn.Conv.partial(
            features=num_heads * in_features,
            strides=(1,) * ndim,
            kernel_size=kernel_size,
            padding='VALID',
            bias=False,
            feature_group_count=num_heads * in_features,
            kernel_init=kernel_init,
            name='conv_q')
        conv_k = nn.Conv.partial(
            features=num_heads * in_features,
            strides=(1,) * ndim,
            kernel_size=kernel_size,
            padding='VALID',
            bias=False,
            feature_group_count=num_heads * in_features,
            kernel_init=kernel_init,
            name='conv_k')

        # decide on the size of the signal to generate
        # (larger than desired to avoid border effects)
        shape = [4 * kernel_size[d] + original_shape[d] for d in range(ndim)]

        # draw noise of appropriate shape on the right device
        z = jax.random.normal(
            rng_key,
            (batchsize * num_realizations,
             *shape,
             num_heads * in_features))

        # apply convolution, get (batchsize*num_realizations, num_heads*keys_dim, *shape)
        kbar = conv_q(z)
        qbar = conv_k(z)

        # truncate to desired shape (remove the start to avoid the border effects)
        for dim in range(ndim):
            k = kernel_size[dim]
            s = original_shape[dim]

            indices = (slice(batchsize * num_realizations), slice(k, k+s, 1))
            qbar = qbar[indices]
            kbar = kbar[indices]

        # making (batchsize, num_realizations, *shape, num_heads, keys_dim)
        kbar = kbar.reshape(
            batchsize, num_realizations, *original_shape, num_heads, in_features)
        qbar = qbar.reshape(
            batchsize, num_realizations, *original_shape, num_heads, in_features)

        # permuting to be
        # (batchsize, *shape, num_heads, keys_dim, num_realizations) as desired
        qbar = jnp.transpose(qbar, [0, *range(2, ndim + 4), 1])
        kbar = jnp.transpose(kbar, [0, *range(2, ndim + 4), 1])

        # final scaling
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
