import math
from typing import Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import lax
from jax_spectral import stft, istft
import numpy as np


class SPE(nn.Module):
    dimension: int = 1
    resolution: Union[int, Tuple[int]] = 200

    def setup(self):
        if isinstance(self.resolution, int) or len(self.resolution.shape) == 0:
            self.resolution = (self.resolution,) * self.dimension

        self.msd = self.param(
            'msd', smooth_init,
            (self.resolution[0] + 1, *self.resolution[1:]))
        self.window_size = [2 * d  for d in self.resolution]
        self.hop_size = [d//4 for d in self.resolution]

    def stft(self, x):
        return stft(
            x,
            nperseg=self.window_size[0],
            noverlap=self.window_size[0] - self.hop_size[0],
            window='hamming',
            #center=True,
            #normalized=True,
            return_onesided=True,
            #pad_mode='reflect',
            #return_complex=True
        )[2]

    def istft(self, x):
        return istft(
            x,
            nperseg=self.window_size[0],
            noverlap=self.window_size[0] - self.hop_size[0],
            window='hamming',
            #center=True,
            #normalized=True,
            input_onesided=True,
            #length=shape[0]
        )[1]

    def __call__(self, num: int, shape: Tuple[int]):
        original_shape = shape
        shape = [max(2*d, s) for (d,s) in zip(self.window_size, shape)]
        print(shape)
        if self.dimension != 1:
            raise NotImplementedError("for now SPE only works in 1d")

        # draw noise of appropriate shape
        # TODO: do not use the same key every time
        p = jax.random.normal(jax.random.PRNGKey(0), (num, *shape))
        eps = jnp.finfo(p.dtype).eps

        msd = nn.relu(self.msd)
        msd = msd / (jnp.linalg.norm(msd) + eps)

        # compute its STFT
        p = self.stft(p)

        # filter by the magnitude spectral density
        p = p * msd[None, :, None]

        # compute istft
        p = self.istft(p)  # shape=shape

        # weight by window in case of non perfect reconstruction
        #weight = torch.ones(*shape, device=p.device)
        #p = p / self.istft(self.stft(weight), shape=shape)[None]

        #normalize to get correlations
        p = p / (jnp.linalg.norm(p, axis=0) + eps)

        # truncate if needed
        p = p[tuple(slice(s) for s in (num, ) + original_shape)]
        return p


def smooth_init(key, shape, dtype=jnp.float32):
    msd = np.zeros(shape) + 1e-3
    L = min(msd.shape[0], 10)
    msd[:L] = msd[:L] + np.logspace(0, -2, L)
    return jnp.asarray(msd)
