import jax
import jax.numpy as jnp
import numpy as np
from flax import nn
import math

from . import signal


class SPE(nn.Module):

    def stft(self, x, window_size, hop_size):
        return signal.stft(
                    x,
                    nperseg=window_size[0],
                    noverlap=window_size[0] - hop_size[0],
                    window='hamming',
                    #center=True,
                    #normalized=True,
                    return_onesided=True,
                    #pad_mode='reflect',
                    #return_complex=True
                )[2]

    def istft(self, x, shape, window_size, hop_size):
        return signal.istft(
            x,
            nperseg=window_size[0],
            noverlap=window_size[0] - hop_size[0],
            window='hamming',
            #center=True,
            #normalized=True,
            input_onesided=True,
            #length=shape[0]
        )[1]

    def apply(self, num, shape, dimension=1, resolution=200):
        if isinstance(resolution, int) or len(resolution.shape) == 0:
            resolution = (resolution,) * dimension

        if isinstance(shape, int) or len(shape.shape) == 0:
            shape = (shape,) * dimension

        window_size = [2 * d  for d in resolution]
        hop_size = [d//4 for d in resolution]

        msd = self.param('msd', (resolution[0] + 1, *resolution[1:]),
                         smooth_init)

        original_shape = shape
        # TODO: zip() doesn't work in jax.jit?
        shape = [max(2*d, s) for (d,s) in zip(window_size, shape)]
        print(shape)
        if dimension != 1:
            raise NotImplementedError("for now SPE only works in 1d")

        # draw noise of appropriate shape
        p = jax.random.normal(jax.random.PRNGKey(0), (num, *shape))

        msd = nn.relu(msd)
        msd = msd / jnp.linalg.norm(msd)

        print('nfft', window_size[0], 'hop', hop_size[0])
        # compute its STFT
        p = self.stft(p, window_size=window_size, hop_size=hop_size)

        # filter by the magnitude spectral density
        p = p * msd[None, :, None]

        # compute istft
        p = self.istft(p, shape=shape, window_size=window_size, hop_size=hop_size)

        # weight by window in case of non perfect reconstruction
        #weight = torch.ones(*shape, device=p.device)
        #p = p / self.istft(self.stft(weight), shape=shape)[None]

        #normalize to get correlations
        p = p / jnp.linalg.norm(p, axis=0)

        # truncate if needed
        p = p[tuple(slice(s) for s in (num, ) + original_shape)]
        return p


def smooth_init(key, shape, dtype=jnp.float32):
    msd = np.zeros(shape) + 1e-3
    L = min(msd.shape[0], 10)
    msd[:L] = msd[:L] + np.logspace(0, -2, L)
    return jnp.asarray(msd)
