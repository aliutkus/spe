import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from warnings import warn
import itertools
import ipdb
import math
import numpy as np


class SPE(nn.Module):
    def __init__(self, dimension=1, shape=200):
        super(SPE, self).__init__()

        # making the resolution a tuple if it's an int
        if isinstance(shape, int):
            shape = (shape,) * dimension

        # initialize the psd with decaying frequencies
        self.register_buffer('psd', SPE.smooth_psd(shape, 0.2)))

    @staticmethod
    def smooth_psd(shape, lengthscale=0.2):
        ticks = [torch.range(d) for d in shape]
        frequencies = torch.stack(
            torch.meshgrid(ticks)
        )
        return torch.exp(
        - torch.sum(frequencies ** 2, dim=0) * 1./4.*lengthscale**2
        )


def spe(shape, n_draws=1, lengthscale=0.2):
    """Stochastic Positional Embedding

    Parameters:
    -----------
    shape : tuple of int
        the shape of the positional embeddings to generate
    n_draws: int
        the number of embeddings to generate
    """
    import math
    import numpy as np
    from torch.fft import irfftn, rfftn, stft


    # generating nd-frequencies
    ticks = [torch.linspace(0, 1000, 1000) for d in shape]
    frequencies = torch.stack(
        torch.meshgrid(ticks)
    )

    # power spectral densities on these frequencies
    psd = torch.exp(
        - torch.sum(frequencies ** 2, dim=0) * 1./4.*lengthscale**2
    )

    # generating random phase
    phase = torch.exp( 2j * math.pi * torch.rand((n_draws,) + psd.shape))

    # generating signals with the desired psd
    render_shape = [d*2 for d in shape]
    embeddings = torch.fft.irfftn(psd[None] * phase, render_shape)

    # reshaping (truncate to desired shape)
    embeddings = embeddings[[slice(0,d) for d in (n_draws, *shape)]]

    # normalize
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    embeddings = embeddings / embeddings.norm(dim=0, keepdim=True)
    return embeddings
