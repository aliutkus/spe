import torch
from torch import nn
import math


class SPE(nn.Module):
    def __init__(self, dimension=1, resolution=200):
        super(SPE, self).__init__()

        # saving dimension
        self.dimension = dimension

        # making the resolution a tuple if it's an int
        if isinstance(resolution, int):
            resolution = (resolution,) * dimension

        # window size is twice +1 the size of the PSD
        self.window_size = [2 * d  for d in resolution]
        self.hop_size = [d//4 for d in resolution]#resolution

        # initialize the psd with decaying frequencies
        self.register_parameter(
            'msd', nn.Parameter(smooth_init(resolution)))

    def stft(self, x):
        window = torch.hamming_window(
            self.window_size[0]).to(self.msd.device)

        return torch.stft(
                    x,
                    n_fft=self.window_size[0],
                    hop_length=self.hop_size[0],
                    window=window,
                    center=True,
                    normalized=True,
                    onesided=True,
                    pad_mode='reflect',
                    return_complex=True
                )
    def istft(self, x, shape):
        window = torch.hamming_window(
            self.window_size[0]).to(self.msd.device)

        return torch.istft(
            x,
            n_fft=self.window_size[0],
            hop_length=self.hop_size[0],
            window=window,
            center=True,
            normalized=True,
            onesided=True,
            length=shape[0]
        )

    def forward(self, num, shape):
        if isinstance(shape, int):
            shape = (shape,) * self.dimension

        original_shape = shape
        shape = [max(2*d, s) for (d,s) in zip(self.window_size, shape)]
        print(shape)
        if self.dimension != 1:
            raise NotImplementedError("for now SPE only works in 1d")

        # draw noise of appropriate shape
        p = torch.randn(num, *shape, device=self.msd.device)
        eps = torch.finfo(p.dtype).eps

        msd = torch.relu(self.msd)
        msd = msd / (torch.norm(msd) + eps)

        print('nfft', self.window_size[0], 'hop', self.hop_size[0])
        # compute its STFT
        p = self.stft(p)

        # filter by the magnitude spectral density
        p = p * msd[None, :, None]

        # compute istft
        p = self.istft(p, shape=shape)

        # weight by window in case of non perfect reconstruction
        #weight = torch.ones(*shape, device=p.device)
        #p = p / self.istft(self.stft(weight), shape=shape)[None]

        #normalize to get correlations
        p = p / (torch.norm(p, dim=0) + eps)

        # truncate if needed
        p = p[[slice(s) for s in (num, ) + original_shape]]
        return p


def smooth_init(shape):
    msd = torch.zeros(shape[0]+1) + 1e-3
    L = min(msd.shape[0], 10)
    msd[:L] = msd[:L] + torch.logspace(0, -2, L)
    return msd

class ConvSPE(nn.Module):
    def __init__(self, dimension=1, kernel_size=200):
        super(ConvSPE, self).__init__()


        if dimension==1:
            conv_class = nn.Conv1d
        elif dimension==2:
            conv_class = nn.Conv2d
        elif dimension==3:
            conv_class = nn.Conv3d
        else:
            raise Exception('dimension must be 1, 2 or 3')

        # making kernel_size a list of length dimension in any case
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size,] * dimension 

        # saving dimensions
        self.dimension = dimension
        self.kernel_size = kernel_size

        # create the convolution layer
        self.conv = conv_class(
                        in_channels=1,
                        out_channels=1,
                        stride=1,
                        kernel_size=kernel_size,
                        padding=0,
                        bias=False)

        # smooth_init        
        init_weight = 1.
        for d in range(dimension):
            win = torch.hann_window(kernel_size[d]//4)
            index = (None, None, *((None,)*d), Ellipsis, *(None,)*(dimension-1-d))
            init_weight = init_weight * win[index]
        self.conv.weight.data = init_weight


    def forward(self, num, shape):
        if isinstance(shape, int):
            shape = (shape,) * self.dimension

        original_shape = shape

        # decide on the size of the signal to generate
        # (larger than desired to avoid border effects)
        shape = [2 * max(d, s) for (d,s) in zip(self.kernel_size, shape)]
        print(shape)

        # draw noise of appropriate shape on the right device
        p = torch.randn(num, 1, *shape, device=self.conv.weight.device)
        # apply convolution
        p = self.conv(p)[:, 0, ...]
        #normalize to get correlations
        p = p / torch.norm(p, dim=0)

        # truncate to desired shape
        p = p[[slice(s) for s in (num, ) + original_shape]]

        return p