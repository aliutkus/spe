import torch
from torch import nn
import math
from torch.fft import rfft, irfft
from torch.nn.functional import pad

class SpectralSPE(nn.Module):
    """
    Stochastic positional encoding: spectral method

    only available for time sequences for now"""

    def __init__(self, dimension=64, max_lag=200):
        """
        Create a Spectral SPE object.

        Parameters:
        ----------
        dimension: int
            the dimension for each keys/query
        max_lag: int
            the maximum lag handled by the encoding
        """
        super(SpectralSPE, self).__init__()

        # initialize the psd with decaying frequencies
        self.max_lag = max_lag
        kernel = torch.zeros(dimension, 2*max_lag+1)
        self.pos = torch.arange(max_lag, 2 * max_lag +1)
        self.neg = torch.arange(max_lag,0,-1)
        #kernel = torch.linspace(-10, 10, 2*max_lag).expand(dimension, 2*max_lag)
        start =min(max_lag, 100)
        #kernel[:, self.pos[start::20]] = 1
        kernel[:, self.pos[:start]] = torch.linspace(1, 0, start)[None]
        kernel[:, self.neg[:start]] = torch.linspace(1, 0, start)[None]
        #kernel = kernel-kernel.mean(dim=1, keepdim=True)
        kernel = kernel/kernel.norm(dim=1, keepdim=True)
        self.register_parameter(
            'kernel', nn.Parameter(kernel))

    def forward(self, queries, keys, num):
        assert (queries.shape[0] == keys.shape[0]
                and queries.shape[2] == keys.shape[2]), \
            "Queries and keys should have shape matching except "\
            "for length. got queries: {} and keys: {}".format(queries.shape, keys.shape)

        # get shapes
        (batchsize, m, d) = queries.shape
        n = keys.shape[1] 

        # get the kernel and take its RFFT
        kernel = self.kernel #/ self.kernel.norm(dim=1, keepdim=True)
        max_lag = min(self.max_lag-1, max(m,n))
        kernel = kernel[:,self.neg[max_lag]:self.pos[max_lag]]
        k = kernel.shape[1]

        # draw noise of appropriate shape as a PE for keys
        pe_k = torch.randn(d, batchsize, num, 2*k+m+n, device=self.kernel.device)/math.sqrt(num*d)
        #z = torch.randn(d, batchsize, num, k+n, device=self.kernel.device)/math.sqrt(num*d)
        #pe_k = pad(z, (0, k+m))

        psd = rfft(kernel, 2*k+m+n)

        pe_q = rfft(pe_k.view(-1, 2*k+m+n))
        num_f = pe_q.shape[-1]
        pe_q = pe_q.view(d, -1, num_f)
        pe_q = pe_q * psd[:, None]
        pe_q = pe_q.view(-1, num_f)
        pe_q = irfft(pe_q)[..., k+max_lag:(k+max_lag+m)]

        pe_q = pe_q.view(d, batchsize, num, m)
        pe_k = pe_k[..., k:(k+n)]

        # making (batchsize, m/n, d, num)
        pe_q = pe_q.permute(1, 3, 0, 2)
        pe_k = pe_k.permute(1, 3, 0, 2)

        # sum over d after multiplying by queries and keys
        qhat = (pe_q * queries[..., None]).sum(axis=-2)
        khat = (pe_k * keys[..., None]).sum(axis=-2)
        qhat = qhat - qhat.mean(axis=1, keepdim=True)
        #khat = khat - khat.mean(axis=1, keepdim=True)

        return qhat, khat

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