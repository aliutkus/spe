import torch
from torch import nn
import math
from torch.fft import rfft, irfft
from torch.nn.functional import pad
from gpytorch.utils.toeplitz import toeplitz_matmul

class SpectralSPE(nn.Module):
    """
    Stochastic positional encoding: spectral method

    only available for time sequences for now"""

    def __init__(self, dimension=64, max_lag=200, init_lengthscale=50):
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
        pos_lags = torch.zeros(dimension, max_lag)+1e-3
        neg_lags = torch.zeros(dimension, max_lag)+1e-3
        zero_lags = torch.zeros(dimension,)+1e-3

        # intialize for smooth signals with a bit of noise
        L = min(max_lag, init_lengthscale)
        pos_lags[:, :L] = torch.linspace(1,0,L)[None]
        neg_lags[:, :L] = torch.linspace(1,0,L)[None]
        pos_lags = pos_lags / pos_lags.norm(dim=1, keepdim=True)
        neg_lags = neg_lags / neg_lags.norm(dim=1, keepdim=True)

        # register the parameters
        self.register_parameter('pos_lags', nn.Parameter(pos_lags))
        self.register_parameter('neg_lags', nn.Parameter(neg_lags))
        self.register_parameter('zero_lags', nn.Parameter(zero_lags))

    def forward(self, queries, keys, num):
        """
        Perform SPE.
        queries and keys are (batchsize, keys_dim, sequence_length) tensors
        output is: (batchsize, num, sequence_length)
        """
        assert (queries.shape == keys.shape), \
            "As of current implementation, queries and keys must have the same shape. "\
            "got queries: {} and keys: {}".format(queries.shape, keys.shape)

        # get shapes and draw signals a bit larger to avoid border effects
        (batchsize, d, m) = queries.shape
        m = m + 2*self.max_lag
        n = keys.shape[-1] + 2*self.max_lag 
        pe_k = torch.randn(d, n,  batchsize*num, device=self.pos_lags.device)/math.sqrt(num*d)

        # prepare the toeplitz first row and col
        toeplitz_first_row = self.neg_lags[:, :min(self.max_lag, n-1)]
        toeplitz_first_row = pad(toeplitz_first_row, (1, n-1-toeplitz_first_row.shape[-1]))
        toeplitz_first_row[:, 0] = self.zero_lags

        toeplitz_first_col = self.pos_lags[:, :min(self.max_lag, m-1)]
        toeplitz_first_col = pad(toeplitz_first_col, (1, m-1-toeplitz_first_col.shape[-1]))
        toeplitz_first_col[:, 0] = self.zero_lags

        # perform toeplitz multiplication, get (d, n, batchsize * num)
        pe_q = toeplitz_matmul(toeplitz_first_col, toeplitz_first_row, pe_k).view(d, m, batchsize, num)
        pe_k = pe_k.view(d, n, batchsize, num)

        # truncate to get appropriate shapes
        pe_q = pe_q[:,self.max_lag:(self.max_lag + keys.shape[-1])]
        pe_k = pe_k[:,self.max_lag:(self.max_lag + keys.shape[-1])]

        # making (batchsize, num, d, m/n)
        pe_q = pe_q.permute(2, 3, 0, 1)
        pe_k = pe_k.permute(2, 3, 0, 1)

        # sum over d after multiplying by queries and keys
        qhat = (pe_q * queries[:, None]).sum(axis=2)
        khat = (pe_k * keys[:, None]).sum(axis=2)

        return qhat, khat

class ConvSPE(nn.Module):
    def __init__(self, rank=1, dimension=64, kernel_size=200):
        super(ConvSPE, self).__init__()

        if rank==1:
            conv_class = nn.Conv1d
        elif rank==2:
            conv_class = nn.Conv2d
        elif rank==3:
            conv_class = nn.Conv3d
        else:
            raise Exception('rank must be 1, 2 or 3')

        # making kernel_size a list of length dimension in any case
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size,] * rank 

        # saving dimensions
        self.dimension = dimension
        self.kernel_size = kernel_size

        # create the convolution layer
        self.conv = conv_class(
                        in_channels=dimension,
                        out_channels=dimension,
                        stride=1,
                        kernel_size=kernel_size,
                        padding=0,
                        bias=False,
                        groups=dimension)

        # smooth_init        
        init_weight = 1.
        for d in range(rank):
            win = torch.hann_window(kernel_size[d])
            win[kernel_size[d]//2] = 0
            index = (None, None, *((None,)*d), Ellipsis, *(None,)*(rank-1-d))
            init_weight = init_weight * win[index]
        init_weight = init_weight / torch.sqrt(init_weight.norm())
        self.conv.weight.data = init_weight.repeat(dimension, 1, *((1,)*rank))


    def forward(self, queries, keys, num):
        """
        perform SPE. 
        queries and keys are (batchsize, keys_dim, *shape) tensors
        output is: (batchsize, num, *shape)
        """
        assert (queries.shape == keys.shape), \
            "As of current implementation, queries and keys must have the same shape. "\
            "got queries: {} and keys: {}".format(queries.shape, keys.shape)


        batchsize = queries.shape[0] 
        d = queries.shape[1]
        original_shape = queries.shape[2:]

        # decide on the size of the signal to generate
        # (larger than desired to avoid border effects)
        shape = [2*k+s for (k,s) in zip(self.kernel_size, original_shape)]
        #shape = [2 * max(d, s) for (d,s) in zip(self.kernel_size, original_shape)]

        # draw noise of appropriate shape on the right device
        pe_k = torch.randn(batchsize * num, self.dimension, *shape, device=self.conv.weight.device) / math.sqrt(num*d)

        # apply convolution, get (batchsize*num, d, *shape)
        pe_q = self.conv(pe_k)

        # truncate to desired shape
        for dim in range(len(shape)):
            k = self.kernel_size[dim]
            s = original_shape[dim]
            indices_k = [slice(batchsize*num), slice(self.dimension)] + [slice(k//2, s+k//2, 1),]
            indices_q = [slice(batchsize*num), slice(self.dimension)] + [slice(0, s, 1),]
            pe_k = pe_k[indices_k]
            pe_q = pe_q[indices_q]

        # making (batchsize, num, d, *shape)
        pe_k = pe_k.view(batchsize, num, d, *original_shape)
        pe_q = pe_q.view(batchsize, num, d, *original_shape)


        # sum over d after multiplying by queries and keys
        qhat = (pe_q * queries[:, None]).sum(axis=2)
        khat = (pe_k * keys[:, None]).sum(axis=2)

        return qhat, khat