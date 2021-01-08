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

        # window size is twice +1 the size of the PSD
        self.window_size = 2 * max_lag
        self.hop_size = max_lag//4

        # initialize the psd with decaying frequencies
        self.register_parameter(
            'psd', nn.Parameter(smooth_init(max_lag, dimension)))

    def stft(self, x):
        window = torch.hamming_window(
            self.window_size).to(self.psd.device)

        return torch.stft(
                    x,
                    n_fft=self.window_size,
                    hop_length=self.hop_size,
                    window=window,
                    center=True,
                    normalized=True,
                    onesided=True,
                    pad_mode='reflect',
                    return_complex=True
                )
    def istft(self, x, shape):
        window = torch.hamming_window(
            self.window_size).to(self.psd.device)

        return torch.istft(
            x,
            n_fft=self.window_size,
            hop_length=self.hop_size,
            window=window,
            center=True,
            normalized=True,
            onesided=True,
            length=shape
        )

    def forward(self, queries, keys, num):
        assert queries.shape == keys.shape, \
            "Queries and keys should have self matching in "\
            "current implementation of SpectralSPE"

        # get shapes
        (batchsize, n, d) = queries.shape


        length = max(2*self.window_size, n)
        # draw noise of appropriate shape
        z_m = torch.randn(d, batchsize, num, length, device=self.psd.device)/math.sqrt(num)
        z_n = torch.randn(d, batchsize, num, length, device=self.psd.device)/math.sqrt(num)

        # compute its STFT

        pe = {}
        for key, z in zip(['q', 'k'], (z_n, z_m)):
            # compute the pe
            pe[key] = self.stft(z.view(-1, length))
            [num_f, num_t] = pe[key].shape[-2:]
            pe[key] = pe[key].view(d, batchsize, num, num_f, num_t)
            
            psd = torch.relu(self.psd[:, None, None, :, None])
            pe[key] = pe[key] * psd
            pe[key] = pe[key].view(-1, num_f, num_t)
            pe[key] = self.istft(pe[key], shape = length)[:,:n]
            pe[key] = pe[key].view(d, batchsize, num, n)

        pe['q'] = pe['q'] + z_m[..., :n]
        pe['k'] = pe['k'] + z_n[..., :n]

        # making (batchsize, n, d, num)
        pe['q'] = pe['q'].permute(1, 3, 0, 2)
        pe['k'] = pe['k'].permute(1, 3, 0, 2)

        qhat = (pe['q'] * queries[..., None]).sum(axis=-2)
        khat = (pe['k'] * keys[..., None]).sum(axis=-2)


        return qhat, khat

class SimpleSpectralSPE(nn.Module):
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
        super(SimpleSpectralSPE, self).__init__()

        # initialize the psd with decaying frequencies
        test = torch.zeros(dimension, 2*max_lag+1)

        #self.register_parameter(
        #    'kernel', nn.Parameter(torch.hamming_window(2 * max_lag + 1)))
        #test = pad(torch.ones(max_lag//4), (7*max_lag//4+1,0))+1e-3
        test[:,50::20] = 1.
        #self.register_parameter(
        #    'kernel', nn.Parameter(10.*torch.rand(2 * max_lag + 1)))
        self.register_parameter(
            'kernel', nn.Parameter(test))

    def forward(self, queries, keys, num):
        assert (queries.shape[0] == keys.shape[0]
                and queries.shape[2] == keys.shape[2]), \
            "Queries and keys should have shape matching except "\
            "for length. got queries: {} and keys: {}".format(queries.shape, keys.shape)

        # get shapes
        (batchsize, m, d) = queries.shape
        n = keys.shape[1] 
        k = self.kernel.shape[1]

        # draw noise of appropriate shape
        z_m = torch.randn(d, batchsize, num, 2*k+m+n, device=self.kernel.device)/math.sqrt(num*d)
        z_n = torch.randn(d, batchsize, num, 2*k+m+n, device=self.kernel.device)/math.sqrt(num*d)

        # get the kernel
        kernel = self.kernel
        #psd = rfft(kernel, 2*n)
        #kernel = kernel/kernel.norm()
        psd = {
            'q':rfft(kernel, 2*k+m+n),
            'k':rfft(kernel.flip(dims=(-1,)), 2*k+m+n)
        }

        pe = {}
        for key, z, length in zip(['q', 'k'], (z_n, z_m), (m, n)):
            # compute the pe
            pe[key] = rfft(z.view(-1, z.shape[-1]))
            num_f = pe[key].shape[-1]
            pe[key] = pe[key].view(d, -1, num_f)
            pe[key] = pe[key] * psd[key][:, None]
            pe[key] = pe[key].view(-1, num_f)
            pe[key] = irfft(pe[key])[..., k:(k+length)]
            pe[key] = pe[key].view(d, batchsize, num, length)
        
        pe['q'] = pe['q'] + z_m[..., k:(k+m)]#zhat['k']#
        pe['k'] = z_n[..., k:(k+n)]#zhat['q']# 

        # making (batchsize, m/n, d, num)
        pe['q'] = pe['q'].permute(1, 3, 0, 2)
        pe['k'] = pe['k'].permute(1, 3, 0, 2)

        # sum over d after multiplying by queries and keys
        qhat = (pe['q'] * queries[..., None]).sum(axis=-2)
        khat = (pe['k'] * keys[..., None]).sum(axis=-2)

        return qhat, khat


def smooth_init(max_lag, dimension):
    psd = torch.zeros(dimension, max_lag+1)
    L = min(psd.shape[1], 5)
    psd[:,:L] = psd[:,:L] + torch.logspace(0, -2, L)[None, :]
    return psd


def toeplitz_matmul(toeplitz_column, toeplitz_row, tensor):
    """
    Performs multiplication T * M where the matrix T is Toeplitz.
    Args:
        - toeplitz_column (vector n or b x n) - First column of the Toeplitz matrix T.
        - toeplitz_row (vector n or b x n) - First row of the Toeplitz matrix T.
        - tensor (matrix n x p or b x n x p) - Matrix or vector to multiply the Toeplitz matrix with.
    Returns:
        - tensor (n x p or b x n x p) - The result of the matrix multiply T * M.
    """
    if toeplitz_column.size() != toeplitz_row.size():
        raise RuntimeError("c and r should have the same length (Toeplitz matrices are necessarily square).")

    toeplitz_shape = torch.Size((*toeplitz_column.shape, toeplitz_row.size(-1)))
    output_shape = broadcasting._matmul_broadcast_shape(toeplitz_shape, tensor.shape)
    broadcasted_t_shape = output_shape[:-1] if tensor.dim() > 1 else output_shape

    if tensor.ndimension() == 1:
        tensor = tensor.unsqueeze(-1)
    toeplitz_column = toeplitz_column.expand(*broadcasted_t_shape)
    toeplitz_row = toeplitz_row.expand(*broadcasted_t_shape)
    tensor = tensor.expand(*output_shape)

    if not torch.equal(toeplitz_column[..., 0], toeplitz_row[..., 0]):
        raise RuntimeError(
            "The first column and first row of the Toeplitz matrix should have "
            "the same first element, otherwise the value of T[0,0] is ambiguous. "
            "Got: c[0]={} and r[0]={}".format(toeplitz_column[0], toeplitz_row[0])
        )

    if type(toeplitz_column) != type(toeplitz_row) or type(toeplitz_column) != type(tensor):
        raise RuntimeError("The types of all inputs to ToeplitzMV must match.")

    *batch_shape, orig_size, num_rhs = tensor.size()
    r_reverse = toeplitz_row[..., 1:].flip(dims=(-1,))

    c_r_rev = torch.zeros(*batch_shape, orig_size + r_reverse.size(-1), dtype=tensor.dtype, device=tensor.device)
    c_r_rev[..., :orig_size] = toeplitz_column
    c_r_rev[..., orig_size:] = r_reverse

    temp_tensor = torch.zeros(
        *batch_shape, 2 * orig_size - 1, num_rhs, dtype=toeplitz_column.dtype, device=toeplitz_column.device
    )
    temp_tensor[..., :orig_size, :] = tensor

    fft_M = fft(temp_tensor.transpose(-1, -2).contiguous())
    fft_c = fft(c_r_rev).unsqueeze(-2).expand_as(fft_M)
    fft_product = fft_M.mul_(fft_c)

    output = ifft(fft_product).real.transpose(-1, -2)
    output = output[..., :orig_size, :]
    return output


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