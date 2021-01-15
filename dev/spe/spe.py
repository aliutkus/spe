import torch
from torch import nn
import math


class ConvSPE(nn.Module):
    """Convolutional stochastic positional encoding.

    Args:
        ndim: The number of attention dimensions (e.g. 1 = sequence,
            2 = image).
        num_heads: The number of attention heads.
        in_features: The number of input features per attention head.
            If the actual key/query dimension is greater, only the
            first `in_features` will be used and the rest will be
            copied to the output unchanged. This is useful for keeping
            some features non-positional.
        num_realizations: The number of realizations of the stochastic
            process (R).
        kernel_size: The size of the convolution kernel.
    """

    def __init__(
        self,
        ndim=1,
        num_heads=8,
        in_features=64,
        num_realizations=256,
        kernel_size=200
    ):
        super(ConvSPE, self).__init__()

        if ndim == 1:
            conv_class = nn.Conv1d
        elif ndim == 2:
            conv_class = nn.Conv2d
        elif ndim == 3:
            conv_class = nn.Conv3d
        else:
            raise Exception('`ndim` must be 1, 2 or 3')

        # making kernel_size a list of length dimension in any case
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, ] * ndim

        # saving dimensions
        self.ndim = ndim
        self.in_features = in_features
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.num_realizations = num_realizations

        # create the two convolution layers
        self.conv_q = conv_class(
            in_channels=num_heads * in_features,
            out_channels=num_heads * in_features,
            stride=1,
            kernel_size=kernel_size,
            padding=0,
            bias=False,
            groups=num_heads * in_features)
        self.conv_k = conv_class(
            in_channels=num_heads * in_features,
            out_channels=num_heads * in_features,
            stride=1,
            kernel_size=kernel_size,
            padding=0,
            bias=False,
            groups=num_heads * in_features)

        # random init
        self.conv_q.weight.data = torch.rand(self.conv_q.weight.shape)
        self.conv_k.weight.data = torch.rand(self.conv_k.weight.shape)

        return

        # smooth init
        init_weight = 1.
        for d in range(dim):
            win = torch.hann_window(kernel_size[d])
            index = (None, None, *((None,)*d), Ellipsis, *(None,)*(dim-1-d))
            init_weight = init_weight * win[index]
        init_weight = init_weight / torch.sqrt(init_weight.norm())
        init_weight = init_weight.repeat(keys_dim * num_heads, 1, *((1,)*dim))
        self.conv_q.weight.data = init_weight.clone()
        self.conv_k.weight.data = init_weight.clone()

    def forward(self, queries, keys):
        """
        Perform SPE.

        Expects keys and queries of shape `(batch_size, ..., num_heads,
        key_dim)` and outputs keys and queries of shape `(batch_size,
        ..., num_heads, num_realizations + key_dim - in_features)`.
        """
        assert (queries.shape == keys.shape), \
            "As of current implementation, queries and keys must have the same shape. "\
            "got queries: {} and keys: {}".format(queries.shape, keys.shape)

        batchsize = queries.shape[0]

        if queries.shape[-1] < self.in_features:
            raise ValueError('Expected keys/queries of dimension at least'
                             f'{self.in_features}, got {queries.shape[-1]}.')

        # split off the non-positional part
        queries, queries_rest = _split_features(queries, self.in_features)
        keys, keys_rest = _split_features(keys, self.in_features)

        # making queries and keys (batchsize, num_heads, keys_dim, *shape)
        queries = queries.permute(
            0, self.ndim + 1, self.ndim + 2, *[d for d in range(1, self.ndim + 1)])
        keys = keys.permute(0, self.ndim + 1, self.ndim + 2,
                            *[d for d in range(1, self.ndim + 1)])

        # d = queries.shape[1] #d=num_heads*keys_dim
        original_shape = queries.shape[3:]

        # decide on the size of the signal to generate
        # (larger than desired to avoid border effects)
        shape = [4*k+s for (k, s) in zip(self.kernel_size, original_shape)]

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            batchsize*self.num_realizations,
            self.num_heads * self.in_features,
            *shape,
            device=self.conv_q.weight.device) / math.sqrt(self.num_realizations * self.in_features)

        # apply convolution, get (batchsize*num_realizations, num_heads*keys_dim, *shape)
        pe_k = self.conv_k(z)
        pe_q = self.conv_q(z)

        # truncate to desired shape
        for dim in range(len(shape)):
            k = self.kernel_size[dim]
            s = original_shape[dim]

            indices = [slice(batchsize*self.num_realizations),
                       slice(self.num_heads*self.in_features)] + [slice(k, k+s, 1), ]
            pe_k = pe_k[indices]
            pe_q = pe_q[indices]

        # making (batchsize, num_realizations, num_heads, keys_dim, *shape)
        pe_k = pe_k.view(batchsize, self.num_realizations,
                         self.num_heads, self.in_features, *original_shape)
        pe_q = pe_q.view(batchsize, self.num_realizations,
                         self.num_heads, self.in_features, *original_shape)

        # sum over d after multiplying by queries and keys
        qhat = (pe_q * queries[:, None]).sum(axis=3)
        khat = (pe_k * keys[:, None]).sum(axis=3)

        # qhat are (batchsize, num_realizations, num_heads, *shape), making them (batchsize, *shape, num_heads, num_realizations)
        qhat = qhat.permute(0, *[x for x in range(3, self.ndim+3)], 2, 1)
        khat = khat.permute(0, *[x for x in range(3, self.ndim+3)], 2, 1)

        # concatenate with the non-positional part of keys and queries
        qhat = torch.cat([qhat, queries_rest], dim=-1)
        khat = torch.cat([khat, keys_rest], dim=-1)

        return qhat, khat


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

    def __init__(
        self,
        num_heads=8,
        in_features=64,
        num_realizations=256,
        num_sines=10,
    ):
        super(SineSPE, self).__init__()

        # saving dimensions
        self.num_heads = num_heads
        self.in_features = in_features
        self.num_sines = num_sines
        self.num_realizations = num_realizations

        # register the parameter
        for param in ['freqs', 'offsets', 'gains']:
            self.register_parameter(
                param,
                nn.Parameter(
                    torch.randn(
                        num_heads,
                        in_features,
                        num_sines
                    )
                )
            )

    def forward(self, queries, keys):
        """
        Perform sinusoidal SPE.

        Expects keys and queries of shape `(batch_size, ..., num_heads,
        key_dim)` and outputs keys and queries of shape `(batch_size,
        ..., num_heads, num_realizations + key_dim - in_features)`.
        """
        assert (queries.shape == keys.shape), \
            "As of current implementation, queries and keys must have the same shape. "\
            "got queries: {} and keys: {}".format(queries.shape, keys.shape)

        batchsize = queries.shape[0]
        length = queries.shape[1]

        if queries.shape[-1] < self.in_features:
            raise ValueError('Expected keys/queries of dimension at least'
                             f'{self.in_features}, got {queries.shape[-1]}.')

        # split off the non-positional part
        queries, queries_rest = _split_features(queries, self.in_features)
        keys, keys_rest = _split_features(keys, self.in_features)

        # build omega_q and omega_k,
        # with shape (num_heads, keys_dim, length, 2*num_sines)
        indices = torch.linspace(0, length-1, length, device=self.freqs.device)

        # making sure the frequencies are in [0, 0.5]
        freqs = torch.sigmoid(self.freqs[:, :, None, :])/2.

        phases_q = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
            + self.offsets[:, :, None, :]
        )
        omega_q = torch.stack([torch.cos(phases_q), torch.sin(phases_q)], dim=-1).view(
            self.num_heads, self.in_features, length, 2*self.num_sines
        )

        phases_k = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
        )
        omega_k = torch.stack([torch.cos(phases_k), torch.sin(phases_k)], dim=-1).view(
            self.num_heads, self.in_features, length, 2*self.num_sines
        )

        # gains is (num_heads, keys_dim, 2*num_sines). Making then nonnegative with softplus
        gains = nn.functional.softplus(self.gains).repeat(1, 1, 2)

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            batchsize, self.num_heads, self.in_features, 2 * self.num_sines,
            self.num_realizations,
            device=self.freqs.device) / math.sqrt(self.num_realizations * self.in_features)

        # scale each of the 2*num_sines by the appropriate gain
        # z is still (batchsize, num_heads, keys_dim, 2*num_sines, num_realizations)
        z = z * gains[None, ..., None]

        # computing the sum over the sines. gets (batchsize, num_heads, keys_dim, length, num_realizations)
        qbar = torch.matmul(omega_q[None], z)
        kbar = torch.matmul(omega_k[None], z)

        # permuting them to be (batchsize, length, num_heads, keys_dim, num_realizations)
        qbar = qbar.permute(0, 3, 1, 2, 4)
        kbar = kbar.permute(0, 3, 1, 2, 4)

        # sum over the keys_dim after multiplying by queries and keys
        qhat = (qbar * queries[..., None]).sum(axis=-2)
        khat = (kbar * keys[..., None]).sum(axis=-2)

        # concatenate with the non-positional part of keys and queries
        qhat = torch.cat([qhat, queries_rest], dim=-1)
        khat = torch.cat([khat, keys_rest], dim=-1)

        return qhat, khat


def _split_features(x: torch.Tensor, num_positional: int):
    return x.split([num_positional, x.shape[-1] - num_positional], dim=-1)
