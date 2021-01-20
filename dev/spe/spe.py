import math
from typing import Optional, Tuple, Union

import torch
from torch import nn


class ConvSPE(nn.Module):
    """Convolutional stochastic positional encoding.

    Args:
        ndim: The number of attention dimensions (e.g. 1 = sequence,
            2 = image).
        num_heads: The number of attention heads.
        in_features: The number of input features per attention head.
        num_realizations: The number of realizations of the stochastic
            process (R).
        kernel_size: The size of the convolution kernel.
        gated: Whether to use the gated version, which learns to balance
            positional and positionless features.
    """

    def __init__(
        self,
        ndim: int = 1,
        num_heads: int = 8,
        in_features: int = 64,
        num_realizations: int = 256,
        kernel_size: Union[int, Tuple[int, ...]] = 200,
        gated: bool = True,
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
            kernel_size = (kernel_size,) * ndim

        # saving dimensions
        self.ndim = ndim
        self.in_features = in_features
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.num_realizations = num_realizations
        self.gated = gated

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

        # reset qbar and kbar
        self.reset()

        self.register_parameter('bias', nn.Parameter(
            torch.randn(num_heads, in_features) + 5.
        ))

    def reset(self):
        """
        Reset noise.
            at training, this is typically done for each new batch.
            at testing, this is typically never done
        """
        self.qbar = None
        self.kbar = None

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform SPE.

        Expects keys and queries of shape `(batch_size, ..., num_heads,
        key_dim)` and outputs keys and queries of shape `(batch_size,
        ..., num_heads, num_realizations + key_dim - in_features)`.
        """
        assert (queries.shape == keys.shape), \
            "As of current implementation, queries and keys must have the same shape. "\
            "got queries: {} and keys: {}".format(queries.shape, keys.shape)


        if queries.shape[-1] < self.in_features:
            raise ValueError('Expected keys/queries of dimension equal to'
                             f'{self.in_features}, got {queries.shape[-1]}.')


        # making queries and keys (batchsize, num_heads, keys_dim, *shape)
        queries = queries.permute(
            0, self.ndim + 1, self.ndim + 2, *[d for d in range(1, self.ndim + 1)])
        keys = keys.permute(0, self.ndim + 1, self.ndim + 2,
                            *[d for d in range(1, self.ndim + 1)])

        # Qbar and Kbar should be
        #(batchsize, num_realizations, num_heads, keys_dim, *shape)
        # if it's not the case, draw them anew. If it's the case, assume we keep them.
        desired_shape = (queries.shape[0], self.num_realizations, *queries.shape[1:])
        if self.qbar is None or self.qbar.shape != desired_shape:
            self._draw_noise(queries)

        # sum over d after multiplying by queries and keys
        qhat = (self.qbar * queries[:, None]).sum(axis=3)
        khat = (self.kbar * keys[:, None]).sum(axis=3)

        # qhat are (batchsize, num_realizations, num_heads, *shape), making them (batchsize, *shape, num_heads, num_realizations)
        qhat = qhat.permute(0, *[x for x in range(3, self.ndim+3)], 2, 1)
        khat = khat.permute(0, *[x for x in range(3, self.ndim+3)], 2, 1)


        return qhat, khat

    def _draw_noise(self, queries):
        """
        generate the random QBar and Kbar, depending on the parameters,
        and store them in the module.
        Args:
            queries: (batchsize, num_heads, keys_dim, *shape)
        """
        batchsize = queries.shape[0]
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
        self.kbar = self.conv_q(z)
        self.qbar = self.conv_k(z)

        # truncate to desired shape
        for dim in range(len(shape)):
            k = self.kernel_size[dim]
            s = original_shape[dim]

            indices = [slice(batchsize*self.num_realizations),
                       slice(self.num_heads*self.in_features)] + [slice(k, k+s, 1), ]
            self.qbar = self.qbar[indices]
            self.kbar = self.kbar[indices]

        # making (batchsize, num_realizations, num_heads, keys_dim, *shape)
        self.kbar = self.kbar.view(batchsize, self.num_realizations,
                         self.num_heads, self.in_features, *original_shape)
        self.qbar = self.qbar.view(batchsize, self.num_realizations,
                         self.num_heads, self.in_features, *original_shape)

        if self.gated:
            # and finally take incorporate the constant bias for Pd. First draw noise
            # such that noise noise^T = 1, for each head, feature, realization.
            bias_noise = torch.randn(
                1, 1, self.num_heads, self.in_features, self.num_realizations,
                device=z.device)
            # normalize it so that it's an additive 1 to Pd
            bias_noise = bias_noise / bias_noise.norm(dim=4, keepdim=True)
            # weight it by the bias parameter after constraining to [0 1]
            bias = torch.sigmoid(self.bias[None, None, ..., None])
            # add to queries and keys.
            self.qbar = bias * (self.qbar)  + (1.-bias) * bias_noise
            self.kbar = bias * (self.kbar)  + (1.-bias) * bias_noise


class SineSPE(nn.Module):
    """Sinusoidal stochastic positional encoding.

    Args:
        num_heads: The number of attention heads.
        in_features: The number of input features per attention head.
        num_realizations: The number of realizations of the stochastic
            process (R).
        num_sines: The number of sin and cos components (K).
        max_len: The maximum expected length of keys and queries. Can be
            set either here or by calling `reset()`, otherwise it is
            inferred from the actual keys and queries.
        gated: Whether to use the gated version, which learns to balance
            positional and positionless features.
    """

    def __init__(
        self,
        num_heads: int = 8,
        in_features: int = 64,
        num_realizations: int = 256,
        num_sines: int = 10,
        max_len: Optional[int] = None,
        gated: bool = True,
    ):
        super(SineSPE, self).__init__()

        # saving dimensions
        self.num_heads = num_heads
        self.in_features = in_features
        self.num_sines = num_sines
        self.num_realizations = num_realizations
        self.gated = gated

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

        # bias initial frequencies to low values for long term range
        self.freqs.data[...] -= 5.

        self.register_parameter('bias', nn.Parameter(
            torch.randn(num_heads, in_features) + 5.
        ))

        # reset qbar and kbar
        self.reset(max_len)

    def reset(self, max_len: Optional[int] = None):
        """
        Reset positional encodings.

        At training, this is typically done for each new batch.
        At testing, this is typically never done.

        Args:
            max_len: The maximum expected length of keys and queries.
        """
        self.qbar = None
        self.kbar = None
        if max_len is not None or not hasattr(self, 'max_len'):
            self.max_len = max_len

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SPE to queries and keys.

        Expects keys and queries of shape `(batch_size, ..., num_heads,
        key_dim)` and outputs keys and queries of shape `(batch_size,
        ..., num_heads, num_realizations + key_dim - in_features)`.
        """
        assert (queries.shape == keys.shape), \
            "As of current implementation, queries and keys must have the same shape. "\
            "got queries: {} and keys: {}".format(queries.shape, keys.shape)

        if queries.shape[-1] < self.in_features:
            raise ValueError('Expected keys/queries of dimension equal to '
                             f'{self.in_features}, got {queries.shape[-1]}.')

        length = keys.shape[1]
        if self.qbar is None:
            if self.max_len is None:
                self.max_len = length
            self._draw_noise()
        if self.qbar.shape[1] < length:
            raise RuntimeError(f'Keys/queries have length {length}, '
                               f'but expected at most {self.qbar.shape[1]}')
        desired_shape = (1, self.qbar.shape[1], *queries.shape[2:], self.num_realizations)
        if self.qbar.shape[2:] != desired_shape[2:]:
            raise RuntimeError(f'Positional encodings have shape {self.qbar.shape}, '
                               f'but need {desired_shape} for queries of shape '
                               f'{queries.shape}')

        # sum over the keys_dim after multiplying by queries and keys
        # qbar/kbar is (1, max_len, ...), truncating and broadcasting over the batch
        qhat = (self.qbar[:, :length] * queries[..., None]).sum(axis=-2)
        khat = (self.kbar[:, :length] * keys[..., None]).sum(axis=-2)

        return qhat, khat

    def _draw_noise(self):
        """
        Generate the random QBar and Kbar, depending on the parameters,
        and store them in the module.
        """
        # build omega_q and omega_k,
        # with shape (num_heads, keys_dim, length, 2*num_sines)
        indices = torch.linspace(0, self.max_len-1, self.max_len, device=self.freqs.device)

        # making sure the frequencies are in [0, 0.5]
        freqs = torch.sigmoid(self.freqs[:, :, None, :])/2.

        phases_q = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
            + self.offsets[:, :, None, :]
        )
        omega_q = torch.stack([torch.cos(phases_q), torch.sin(phases_q)], dim=-1).view(
            1, self.num_heads, self.in_features, self.max_len, 2*self.num_sines
        )

        phases_k = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
        )
        omega_k = torch.stack([torch.cos(phases_k), torch.sin(phases_k)], dim=-1).view(
            1, self.num_heads, self.in_features, self.max_len, 2*self.num_sines
        )

        # gains is (num_heads, keys_dim, 2*num_sines). Making then nonnegative with softplus
        gains = nn.functional.softplus(self.gains).repeat(1, 1, 2)

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            1, self.num_heads, self.in_features, 2 * self.num_sines,
            self.num_realizations,
            device=self.freqs.device) / math.sqrt(self.num_realizations * self.in_features)

        # scale each of the 2*num_sines by the appropriate gain
        # z is still (1, num_heads, keys_dim, 2*num_sines, num_realizations)
        z = z * gains[None, ..., None]

        # computing the sum over the sines.
        # gets (1, num_heads, keys_dim, length, num_realizations)
        self.qbar = torch.matmul(omega_q, z)
        self.kbar = torch.matmul(omega_k, z)

        # permuting them to be (1, length, num_heads, keys_dim, num_realizations)
        self.qbar = self.qbar.permute(0, 3, 1, 2, 4)
        self.kbar = self.kbar.permute(0, 3, 1, 2, 4)

        if self.gated:
            # and finally take incorporate the constant bias for Pd. First draw noise
            # such that noise noise^T = 1, for each head, feature, realization.
            bias_noise = torch.randn(
                1, 1, self.num_heads, self.in_features, self.num_realizations,
                device=z.device)
            # normalize it so that it's an additive 1 to Pd
            bias_noise = bias_noise / bias_noise.norm(dim=4, keepdim=True)
            # weight it by the bias parameter after constraining to [0 1]
            bias = torch.sigmoid(self.bias[None, None, ..., None])
            # add to queries and keys.
            self.qbar = bias * (self.qbar)  + (1.-bias) * bias_noise
            self.kbar = bias * (self.kbar)  + (1.-bias) * bias_noise
