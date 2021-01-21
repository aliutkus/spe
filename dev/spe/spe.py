import math
from typing import Optional, Tuple, Union

import torch
from torch import nn


class SineSPE(nn.Module):
    """
    code generator for sinusoidal stochastic positional encoding.

    Args:
        num_heads: The number of attention heads.
        in_features: The number of input features per attention head.
        num_realizations: The number of realizations of the stochastic
            process (R).
        num_sines: The number of sin and cos components (K).
        gated: Whether to use the gated version, which learns to balance
            positional and positionless features.
    """

    def __init__(
        self,
        num_heads: int = 8,
        in_features: int = 64,
        num_realizations: int = 256,
        num_sines: int = 10,
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

        self.gains.data[...] = 1.
        # bias initial frequencies to low values for long term range
        self.freqs.data[...] -= 4.

    def forward(self, queries):
        """
        Generate the code, composed of a random QBar and Kbar,
        depending on the parameters, and return them for use with a
        SPE module to actually encode queries and keys.

        Args:
        queries: a torch.Tensor that is only used to infer the shape of the codes to generate
        """

        # get shape of the queries. Here it's only 1d
        max_len = queries.shape[1]

        # build omega_q and omega_k,
        # with shape (num_heads, keys_dim, length, 2*num_sines)
        indices = torch.linspace(0, max_len-1, max_len, device=self.freqs.device)

        # making sure the frequencies are in [0, 0.5]
        freqs = torch.sigmoid(self.freqs[:, :, None, :])/2.

        phases_q = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
            + self.offsets[:, :, None, :]
        )
        omega_q = torch.stack([torch.cos(phases_q), torch.sin(phases_q)], dim=-1).view(
            1, self.num_heads, self.in_features, max_len, 2*self.num_sines
        )

        phases_k = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
        )
        omega_k = torch.stack([torch.cos(phases_k), torch.sin(phases_k)], dim=-1).view(
            1, self.num_heads, self.in_features, max_len, 2*self.num_sines
        )

        # gains is (num_heads, keys_dim, 2*num_sines). Making then nonnegative with softplus
        gains = nn.functional.softplus(self.gains)
        #gains = gains / torch.sqrt(gains.norm(dim=-1, keepdim=True))
        gains = gains.repeat(1, 1, 2)


        # draw noise of appropriate shape on the right device
        z = torch.randn(
            1, self.num_heads, self.in_features, 2 * self.num_sines,
            self.num_realizations,
            device=self.freqs.device) / math.sqrt(self.in_features * self.num_sines * 2)

        # scale each of the 2*num_sines by the appropriate gain
        # z is still (1, num_heads, keys_dim, 2*num_sines, num_realizations)
        z = z * gains[None, ..., None]

        # computing the sum over the sines.
        # gets (1, num_heads, keys_dim, length, num_realizations)
        qbar = torch.matmul(omega_q, z)
        kbar = torch.matmul(omega_k, z)

        # permuting them to be (1, length, num_heads, keys_dim, num_realizations)
        qbar = qbar.permute(0, 3, 1, 2, 4)
        kbar = kbar.permute(0, 3, 1, 2, 4)

        return (qbar, kbar)


class ConvSPE(nn.Module):
    """
    code generator for convolutive stochastic positional encoding.

    Args:
        ndim: The number of attention dimensions (e.g. 1 = sequence,
            2 = image).
        num_heads: The number of attention heads.
        in_features: The number of input features per attention head.
        num_realizations: The number of realizations of the stochastic
            process (R).
        kernel_size: The size of the convolution kernel.
    """

    def __init__(
        self,
        ndim: int = 1,
        num_heads: int = 8,
        in_features: int = 64,
        num_realizations: int = 256,
        kernel_size: Union[int, Tuple[int, ...]] = 200,
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
        self.conv_q.weight.data = self.conv_q.weight.data / torch.sqrt(self.conv_q.weight.data.norm(keepdim=True))
        self.conv_k.weight.data = self.conv_k.weight.data / torch.sqrt(self.conv_k.weight.data.norm(keepdim=True))


    def forward(self, queries):
        """
        generate the random QBar and Kbar, depending on the parameters,
        Args:
            queries: (batchsize, *shape, num_heads, keys_dim)
        """
        batchsize = 1
        original_shape = queries.shape[1:-2]

        # decide on the size of the signal to generate
        # (larger than desired to avoid border effects)
        shape = [4*k+s for (k, s) in zip(self.kernel_size, original_shape)]

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            batchsize*self.num_realizations,
            self.num_heads * self.in_features,
            *shape,
            device=self.conv_q.weight.device) / math.sqrt(self.in_features)

        # apply convolution, get (batchsize*num_realizations, num_heads*keys_dim, *shape)
        kbar = self.conv_q(z)
        qbar = self.conv_k(z)

        # truncate to desired shape (remove the start to avoid the border effects)
        for dim in range(len(shape)):
            k = self.kernel_size[dim]
            s = original_shape[dim]

            indices = [slice(batchsize*self.num_realizations),
                       slice(self.num_heads*self.in_features)] + [slice(k, k+s, 1), ]
            qbar = qbar[indices]
            kbar = kbar[indices]

        # making (batchsize, num_realizations, num_heads, keys_dim, *shape)
        kbar = kbar.view(batchsize, self.num_realizations,
                         self.num_heads, self.in_features, *original_shape)
        qbar = qbar.view(batchsize, self.num_realizations,
                         self.num_heads, self.in_features, *original_shape)


        # permuting to be (batchsize, *shape, num_heads, keys_dim, num_realizations) as desired
        qbar = qbar.permute(0, *[x for x in range(4, self.ndim+4)], 2, 3, 1)
        kbar = kbar.permute(0, *[x for x in range(4, self.ndim+4)], 2, 3, 1)

        return (qbar, kbar)


class SPEFilter(nn.Module):
    """Stochastic positional encoding filter

    Applies a positional code provided by a SPE module on actual queries and keys.
    Implements gating, i.e. some "dry" parameter, that lets original queries and keys through if activated.

    Args:
    gated: whether to use the gated version, which learns to balance
        positional and positionless features.
    spe: if gated, then a spe instance must be provided
    """
    def __init__(
        self,
        gated: bool=True,
        spe = None,
    ):
        super(SPEFilter, self).__init__()

        self.gated = gated

        # create the gating parameters if required
        if gated:
            if spe is None:
                raise RuntimeError('the spe instance has to be provided if gated is True.')
            self.spe = spe
            self.register_parameter('gate', nn.Parameter(
                torch.randn(spe.num_heads, spe.in_features) - 2.
            ))

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        code: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SPE on keys with a given code.

        Expects keys and queries of shape `(batch_size, ..., num_heads,
        key_dim)` and outputs keys and queries of shape `(batch_size,
        ..., num_heads, num_realizations)`. code is the tuple
        of the 2 tensors provided by the code instance, each one of
        shape (1, ..., num_heads, key_dim, num_realizations)
        """
        assert (queries.shape == keys.shape), \
            "As of current implementation, queries and keys must have the same shape. "\
            "got queries: {} and keys: {}".format(queries.shape, keys.shape)


        # qbar and kbar are (1, *shape, num_heads, keys_dim, num_realizations)
        (qbar, kbar) = code

        # check shapes: size of codes should be bigger than queries, keys
        code_shape = qbar.shape[1:-3]
        query_shape = queries.shape[1:-2]
        if (len(code_shape) != len(query_shape)
            or torch.any(
                torch.tensor(code_shape) < torch.tensor(query_shape)
            )):
                raise RuntimeError(f'Keys/queries have length {query_shape}, '
                                f'but expected at most {code_shape}')
        if qbar.shape[-3:-1] != queries.shape[-2:]:
            raise RuntimeError(f'shape mismatch. codes have shape {qbar.shape}, '
                               f'but queries are {queries.shape}')


        # truncate qbar and kbar for matching current queries and keys
        for dim in range(len(query_shape)):
            indices = [slice(1), *[slice(qbar.shape[1+k]) for k in range(dim)],
                       slice(query_shape[dim])]
            qbar = qbar[indices]
            kbar = kbar[indices]

        # apply gate if required
        if self.gated:
            print('gating !')
            # incorporate the constant bias for Pd if required. First draw noise
            # such that noise noise^T = 1, for each head, feature, realization.
            gating_noise = torch.randn(
                self.spe.num_heads, self.spe.in_features, self.spe.num_realizations,
                device=queries.device) / math.sqrt(self.spe.in_features)
            # normalize it so that it's an additive 1 to Pd
            #gating_noise = gating_noise / gating_noise.norm(dim=2, keepdim=True)

            # constrain the gate parameter to be in [0 1]
            gate = torch.sigmoid(self.gate[..., None])

            # qbar is (1, *shape, num_heads, keys_dim, num_realizations)
            # gating noise is (num_heads, keys_dim, num_realizations)
            # gate is (num_heads, keys_dim, 1)
            #import ipdb; ipdb.set_trace()
            qbar = (1.-gate) * qbar  + gate * gating_noise
            kbar = (1.-gate) * kbar  + gate * gating_noise

        #qbar = qbar / qbar.norm(dim = -1, keepdim=True)
        #kbar = kbar / kbar.norm(dim = -1, keepdim=True)

        # sum over d after multiplying by queries and keys
        # qbar/kbar are (1, *shape, num_heads, keys_dim, num_realizations)
        # queries/keys  (batchsize, *shape, num_heads, keys_dim)
        qhat = (qbar * queries[..., None]).sum(axis=-2)
        khat = (kbar * keys[..., None]).sum(axis=-2)

        # result is (batchsize, ..., num_heads, num_realizations)
        return qhat, khat
