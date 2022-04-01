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
    """

    def __init__(
        self,
        num_heads: int = 8,
        in_features: int = 64,
        num_realizations: int = 256,
        num_sines: int = 1
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

        # normalize the gains
        self.gains.data[...] /= torch.sqrt(
            self.gains.norm(dim=-1, keepdim=True)) / 2.

        # bias initial frequencies to low values for long term range
        self.freqs.data[...] -= 4.

        self.code_shape = (num_heads, in_features)

    def forward(self, shape, num_realizations=None):
        """
        Generate the code, composed of a random QBar and Kbar,
        depending on the parameters, and return them for use with a
        SPE module to actually encode queries and keys.

        Args:
            shape: The outer shape of the inputs: (batchsize, *size)
            num_realizations: if provided, overrides self.num_realizations
        """
        if len(shape) != 2:
            raise ValueError('Only 1D inputs are supported by SineSPE')

        # get shape of the queries. Here it's only 1d
        max_len = shape[1]

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

        # gains is (num_heads, keys_dim, num_sines). Making then nonnegative with softplus
        gains = nn.functional.softplus(self.gains)

        # now upsample it
        gains = torch.stack(
            (gains, gains), dim=-1).view(
                self.num_heads, self.in_features, 2*self.num_sines)

        # the number of realizations is overrided by the function argument if provided
        if num_realizations is None:
            num_realizations = self.num_realizations

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            1, self.num_heads, self.in_features, 2 * self.num_sines,
            num_realizations,
            device=self.freqs.device) / math.sqrt(self.num_sines * 2)

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

        # final scaling
        scale = (num_realizations * self.in_features)**0.25
        return (qbar/scale, kbar/scale)

    def get_posattn_matrix(self, max_len=2048):
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
        gains = torch.stack(
            (gains, gains), dim=-1).view(
                self.num_heads, self.in_features, 2*self.num_sines)

        gains_squared_diag = torch.diag_embed(gains ** 2)

        print ('[get posattn matrix] Omega_q: {}, lambda: {}, Omega_k: {}'.format(
            omega_q.size(), gains_squared_diag.size(), omega_k.size()
        ))
        # print (gains_squared_diag[0, 0])

        # get (1, num_heads, keys_dim) attention maps, each of size (max_len, max_len)
        omega_q_mult_gains_squared_diag = torch.einsum(
            'ihdmk, hdku -> ihdmu',
            omega_q, gains_squared_diag
        )
        pos_attn_matrices = torch.einsum(
            'ihdmk, ihdnk -> ihdmn',
            omega_q_mult_gains_squared_diag, omega_k
        )
        print ('[get posattn matrix] pos_attn: {}'.format(
            pos_attn_matrices.size()
        ))

        return pos_attn_matrices


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

        scale = math.sqrt(torch.prod(torch.tensor(kernel_size).float())/2)
        self.conv_q.weight.data = self.conv_q.weight.data / scale
        self.conv_k.weight.data = self.conv_k.weight.data / scale

        self.code_shape = (num_heads, in_features)

    def forward(self, shape, num_realizations=None):
        """
        generate the random QBar and Kbar, depending on the parameters,
        Args:
            shape: The outer shape of the inputs: (batchsize, *size)
        """
        batchsize = 1
        original_shape = shape[1:]

        # decide on the size of the signal to generate
        # (larger than desired to avoid border effects)
        shape = [4*k+s for (k, s) in zip(self.kernel_size, original_shape)]

        # the number of realizations is overrided by the function argument if provided
        if num_realizations is None:
            num_realizations = self.num_realizations

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            batchsize*num_realizations,
            self.num_heads * self.in_features,
            *shape,
            device=self.conv_q.weight.device)

        # apply convolution, get (batchsize*num_realizations, num_heads*keys_dim, *shape)
        kbar = self.conv_q(z)
        qbar = self.conv_k(z)

        # truncate to desired shape (remove the start to avoid the border effects)
        indices = [slice(batchsize * num_realizations), slice(self.num_heads * self.in_features)]
        for dim in range(len(shape)):
            k = self.kernel_size[dim]
            s = original_shape[dim]
            indices.append(slice(k, k + s, 1))

        qbar = qbar[indices]
        kbar = kbar[indices]

        # making (batchsize, num_realizations, num_heads, keys_dim, *shape)
        kbar = kbar.view(batchsize, num_realizations,
                         self.num_heads, self.in_features, *original_shape)
        qbar = qbar.view(batchsize, num_realizations,
                         self.num_heads, self.in_features, *original_shape)

        # permuting to be
        # (batchsize, *shape, num_heads, keys_dim, num_realizations) as desired
        qbar = qbar.permute(0, *[x for x in range(4, self.ndim+4)], 2, 3, 1)
        kbar = kbar.permute(0, *[x for x in range(4, self.ndim+4)], 2, 3, 1)

        # final scaling
        scale = (num_realizations * self.in_features)**0.25
        return (qbar/scale, kbar/scale)

    def get_posattn_matrix(self, shape, num_realizations=None):
        batchsize = 1
        original_shape = shape[1:]

        # decide on the size of the signal to generate
        # (larger than desired to avoid border effects)
        shape = [4*k+s for (k, s) in zip(self.kernel_size, original_shape)]

        # the number of realizations is overrided by the function argument if provided
        if num_realizations is None:
            num_realizations = self.num_realizations

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            batchsize*num_realizations,
            self.num_heads * self.in_features,
            *shape,
            device=self.conv_q.weight.device)

        # apply convolution, get (batchsize*num_realizations, num_heads*keys_dim, *shape)
        kbar = self.conv_q(z)
        qbar = self.conv_k(z)

        for dim in range(len(shape)):
            k = self.kernel_size[dim]
            s = original_shape[dim]

            indices = [slice(batchsize*num_realizations),
                       slice(self.num_heads*self.in_features)] + [slice(k, k+s, 1), ]
            qbar = qbar[indices]
            kbar = kbar[indices]

        print ('[get posattn matrix] Qbar: {}, Kbar: {}'.format(
            qbar.size(), kbar.size()
        ))

        # get (num_heads * keys_dim) attention maps, each of size (max_len, max_len) 
        pos_attn_matrices = torch.einsum(
            'rdm, rdn -> dmn',
            qbar, kbar
        )
        print ('[get posattn matrix] pos_attn: {}'.format(
            pos_attn_matrices.size()
        ))

        # reshape attention maps to the same shape as those of SineSPE
        pos_attn_matrices = pos_attn_matrices.view(
            batchsize, self.num_heads, self.in_features, original_shape[-1], original_shape[-1]
        )

        return pos_attn_matrices


class SPEFilter(nn.Module):
    """Stochastic positional encoding filter

    Applies a positional code provided by a SPE module on actual queries and keys.
    Implements gating, i.e. some "dry" parameter, that lets original queries and keys through if activated.

    Args:
    gated: whether to use the gated version, which learns to balance
        positional and positionless features.
    code_shape: the inner shape of the codes, i.e. (num_heads, key_dim),
        as given by `spe.code_shape`
    """
    def __init__(
        self,
        gated: bool = True,
        code_shape: Optional[Tuple[int, int]] = None,
    ):
        super(SPEFilter, self).__init__()

        self.gated = gated
        self.code_shape = code_shape

        # create the gating parameters if required
        if gated:
            if code_shape is None:
                raise RuntimeError('code_shape has to be provided if gated is True.')
            self.register_parameter('gate', nn.Parameter(
                torch.randn(code_shape)
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

        # check that codes have the shape we are expecting
        if self.code_shape is not None and qbar.shape[-3:-1] != self.code_shape:
            raise ValueError(
                f'The inner shape of codes is {qbar.shape[-3:-1]}, '
                f'but expected {self.code_shape}')

        # check shapes: size of codes should be bigger than queries, keys
        code_size = qbar.shape[1:-3]
        query_size = queries.shape[1:-2]
        if (len(code_size) != len(query_size)
            or torch.any(
                torch.tensor(code_size) < torch.tensor(query_size)
            )):
                raise ValueError(f'Keys/queries have length {query_size}, '
                                 f'but expected at most {code_size}')
        if qbar.shape[-3:-1] != queries.shape[-2:]:
            raise ValueError(f'shape mismatch. codes have shape {qbar.shape}, '
                             f'but queries are {queries.shape}')

        # truncate qbar and kbar for matching current queries and keys,
        # but only if we need to
        for dim in range(len(query_size)):
            if code_size[dim] > query_size[dim]:
                indices = [slice(1), *[slice(qbar.shape[1+k]) for k in range(dim)],
                           slice(query_size[dim])]
                qbar = qbar[indices]
                kbar = kbar[indices]

        # apply gate if required
        if self.gated:
            # incorporate the constant bias for Pd if required. First draw noise
            # such that noise noise^T = 1, for each head, feature, realization.
            # qbar is : (1, *shape, num_heads, keys_dim, num_realizations)
            in_features = qbar.shape[-2]
            num_realizations = qbar.shape[-1]
            gating_noise = torch.randn(
                self.code_shape+(num_realizations,),
                device=queries.device) / (in_features * num_realizations)**0.25
            # normalize it so that it's an additive 1 to Pd
            #gating_noise = gating_noise / gating_noise.norm(dim=2, keepdim=True)

            # constrain the gate parameter to be in [0 1]
            gate = torch.sigmoid(self.gate[..., None])

            # qbar is (1, *shape, num_heads, keys_dim, num_realizations)
            # gating noise is (num_heads, keys_dim, num_realizations)
            # gate is (num_heads, keys_dim, 1)
            #import ipdb; ipdb.set_trace()
            qbar = torch.sqrt(1.-gate) * qbar  + torch.sqrt(gate) * gating_noise
            kbar = torch.sqrt(1.-gate) * kbar  + torch.sqrt(gate) * gating_noise

        # sum over d after multiplying by queries and keys
        # qbar/kbar are (1, *shape, num_heads, keys_dim, num_realizations)
        # queries/keys  (batchsize, *shape, num_heads, keys_dim)
        qhat = (qbar * queries[..., None]).sum(axis=-2)
        khat = (kbar * keys[..., None]).sum(axis=-2)

        # result is (batchsize, ..., num_heads, num_realizations)
        return qhat, khat
