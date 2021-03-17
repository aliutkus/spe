import math
from typing import Optional, Tuple, Union
import torch
from torch import nn


class SineFPE(nn.Module):
    """
    code generator for deterministic sinusoidal positional encoding.

    Args:
        num_heads: The number of attention heads.
        in_features: The number of input features per attention head.
        num_sines: The number of sin and cos components (K).
    """

    def __init__(
        self,
        num_heads: int = 8,
        in_features: int = 64,
        num_sines: int = 1
    ):
        super(SineFPE, self).__init__()

        # saving dimensions
        self.num_heads = num_heads
        self.in_features = in_features
        self.num_sines = num_sines

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

    def forward(self, shape):
        """
        Generate the code, composed of a random QBar and Kbar,
        depending on the parameters, and return them for use with a
        SPE module to actually encode queries and keys.

        Args:
            shape: The outer shape of the inputs: (batchsize, length)
        """
        if len(shape) != 2:
            raise ValueError('Only 1D inputs are supported by SinePE')

        # get shape of the queries. Here it's only 1d
        max_len = shape[1]

        # build omega_q and omega_k,
        # with shape (length, num_heads, keys_dim, 2*num_sines)
        positions = torch.linspace(0, max_len-1, max_len, device=self.freqs.device)

        # making sure the frequencies are in [0, 0.5]
        freqs = torch.sigmoid(self.freqs[None, :, :, :])/2.

        phases_q = (
            2 * math.pi
            * freqs * positions[:, None, None, None]
            + self.offsets[None, :, :, :]
        )
        omega_q = torch.stack([torch.cos(phases_q), torch.sin(phases_q)], dim=-1).view(
            1, max_len, self.num_heads, self.in_features, 2*self.num_sines
        )

        phases_k = (
            2 * math.pi
            * freqs * positions[:, None, None, None]
        )
        omega_k = torch.stack([torch.cos(phases_k), torch.sin(phases_k)], dim=-1).view(
            1, max_len, self.num_heads, self.in_features, 2*self.num_sines
        )

        # gains is (num_heads, keys_dim, num_sines). Making then nonnegative with softplus
        gains = nn.functional.softplus(self.gains)

        # now upsample it to (num_heads, keys_dim, 2*num_sines)
        gains = torch.stack(
            (gains, gains), dim=-1).view(
                self.num_heads, self.in_features, 2*self.num_sines)

        # scale each of the 2*num_sines by the appropriate gain
        qbar = omega_q * gains[None, ...]
        kbar = omega_k * gains[None, ...]

        # final scaling
        scale = (2 * self.num_sines * self.in_features)**0.25
        return (qbar/scale, kbar/scale)


class FPEFilter(nn.Module):
    """Positional encoding filter

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
        super(FPEFilter, self).__init__()

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
        Apply PE on keys.
        """
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
            # NOTE: Naïve gating!

            # constrain the gate parameter to be in [0 1]
            gate = torch.sigmoid(self.gate[..., None])

            # qbar is (1, *shape, num_heads, keys_dim, num_realizations)
            # gating noise is (num_heads, keys_dim, num_realizations)
            # gate is (num_heads, keys_dim, 1)
            qbar = torch.sqrt(1.-gate) * qbar  + torch.sqrt(gate)
            kbar = torch.sqrt(1.-gate) * kbar  + torch.sqrt(gate)

        # Multiply by positional code to get (batchsize, length, num_heads, keys_dim, 2 * num_sines),
        # then flatten to (batchsize, length, num_heads, keys_dim * 2 * num_sines)
        qhat = (queries[..., None] * qbar).view(*queries.shape[:-1], -1)
        khat = (keys[..., None] * kbar).view(*keys.shape[:-1], -1)

        # result is (batchsize, ..., num_heads, num_realizations)
        return qhat, khat