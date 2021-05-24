from confugue import configurable
import spe
import torch
from torch import nn
import torch.nn.functional as F

from fast_transformers.transformers import TransformerEncoderLayer
from fast_transformers.masking import TriangularCausalMask, LengthMask
from fast_transformers.attention import CausalLinearAttention, AttentionLayer
from fast_transformers.feature_maps import Favor

device = "cuda:0"


@configurable
class FastTransformerDecoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, dropout=0.1, activation='relu',
               share_pe=False, share_spe_filter=False):
    super(FastTransformerDecoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = dropout
    self.activation = activation
    self.share_pe = share_pe
    self.share_spe_filter = share_spe_filter

    self._spe = None
    self._spe_filters = None
    if 'positional_encoder' in self._cfg:
      make_pe = self._cfg['positional_encoder'].bind(
        num_heads=n_head
      )
      if share_pe:
        self.spe = make_pe()  # Register as a module (only once!)
        self._spe = n_layer * [self.spe]
      else:
        # Make an SPE encoder for each layer and register them all
        self.spe = nn.ModuleList([
          make_pe() for _ in range(n_layer)
        ])
        self._spe = list(self.spe)

      make_filter = self._cfg['spe_filter'].bind(spe.SPEFilter)
      if share_spe_filter:
        self.spe_filters = make_filter(code_shape=self._spe[0].code_shape)
        self._spe_filters = n_layer * [self.spe_filters]
      else:
        # Make a filter for each layer, register them
        self.spe_filters = nn.ModuleList([
          make_filter(code_shape=pe.code_shape) for pe in self._spe
        ])
        self._spe_filters = list(self.spe_filters)

    self.attention_layers = [
        AttentionLayer(
          self._cfg['attention'].configure(
            CausalLinearAttention,
            query_dimensions=d_model // n_head,
            feature_map=self._cfg['feature_map'].configure(
              Favor.factory,
              n_dims=d_model // n_head
            )
          ),
          d_model, n_head,
          # Do not register as submodules of the layer
          positional_encoder=(
            self._spe_filters[l].__call__
            if self._spe_filters else None))
        for l in range(n_layer)
    ]

    self.decoder_layers = nn.ModuleList()
    for l in range(n_layer):
      self.decoder_layers.append(
        TransformerEncoderLayer(
          attention=self.attention_layers[l],
          d_model=d_model,
          d_ff=d_ff,
          dropout=dropout,
          activation=activation
        )
      )

  def forward(self, x, lengths=None, attn_kwargs=None):
    attn_mask = TriangularCausalMask(x.size(1), device=device)

    if lengths is not None:
      length_mask = LengthMask(lengths, device=device)
    else:
      length_mask = None

    attn_kwargs = dict(attn_kwargs) if attn_kwargs else {}

    if self._spe and self.share_pe and attn_kwargs.get('pos_code', None) is None:
      attn_kwargs['pos_code'] = self.spe(x.shape[:-1])

    out = x
    for l in range(self.n_layer):
      layer_attn_kwargs = dict(attn_kwargs)
      if self._spe and not self.share_pe and layer_attn_kwargs.get('pos_code', None) is None:
        layer_attn_kwargs['pos_code'] = self.spe[l](x.shape[:-1])

      out = self.decoder_layers[l](
        out,
        attn_mask=attn_mask,
        length_mask=length_mask,
        attn_kwargs=layer_attn_kwargs
      )

    return out


if __name__ == "__main__":
  dec = FastTransformerDecoder(
    12, 8, 512, 2048, 64
  ).to(device)

  for i in range(1000):
    dec_inp = torch.randn(1, 10240, 512).to(device)
    dec_seg = torch.randn(1, 10240, 64).to(device)
    out = dec(dec_inp, dec_seg)
    print (out.size())
    out.mean().backward()
