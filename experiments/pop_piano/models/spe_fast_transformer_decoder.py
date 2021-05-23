import torch
from torch import nn
import torch.nn.functional as F

from fast_transformers.builders import AttentionBuilder
from fast_transformers.transformers import TransformerEncoderLayer
from fast_transformers.masking import TriangularCausalMask, LengthMask
from fast_transformers.attention import CausalLinearAttention, AttentionLayer
from fast_transformers.feature_maps import Favor

from spe import SPEFilter

class SPEFastTransformerDecoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, 
    dropout=0.1, 
    activation='relu',
    favor_feature_dims=None,
    spe_module=None,
    share_pe=False,
    share_spe_filter=False,
    use_gated_filter=True,
    spe_module_params=None
  ):
    super(SPEFastTransformerDecoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = dropout
    self.activation = activation
    self.share_pe = share_pe
    self.use_gated_filter = use_gated_filter
    self.share_spe_filter = share_spe_filter

    self.spe_module = spe_module
    self._spe = None
    self._spe_filters = None

    if share_pe:
      self.spe = self.spe_module(
                        num_heads=n_head, 
                        **(spe_module_params or {})
                      )
      self._spe = n_layer * [self.spe]
    else:
      self.spe = nn.ModuleList([
        self.spe_module(
              num_heads=n_head,
              in_features=d_model // n_head,
              **(spe_module_params or {})
            )
        for _ in range(n_layer)
      ])
      self._spe = list(self.spe)

    if share_spe_filter:
      self.spe_filters = SPEFilter(
                          code_shape=self._spe[0].code_shape, 
                          gated=use_gated_filter
                        )
      self._spe_filters = n_layer * [self.spe_filters]
    else:
      self.spe_filters = nn.ModuleList([
        SPEFilter(
          code_shape=pe.code_shape, 
          gated=use_gated_filter
        ) 
        for pe in self._spe
      ])
      self._spe_filters = list(self.spe_filters)

    self.favor_feature_dims = 2 * d_model // n_head \
                              if favor_feature_dims is None else favor_feature_dims
    att_builder = AttentionBuilder.from_kwargs(
      query_dimensions=d_model // n_head,
      feature_map=Favor.factory(n_dims=self.favor_feature_dims)
    )

    self.attention_layers = [
        AttentionLayer(
          att_builder.get("causal-linear"), 
          d_model, 
          n_head,
          positional_encoder=self._spe_filters[l].__call__
        )
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
    attn_mask = TriangularCausalMask(x.size(1), device=x.device)

    if lengths is not None:
      length_mask = LengthMask(lengths, device=x.device)
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