import torch
from torch import nn
import torch.nn.functional as F

from spe_fast_transformer_decoder import SPEFastTransformerDecoder
from transformer_helpers import (
  TokenEmbedding,
  weights_init
)

from spe import SineSPE, ConvSPE

class MusicPerformerSPE(nn.Module):
  def __init__(self, n_token, n_layer, n_head, d_model, d_ff, d_embed, spe_type,
    activation='relu', dropout=0.1, favor_feature_dims=None,
    share_pe=False,
    share_spe_filter=False,
    use_gated_filter=True,
    spe_module_params=None
  ):
    super(MusicPerformerSPE, self).__init__()
    self.n_token = n_token
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = dropout
    self.activation = activation
    self.favor_feature_dims = favor_feature_dims

    self.token_emb = TokenEmbedding(n_token, d_embed, d_model)
    self.d_embed = d_embed

    self.spe_type = spe_type
    if self.spe_type == 'SineSPE':
      self.pe = SineSPE
    elif self.spe_type == 'ConvSPE':
      self.pe = ConvSPE
    else:
      raise ValueError('unrecognized SPE implementation: {}'.format(self.spe_type))

    self.dec_out_proj = nn.Linear(d_model, n_token)
    self.d_head = d_model // n_head
    self.transformer_decoder = SPEFastTransformerDecoder(
      n_layer, n_head, d_model, d_ff, dropout, activation, favor_feature_dims,
      spe_module=self.pe,
      share_pe=share_pe,
      share_spe_filter=share_spe_filter,
      use_gated_filter=use_gated_filter,
      spe_module_params=spe_module_params
    )

    self.emb_dropout = nn.Dropout(self.dropout)
    self.use_gated_filter = use_gated_filter
    self.apply(weights_init)
    
    print ('[info] model init completed')

  def forward(self, x, keep_last_only=False, attn_kwargs=None):
    x_emb = self.token_emb(x)
    x_inp = self.emb_dropout(x_emb)

    dec_out = self.transformer_decoder(x_inp, attn_kwargs=attn_kwargs)
    dec_logits = self.dec_out_proj(dec_out)

    if keep_last_only:
      dec_logits = dec_logits[:, -1, :]

    return dec_logits

  def compute_loss(self, dec_logits, dec_tgt, reduction='mean'):
    recons_loss = F.cross_entropy(
      dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
      ignore_index=self.n_token - 1, reduction=reduction
    ).float()

    return {
      'recons_loss': recons_loss,
      'total_loss': recons_loss
    }