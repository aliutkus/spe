from confugue import configurable
import torch
from torch import nn
import torch.nn.functional as F

from .fast_transformer_decoder import FastTransformerDecoder
from .transformer_helpers import (
  TokenEmbedding,
  PositionalEncoding,
  weights_init
)

@configurable
class MusicPerformer(nn.Module):
  def __init__(self, n_token, d_model, d_embed, dropout=0.1, max_len=20480):
    super(MusicPerformer, self).__init__()
    self.n_token = n_token
    self.d_model = d_model
    self.max_len = max_len

    self.token_emb = TokenEmbedding(n_token, d_embed, d_model)
    self.d_embed = d_embed

    if self._cfg.get('add_positional_encoding', True):
      self.pe = self._cfg['positional_encoding'].configure(
          PositionalEncoding, d_embed=d_embed, max_pos=max_len)
    else:
      self.pe = None
    self.dec_out_proj = nn.Linear(d_model, n_token)

    self.transformer_decoder = self._cfg['decoder'].configure(
      FastTransformerDecoder,
      d_model=d_model,
      dropout=dropout
    )

    self.emb_dropout = nn.Dropout(dropout)
    self.apply(weights_init)

  def forward(self, x, attn_kwargs=None):
    x_emb = self.token_emb(x)
    x_inp = self.emb_dropout(x_emb)
    if self.pe:
      x_inp = x_inp + self.pe(x.size(1)).permute(1, 0, 2)

    dec_out = self.transformer_decoder(x_inp, attn_kwargs=attn_kwargs)
    dec_logits = self.dec_out_proj(dec_out)

    return dec_logits

  def compute_loss(self, dec_logits, dec_tgt, pad_index=None):
    if pad_index is None:
      pad_index = -100
    recons_loss = F.cross_entropy(
      dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1),
      ignore_index=pad_index, reduction='mean'
    ).float()

    return {
      'recons_loss': recons_loss,
      'total_loss': recons_loss
    }
