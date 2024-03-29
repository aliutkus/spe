import math
import torch
import numpy as np

from torch import nn
import torch.nn.functional as F


def weight_init_normal(weight, normal_std):
  nn.init.normal_(weight, 0.0, normal_std)

def bias_init(bias):
  nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    # print ('[{}] initializing ...'.format(classname))

    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            weight_init_normal(m.weight, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            weight_init_normal(m.weight, 0.01)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    # else:
    #   print ('[{}] not initialized !!'.format(classname))


class PositionalEncoding(nn.Module):

    def __init__(self, d_embed, max_pos=20480, learnable=False):
        super(PositionalEncoding, self).__init__()
        self.d_embed = d_embed
        self.max_pos = max_pos
        self.learnable = learnable

        pe = torch.zeros(max_pos, d_embed)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        if learnable:
          self.register_parameter('pe', nn.Parameter(pe))
        else:
          self.register_buffer('pe', pe)

    def forward(self, seq_len, bsz=None):
        pos_encoding = self.pe[:seq_len, :]

        if bsz is not None:
          pos_encoding = pos_encoding.expand(seq_len, bsz, -1)

        return pos_encoding


class TokenEmbedding(nn.Module):
  def __init__(self, n_token, d_embed, d_proj):
    super(TokenEmbedding, self).__init__()

    self.n_token = n_token
    self.d_embed = d_embed
    self.d_proj = d_proj
    self.emb_scale = d_proj ** 0.5

    self.emb_lookup = nn.Embedding(n_token, d_embed)
    if d_proj != d_embed:
      self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
    else:
      self.emb_proj = None

  def forward(self, inp_tokens):
    inp_emb = self.emb_lookup(inp_tokens)
    if self.emb_proj is not None:
      inp_emb = self.emb_proj(inp_emb)

    return inp_emb.mul_(self.emb_scale)

if __name__ == "__main__":
  pos_enc = PositionalEncoding(64)
  print (pos_enc.pe.size())

  print (pos_enc(512, 8).size())

  tkn_emb = TokenEmbedding(512, 256, 256)
  rand_inp = torch.randint(high=512, size=(32,))
  print (tkn_emb(rand_inp).size())