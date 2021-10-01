# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration and hyperparameter sweeps."""

import functools

from fast_self_attention import fast_self_attention as favor
import jax_spe as spe

from lra_benchmarks.models.layers.spe import make_spe_transform_fn
from lra_benchmarks.matching.configs import base_match_config


def get_config():
  """Get the default hyperparameter configuration."""
  config = base_match_config.get_config()
  config.random_seed = 1
  config.model_type = "transformer"
  num_realizations = 64
  config.model_kwargs = dict(
    add_pos_emb=False,
    qk_transform_fn_factory=functools.partial(
      make_spe_transform_fn,
      spe_cls=spe.ConvSPE,
      spe_kwargs=dict(
        num_realizations=num_realizations,
        kernel_size=128
      ),
      shared=True
    )
  )
  config.attention_fn = favor.make_fast_softmax_attention(
    qkv_dim=num_realizations,
    lax_scan_unroll=16)
  config.batch_size = 8
  config.learning_rate = 0.005
  config.num_train_steps = 15000
  config.warmup = 3000
  config.eval_frequency = 1500
  return config


def get_hyper(hyper):
  return hyper.product([])
