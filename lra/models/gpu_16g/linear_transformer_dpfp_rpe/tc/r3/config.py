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

from fast_self_attention import fast_self_attention as favor
import jax

from lra_benchmarks.text_classification.configs import base_tc_config

def dpfp(x, nu=1):
  x = jax.numpy.concatenate([jax.nn.relu(x), jax.nn.relu(-x)], axis=-1)
  x_rolled = jax.numpy.concatenate([jax.numpy.roll(x, shift=j, axis=-1) for j in range(1, nu + 1)], axis=-1)
  x_repeat = jax.numpy.concatenate([x] * nu, axis=-1)
  return x_repeat * x_rolled

def get_config():
  """Get the default hyperparameter configuration."""
  config = base_tc_config.get_config()
  config.random_seed = 2
  config.model_type = "transformer"
  config.attention_fn = favor.make_fast_generalized_attention(
    qkv_dim=config.qkv_dim // config.num_heads,
    features_type='deterministic',
    kernel_fn=dpfp,
    lax_scan_unroll=16)
  config.batch_size = 8
  config.learning_rate = config.learning_rate / 32 * 8
  config.num_train_steps = 30000

  config.model_kwargs = dict(
    pos_bias_cfg=dict(
      pos_bias_type="relative_key_query",
      max_seq_len=config.max_length
    ),
  )
  return config


def get_hyper(hyper):
  return hyper.product([])