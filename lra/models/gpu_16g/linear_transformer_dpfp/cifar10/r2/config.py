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

from lra_benchmarks.image.configs.cifar10 import base_cifar10_config
from lra_benchmarks.image.configs.cifar10.base_cifar10_config import TRAIN_EXAMPLES, VALID_EXAMPLES


NUM_EPOCHS = 200

def dpfp(x, nu=1):
  x = jax.numpy.concatenate([jax.nn.relu(x), jax.nn.relu(-x)], axis=-1)
  x_rolled = jax.numpy.concatenate([jax.numpy.roll(x, shift=j, axis=-1) for j in range(1, nu + 1)], axis=-1)
  x_repeat = jax.numpy.concatenate([x] * nu, axis=-1)
  return x_repeat * x_rolled

def get_config():
  """Get the hyperparameter configuration."""
  config = base_cifar10_config.get_config()
  config.random_seed = 1
  config.model_type = "transformer"
  config.learning_rate = .00025
  config.batch_size = 96
  config.eval_frequency = TRAIN_EXAMPLES // config.batch_size
  config.num_train_steps = (TRAIN_EXAMPLES // config.batch_size) * NUM_EPOCHS
  config.num_eval_steps = VALID_EXAMPLES // config.batch_size
  config.factors = 'constant * linear_warmup * cosine_decay'
  config.warmup = (TRAIN_EXAMPLES // config.batch_size) * 1

  config.model.dropout_rate = 0.3
  config.model.attention_dropout_rate = 0.2
  config.model.learn_pos_emb = True
  config.model.num_layers = 1
  config.model.emb_dim = 128
  config.model.qkv_dim = 64
  config.model.mlp_dim = 128
  config.model.num_heads = 8
  config.model.classifier_pool = "CLS"

  config.attention_fn = favor.make_fast_generalized_attention(
    qkv_dim=config.model.qkv_dim // config.model.num_heads,
    features_type='deterministic',
    kernel_fn=dpfp,
    lax_scan_unroll=16)
  return config


def get_hyper(hyper):
  return hyper.product([])
