#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "Expected exactly 2 arguments: config, model_dir" >&2
  exit 1
fi

config=$1
model_dir=$2

set -ex
mkdir -p "$model_dir"
python -m lra_benchmarks.image.train --config="$config" --model_dir="$model_dir/" --task_name=cifar10
