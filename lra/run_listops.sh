#!/bin/bash
data_dir=data/lra_release/listops-1000/

if [[ $# -ne 2 ]]; then
  echo "Expected exactly 2 arguments: config, model_dir" >&2
  exit 1
fi

config=$1
model_dir=$2

set -ex
mkdir -p "$model_dir"
python -m lra_benchmarks.listops.train --config="$config" --model_dir="$model_dir/" --task_name=basic --data_dir="$data_dir"
python -m lra_benchmarks.listops.train --config="$config" --model_dir="$model_dir/" --task_name=basic --data_dir="$data_dir" --test_only
