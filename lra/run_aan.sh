#!/bin/bash
data_dir=data/lra_release/tsv_data/

if [[ $# -ne 2 ]]; then
  echo "Expected exactly 2 arguments: config, model_dir" >&2
  exit 1
fi

config=$1
model_dir=$2

set -ex
mkdir -p "$model_dir"
python -m lra_benchmarks.matching.train --config="$config" --model_dir="$model_dir/" --vocab_file_path="$model_dir"/vocab --data_dir="$data_dir"
python -m lra_benchmarks.matching.train --config="$config" --model_dir="$model_dir/" --vocab_file_path="$model_dir"/vocab --data_dir="$data_dir" --test_only
