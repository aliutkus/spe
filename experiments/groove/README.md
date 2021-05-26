# Groove continuation experiments

Code and configuration files for training Performers on the Groove2Groove dataset.

## Setup

In a Python 3.7 environment:
```bash
pip install -r requirements.txt
pip install -e lib/fast-transformers
pip install -e ../../src/pytorch
pip install -e ./src
```

## Data
Run the Jupyter notebook [`data/prepare.ipynb`](./data/prepare.ipynb) to download and prepare the dataset.

## Training

```bash
python -m spe_music.train_performer_grv2grv --model-dir $DIR
```
`$DIR` should be a directory containing a `config.yaml` file. To log to [Neptune](https://neptune.ai/), set the `NEPTUNE_API_TOKEN` and `NEPTUNE_PROJECT` environment variables.
Optionally, use `--name` to specify the experiment name for Neptune (otherwise it will be equal to the model directory path).

## Evaluation

The evaluation metrics are implemented in [`spe_music.style_eval`](./src/spe_music/style_eval) module. Running the evaluation consists of two steps: 1. generate continuations using the [`exp/continuation.ipynb`](./exp/continuation.ipynb) notebook, 2. compute metrics using the [`exp/style_eval_midi.ipynb`](./exp/continuation.ipynb) notebook.
