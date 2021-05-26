# Experiments on Pop Piano Generation

This directory corresponds to **Section 3.2** of the paper.

## Prerequisites
* Python >3.6
* Additional dependencies
```bash
pip3 install miditoolkit
pip3 install -e ../groove/lib/fast-transformers
pip3 install -e ../../src/pytorch
```

## Usage Notes
For detailed configuration settings, please read the `yaml` files under `configs/` directory.

### Training
```bash
python3 train.py [training config path] 
```
* e.g.
```bash
python3 train.py configs/train/sinespe_default.yaml 
```

### Inference
```bash
python3 inference.py [training config path] [inference config file]
```
* e.g.
```bash
python3 inference.py configs/train/sinespe_default.yaml  config/inference/default.yaml
```

### Evaluation (NLL Loss vs. Position)
```bash
python3 eval.py [training config path] [checkpoint path]
```
