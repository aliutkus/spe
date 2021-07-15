# Long-Range Arena experiments

Original LRA repository: [google-research/long-range-arena](https://github.com/google-research/long-range-arena)

**Note:** A known bug in the ListOps task removes the brackets from the input, making the task essentially impossible to solve. The bug appears to have been fixed in the upstream repository in [c209b2a4](https://github.com/google-research/long-range-arena/commit/c209b2a48eedfd7ffcd13c679f97f3fa466c47bc), but the change is not merged here and we have not tested it. Our results were produced before the fix and are therefore affected by the bug.

## Setup

Install JAX (adjust the `jaxlib` version according to your CUDA version):
```bash
pip install jax==0.2.6 jaxlib==0.1.57+cuda102 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Install the rest of the requirements:
```bash
pip install -r requirements.txt
pip install -e ./fast_attention ./long-range-arena ../../src/jax
```

## Data

Download the data listed [here](https://github.com/google-research/long-range-arena). Adjust the paths in the training scripts (e.g. [`run_aan.sh`](./run_aan.sh)) if needed.

## Running the benchmark

There is a Bash script for each of the LRA tasks. For example, to perform the first run of the Performer on the Retrieval (AAN) task, run:
```bash
./run_aan.sh models/gpu_16g/performer_softmax/aan/r1/config.py models/gpu_16g/performer_softmax/aan/r1
```
Once the run is finished, the results will be in a `results.json` file inside the model directory.
