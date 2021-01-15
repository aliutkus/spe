# spe
Stochastic Positional Embedding


## installation

in the `dev` folder:
> pip install -e .


## usage

first create the positional encoder, either convolutive or sinusoidal

```python
import spe

convpe = spe.ConvSPE(
    dim=1, # dimension of the signal (1, 2 or 3) 
    keys_dim=keys_dim, # dimension for the keys and queries (per head)
    kernel_size=500, # size of the kernel. tuple of ints or int
    num_heads=num_heads, # number of heads
    num_realizations=num_realizations) # dimension of the queries and keys after encoding.

convpe = spe.SineSPE(
    dim=1, # dimension of the signal (1, 2 or 3) 
    keys_dim=keys_dim, # dimension for the keys and queries (per head)
    num_sines=num_sines, # number of sinusoids per pattern (per key dim)
    num_heads=num_heads, # number of heads
    num_realizations=num_realizations) # dimension of the queries and keys after encoding.

```

Then, use it with queries and keys that have a shape as: `(batchsize, length, num_heads, keys_dim)`.

It returns a `(queries, keys)` tuple, each one with new shape `(batchsize, length, num_heads, num_realizations)`, to be used with the performer, or dot product attention.