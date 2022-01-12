# Learnable-Fourier-Features

Unofficial pytorch implementation of the paper "Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding", NeurIPS 2021.
https://arxiv.org/pdf/2106.02795.pdf

Basic usage is as below:
```python
from positional_encoding import LearnableFourierFeaturesas LFF

lff = LFF(pos_dim=2, f_dim=128, h_dim=256, d_dim=64) # learnable fourier features module
pos = torch.randn([4, 1024, 1, 2])  # random positional coordinates
pe = lff(pos)  # forward

```

More detailed explanation of usage can be found in the file ```positional_encoding.py```, and other popular positional encoding methods such as the sinusoidal positional encoding from the paper "Attention is All You Need", and fourier features from the paper "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" are also implemented in the file.
