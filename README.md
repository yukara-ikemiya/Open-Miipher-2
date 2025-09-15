
# What Google can do and You cannot do

## Google USM

## Internal noise dataset

# 💡 Tips

## A layer number of an audio encoder for extracting audio feature

## Degradation types

## Frame-wise decoding of WaveFit in Miipher-2

TBD, 0.6 sec の学習 length、小さい音が学習時に含まれていない件について

# 🤔 Unclear points in the implementation

## 1. Parameter size

## 2. Upsampling method of USM feature

'nearest', 'linear'or 'bilinear' ?

## 3. Loss function of feature cleaner

The loss function for training feature cleaner is defined in the Miipher paper [2].
mean-absolute~ にするために要素数で割るの忘れている的な

# References

1. "Miipher-2: A Universal Speech Restoration Model for Million-Hour Scale Data Restoration", S. Karita et al., WASPAA 2025
1. "Miipher: A Robust Speech Restoration Model Integrating Self-Supervised Speech and Text Representations", Y. Koizumi, WASPAA 2021
1. "WaveFit: An Iterative and Non-autoregressive Neural Vocoder based on Fixed-Point Iteration", Y. Koizumi et al., IEEE SLT, 2022