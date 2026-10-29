 # Wiki103GPT-18

A custom, 18.55-million-parameter causal language model trained from scratch on the WikiText-103 raw dataset using PyTorch.

## Model Architecture
- **Architecture:** GPT-style causal Transformer
- **Parameters:** 18.55 million
- **Transformer layers:** 4
- **Attention heads:** 4 per layer
- **Embedding dimension:** 256
- **Context length:** 128 tokens
- **Tokenizer:** Byte Pair Encoding (BPE), 32,000-token vocabulary
- **Training objective:** Autoregressive next-token prediction

## Implementation
- Causal multi-head self-attention with fused QKV projections
- Pre-LayerNorm Transformer blocks with residual connections
- Feed-forward networks using ReLU activation
- AdamW optimizer and gradient clipping
- FP16 mixed-precision training on supported NVIDIA GPUs
- Training and validation loss tracking
- Model checkpoint saving and resumption
- Text generation using temperature and top-k sampling

## Training Results

The recorded training run completed 50,000 steps on an NVIDIA Tesla T4 GPU.

- **Training time:** Approximately 60.5 minutes
- **Final training loss:** 4.127
- **Final validation loss:** 4.008
- **Validation perplexity:** 55.01

These results describe this particular training run. The generated text demonstrates basic language-pattern learning but can still be repetitive, inaccurate, or incoherent.

## Requirements
Python, PyTorch, Hugging Face Datasets, and the Tokenizers library. The notebook also requires the separately trained BPE tokenizer JSON file.

## Purpose
This project explores implementing, training, evaluating, and optimizing a small GPT-like language model from scratch, with an emphasis on understanding Transformer architecture and practical training workflows.
