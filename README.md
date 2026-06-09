# gpt-from-scratch

A small decoder-only transformer (GPT) implemented from scratch in PyTorch, following
Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).

The goal is understanding, not novelty: every component - autograd, the language-modelling
objective, self-attention - is built by hand before relying on the framework version.

## Components

1. **`micrograd/`** - a minimal scalar autograd engine and backpropagation, plus a tiny MLP
   trained with it.
2. **`bigram/`** - a character-level bigram language model: the simplest possible
   next-token predictor, with sampling and NLL loss.
3. **`gpt/`** - the main event: a decoder-only transformer (single-head -> multi-head
   self-attention -> blocks with residuals + layernorm), trained on tiny Shakespeare
   (~10M parameters, runs on Apple Silicon or a free Colab GPU).

## Status

In progress - components land in the order above.

## Run

```bash
uv sync
# per-component instructions in each directory as they land
```

Training data and checkpoints go in `data/` (gitignored).

## References

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) - Andrej Karpathy
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
