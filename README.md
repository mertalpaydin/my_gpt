# My GPT

A clean, educational implementation of a GPT (Generative Pre-trained Transformer) model in PyTorch, closely following the architecture of GPT-2.

## Features

- **From Scratch Implementation**: Includes `CausalSelfAttention`, `MLP`, `Block`, and the full `GPT` model class.
- **Weight Loading**: configured to load pre-trained weights from local storage (`weights/trained_weights.pth`).
- **Text Generation**: Simple inference script to generate text from a prompt.
- **Educational Resources**: Includes detailed notebooks and reference papers.

## Project Structure

- `model/`: Contains the source code.
  - `gpt.py`: The model architecture (Attention, MLP, Transformer Blocks).
  - `config.py`: Configuration dataclass.
- `main.py`: Entry point for loading weights and generating text.
- `detailed_descriptions/`: Jupyter notebooks explaining core concepts (e.g., `causial_self_attention.ipynb`).
- `papers/`: Original research papers for reference.

## Installation

This project requires Python 3.11+.

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

## Usage

1. **Weights Setup**:
   Ensure you have the model weights placed at:
   ```
   weights/trained_weights.pth
   ```

2. **Run Inference**:
   ```bash
   uv run main.py
   ```
   
   This will:
   - Initialize the GPT model.
   - Load the weights from disk.
   - Generate text starting with the prompt: *"Alan Turing was a"*.

## References

- **Attention Is All You Need** (Vaswani et al., 2017)
- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019)
