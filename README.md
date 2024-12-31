# Transformer Model in PyTorch

This README explains how the Transformer model is implemented in PyTorch, covering the different components involved in building the model from scratch.

---

## Overview

The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al., has revolutionized natural language processing (NLP) by using attention mechanisms rather than recurrent networks to process sequences. This implementation uses PyTorch to build the following components:

- **Multi-Head Attention**: Helps the model focus on different parts of the input sequence simultaneously.
- **Transformer Block**: Combines multi-head attention and feed-forward layers, with residual connections and layer normalization.
- **Encoder**: Processes the input sequence to generate contextualized representations.
- **Transformer Model**: Integrates the encoder into the final model architecture.

---

## Code Breakdown

### 1. **Multi-Head Attention**

This is the core mechanism in the Transformer model, allowing the model to focus on different parts of the input sequence simultaneously. Each attention head learns to attend to different aspects of the sequence, which enhances the model's ability to capture contextual relationships.

#### Key Steps in Multi-Head Attention:
- **Embedding size and heads**: The `embed_size` represents the dimensionality of the input sequence, while `heads` denotes the number of attention heads.
- **Linear transformations**: The input is transformed into queries, keys, and values using linear layers.
- **Scaled dot-product attention**: The attention scores are calculated by taking the dot product of queries and keys, which are then scaled.
- **Masking**: The attention scores can be masked to ignore certain tokens (such as padding tokens).
- **Softmax and Weighted Sum**: The attention scores are normalized using softmax and used to compute a weighted sum of the values.

### 2. **Transformer Block**

A Transformer Block is composed of:
- **Multi-Head Attention**: Computes attention scores and aggregates information from the sequence.
- **Feed-Forward Layer**: A simple feed-forward neural network applied after attention.
- **Residual Connections**: Both attention and feed-forward layers include residual connections, which help stabilize training.
- **Layer Normalization**: Applied to the outputs of each sub-layer to improve training stability.

#### Key Steps in Transformer Block:
- Apply multi-head attention on the input.
- Add residual connections and normalize the result.
- Pass through a feed-forward network.
- Add another residual connection and normalize the final output.

### 3. **Encoder**

The Encoder processes the input sequence by embedding the words and adding positional encodings to the input to give the model information about the position of words in the sequence. It then passes the input through multiple Transformer Blocks to produce a sequence of contextualized embeddings.

#### Key Components of the Encoder:
- **Word Embeddings**: An embedding layer transforms word indices into vectors.
- **Positional Embeddings**: These embeddings encode the position of each word in the sequence, allowing the model to differentiate between words in different positions.
- **Transformer Blocks**: A stack of Transformer Blocks processes the input sequence.

### 4. **Transformer Model**

The complete Transformer model consists of the Encoder and the necessary components to handle input sequences and padding tokens. It also defines a method to generate a mask to ignore padding tokens in the input.

#### Key Components of the Transformer Model:
- **Encoder**: A stack of layers that processes the input sequence.
- **Padding Mask**: A mask is created to ignore padding tokens in the input sequence during attention calculation.
- **Forward Pass**: The model processes the source sequence through the encoder and generates the final output.

---

## How the Code Works

1. **Multi-Head Attention**:
   - The `MultiHeadAttention` class handles splitting the input into different heads, calculating attention scores, and combining the results. The output is passed through a linear layer to aggregate the results of all attention heads.

2. **Transformer Block**:
   - The `TransformerBlock` class applies multi-head attention followed by a feed-forward layer. It also incorporates residual connections and layer normalization to stabilize training.

3. **Encoder**:
   - The `Encoder` class takes in the source sequence and processes it through the embedding layers and a stack of Transformer blocks. It outputs a sequence of contextualized embeddings that capture the relationships between words in the sequence.

4. **Transformer**:
   - The `Transformer` class integrates the encoder and handles the padding token mask. It processes the input sequence and generates the final output.

---

## How to Use the Model

1. **Initialization**:
   - Create an instance of the `Transformer` model with appropriate parameters such as the vocabulary size, embedding size, number of layers, and attention heads.
   
   ```python
   model = Transformer(src_vocab_size=10000, trg_vocab_size=10000, src_pad_idx=0, trg_pad_idx=0)
