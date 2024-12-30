import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Head Self-Attention Mechanism
"""
This class implements the Multi-Head Attention mechanism. It allows the model to 
focus on different parts of the input sequence simultaneously, capturing diverse 
contextual relationships between words.
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size  # Total size of word embeddings
        self.heads = heads  # Number of attention heads
        self.head_dim = embed_size // heads  # Dimension of each head

        # Ensure the embedding size is divisible by the number of heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by number of heads"

        # Layers for generating values, keys, and queries
        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)

        # Final linear layer to combine the attention heads
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Batch size
        N = query.shape[0]

        # Lengths of the input sequences
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Reshape the input to divide into multiple attention heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Calculate the attention scores using dot product of queries and keys
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply mask if provided (to ignore padding tokens)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize the attention scores using softmax
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Calculate the weighted sum of values using the attention scores
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        # Pass the result through the final linear layer
        return self.fc_out(out)

# Transformer Block
"""
This block combines multi-head attention and feed-forward layers. 
It also includes layer normalization and residual connections 
to stabilize the training process and improve performance.
"""
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        # Multi-Head Attention mechanism
        self.attention = MultiHeadAttention(embed_size, heads)

        # Layer Normalization for stability
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed-Forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),  # Expand dimensions
            nn.ReLU(),  # Apply non-linearity
            nn.Linear(forward_expansion * embed_size, embed_size)  # Reduce dimensions
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Apply multi-head attention
        attention = self.attention(value, key, query, mask)

        # Add residual connection and normalize
        x = self.dropout(self.norm1(attention + query))

        # Apply feed-forward network
        forward = self.feed_forward(x)

        # Add residual connection and normalize again
        return self.dropout(self.norm2(forward + x))

# Transformer Encoder
"""
The Encoder processes the input sequence to produce a contextualized representation. 
It includes word embeddings, positional embeddings, and a stack of transformer blocks.
"""
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size  # Size of the word embeddings
        self.device = device  # Device to run the model (CPU/GPU)

        # Embedding layer for words
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        # Positional embedding to encode the position of words
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Stack of transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout=dropout, forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Batch size and sequence length
        N, seq_length = x.shape

        # Create positional indices
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # Combine word and positional embeddings
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # Pass through each transformer block
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# Complete Transformer Model
"""
This class defines the full transformer architecture, including the encoder. 
It processes the input sequence and generates contextualized embeddings.
"""
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100,
    ):
        super(Transformer, self).__init__()
        # Initialize the encoder
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.device = device  # Device to run the model
        self.src_pad_idx = src_pad_idx  # Padding token index

    def make_src_mask(self, src):
        # Create a mask to ignore padding tokens
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def forward(self, src):
        # Generate the source mask
        src_mask = self.make_src_mask(src)

        # Pass the source sequence through the encoder
        enc_src = self.encoder(src, src_mask)

        return enc_src
