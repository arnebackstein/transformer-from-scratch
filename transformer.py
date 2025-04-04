import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=100,    
        max_seq_len=50
    ):
        # We have this here as constants for simplicity
        n_heads = 2
        d_model=64

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # Note: Normally this would be done with nn.Embedding for efficiency
        # embedding = nn.Embedding(vocab_size, d_model)
        # But we use one-hot encoding with Linear for interpretability
        self.token_embedding = nn.Linear(vocab_size, d_model, bias=False)
        
        self.unembedding = nn.Linear(d_model, vocab_size)
        
        # Positional encoding will be computed on-the-fly
        # Create a buffer to store positional encodings
        self.register_buffer(
            'positional_encodings',
            self._create_positional_encodings(max_seq_len, d_model)
        )
        
        # First head
        self.W_q_head_1 = nn.Linear(d_model, d_model//n_heads)
        self.W_k_head_1 = nn.Linear(d_model, d_model//n_heads)
        self.W_v_head_1 = nn.Linear(d_model, d_model//n_heads)

        # Second head
        self.W_q_head_2 = nn.Linear(d_model, d_model//n_heads)
        self.W_k_head_2 = nn.Linear(d_model, d_model//n_heads)
        self.W_v_head_2 = nn.Linear(d_model, d_model//n_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass of the transformer
        x: input tokens of shape (batch_size, seq_len)
        Returns: logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, vocab_size)
        x_one_hot = F.one_hot(x, num_classes=self.vocab_size).float()
        
        # Note: This is equivalent to selecting rows from an embedding matrix
        # but done explicitly for interpretability
        x = self.token_embedding(x_one_hot)  # (batch_size, seq_len, d_model)
        
        x = x + self.positional_encodings[:, :seq_len, :]
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))  # (seq_len, seq_len)
        # Invert it so that True means "mask this position"
        causal_mask = 1 - causal_mask
        causal_mask = causal_mask.bool()
        
        # First head
        Q_head_1 = self.W_q_head_1(x)  # (batch_size, seq_len, d_model//n_heads)
        K_head_1 = self.W_k_head_1(x)  # (batch_size, seq_len, d_model//n_heads)
        V_head_1 = self.W_v_head_1(x)  # (batch_size, seq_len, d_model//n_heads)
        
        # Compute attention scores
        attn_scores_head_1 = torch.matmul(Q_head_1, K_head_1.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        attn_scores_head_1 = attn_scores_head_1 / math.sqrt(self.d_model // self.n_heads)
        
        # Apply causal mask explicitly
        # First expand mask to match batch dimension
        causal_mask = causal_mask.unsqueeze(0)  # (1, seq_len, seq_len)
        # Set masked positions to -inf
        attn_scores_head_1 = attn_scores_head_1.masked_fill(causal_mask, float('-inf'))
        
        # Compute attention weights
        attn_weights_head_1 = F.softmax(attn_scores_head_1, dim=-1)  # (batch_size, seq_len, seq_len)
        attn_output_head_1 = torch.matmul(attn_weights_head_1, V_head_1)  # (batch_size, seq_len, d_model//n_heads)

        # Second head (same process)
        Q_head_2 = self.W_q_head_2(x)
        K_head_2 = self.W_k_head_2(x)
        V_head_2 = self.W_v_head_2(x)
        
        attn_scores_head_2 = torch.matmul(Q_head_2, K_head_2.transpose(-2, -1))
        attn_scores_head_2 = attn_scores_head_2 / math.sqrt(self.d_model // self.n_heads)
        
        # Apply same causal mask
        attn_scores_head_2 = attn_scores_head_2.masked_fill(causal_mask, float('-inf'))
        
        attn_weights_head_2 = F.softmax(attn_scores_head_2, dim=-1)
        attn_output_head_2 = torch.matmul(attn_weights_head_2, V_head_2)

        # Concatenate heads
        att_output = torch.cat((attn_output_head_1, attn_output_head_2), dim=-1)
        x = self.layer_norm1(x + att_output)  # Add & Norm
        
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)  # Add & Norm
        
        x = self.unembedding(x)  # (batch_size, seq_len, vocab_size)
        
        return x 
    
    def _create_positional_encodings(self, max_seq_len, d_model):
        """
        Creates sinusoidal positional encodings
        PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        """
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension
        