import torch
import torch.nn.functional as F
import torch.nn as nn
import math



class TwoLayerMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(TwoLayerMLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim // 2)
        self.linear2 = torch.nn.Linear(input_dim // 2, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.tanh(x)
        return x



class MultiHeadAttentionSubtraction(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttentionSubtraction, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for multi-head attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # # Pooling type: cls, mean, or max
        # assert pooling in ["cls", "mean", "max"], "Pooling must be 'cls', 'mean', or 'max'."
        # self.pooling = pooling

        self.transform_model = TwoLayerMLP(embed_dim)

    def forward(self, x, mask, text_array):
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device)
        batch_size, seq_length, embed_dim = x.size()

        # # Apply masked positional encoding
        # x = self.pos_encoding(x, mask)

        text_array = text_array.unsqueeze(1).expand(-1, seq_length, -1)
        x = x - text_array
        
        # Linear projections for queries, keys, and values
        Q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask to attention scores
        attention_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, seq_length, -1)
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax to get attention weights, then apply dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)  # Dropout on attention weights

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)  # Dropout on final output

        # mean pooling
        output = output.mean(dim=1)
        # normalize
        output = output / output.norm(dim=-1, keepdim=True)

        score = self.transform_model(output)

        return score