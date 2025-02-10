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

class TwoLayerMLPClass(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TwoLayerMLPClass, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim // 2)
        self.linear2 = torch.nn.Linear(input_dim // 2, num_classes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Define the Masked Cosine Positional Encoding
class MaskedCosinePositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super(MaskedCosinePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, x, mask):
        batch_size, seq_length, embed_dim = x.size()
        
        # Ensure embedding dimension matches
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch."
        
        # Initialize positional encodings to zero
        pe = torch.zeros(batch_size, seq_length, embed_dim, device=x.device)
        
        # Find valid positions based on mask
        valid_positions = mask == 1  # Shape: (batch_size, seq_length)
        
        # Create position tensor only for valid positions
        position = torch.arange(0, seq_length, dtype=torch.float, device=x.device).unsqueeze(0)
        position = position.expand(batch_size, -1)  # Shape: (batch_size, seq_length)
        
        # Only apply positions where mask == 1
        position = position * valid_positions.float()  # Positions are zero where mask is 0
        
        # Compute div term for cosine and sine terms
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=x.device).float() * (-math.log(10000.0) / embed_dim))
        
        # Apply cosine and sine functions only for masked positions
        pe[:, :, 0::2] = torch.cos(position.unsqueeze(-1) * div_term) * valid_positions.unsqueeze(-1)
        pe[:, :, 1::2] = torch.sin(position.unsqueeze(-1) * div_term) * valid_positions.unsqueeze(-1)
        # Add positional encoding to input
        x = x + pe
        return x

# Define single-head self-attention with positional encoding
class SingleHeadAttentionWithPositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super(SingleHeadAttentionWithPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_encoding = MaskedCosinePositionalEncoding(embed_dim)

    def forward(self, x, mask = None):
        # Apply masked positional encoding
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device)
        x = self.pos_encoding(x, mask)  # Shape: (batch_size, seq_length, embed_dim)
        
        # Compute queries, keys, and values
        Q = self.q_proj(x)  # Shape: (batch_size, seq_length, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)  # (batch_size, seq_length, seq_length)
        
        # Apply mask to attention scores (set scores to large negative value where mask == 0)
        attention_mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)  # (batch_size, seq_length, seq_length)
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_length, seq_length)
        # Calculate weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, seq_length, embed_dim)
        
        # Final linear projection
        output = self.out_proj(attn_output)  # (batch_size, seq_length, embed_dim)

        # mean pooling
        # output = output = output[:, -1, :]
        output = output.mean(dim=1)

        # normalize
        output = output / output.norm(dim=-1, keepdim=True)
        
        return output


class SingleHeadAttentionWithOutPositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super(SingleHeadAttentionWithOutPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_encoding = MaskedCosinePositionalEncoding(embed_dim)

    def forward(self, x, mask = None):
        # Apply masked positional encoding
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device)
        # x = self.pos_encoding(x, mask)  # Shape: (batch_size, seq_length, embed_dim)
        
        # Compute queries, keys, and values
        Q = self.q_proj(x)  # Shape: (batch_size, seq_length, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)  # (batch_size, seq_length, seq_length)
        
        # Apply mask to attention scores (set scores to large negative value where mask == 0)
        attention_mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)  # (batch_size, seq_length, seq_length)
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_length, seq_length)
        # Calculate weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, seq_length, embed_dim)
        
        # Final linear projection
        output = self.out_proj(attn_output)  # (batch_size, seq_length, embed_dim)

        # mean pooling
        output = output.mean(dim=1)
        # output = output[:, -1, :]

        # normalize
        # output = output / output.norm(dim=-1, keepdim=True)
        
        return output

# # Example usage
# embed_dim = 512
# seq_length = 10
# batch_size = 32

# # Initialize the model
# model = SingleHeadAttentionWithPositionalEncoding(embed_dim)

# # Input tensor of shape (batch_size, seq_length, embed_dim)
# x = torch.randn(batch_size, seq_length, embed_dim)

# # Mask (1 for valid tokens, 0 for padding)
# mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * batch_size, dtype=torch.float32)

# # Run forward pass
# output = model(x, mask)
# print(output.shape)  # Expected shape: (batch_size, seq_length, embed_dim)



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
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

        self.transform_model = TwoLayerMLP(embed_dim)
        
        # # Pooling type: cls, mean, or max
        # assert pooling in ["cls", "mean", "max"], "Pooling must be 'cls', 'mean', or 'max'."
        # self.pooling = pooling

    def forward(self, x, mask, text_array):
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device)
        batch_size, seq_length, embed_dim = x.size()

        # # Apply masked positional encoding
        # x = self.pos_encoding(x, mask)
        
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

        # # Pooling to get a single output
        # if self.pooling == "cls":
        #     output = output[:, 0, :]  # Use the first token’s representation
        # elif self.pooling == "mean":
        #     output = (output * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        # elif self.pooling == "max":
        #     output = (output * mask.unsqueeze(-1)).masked_fill(mask.unsqueeze(-1) == 0, float('-inf')).max(dim=1)[0]

        # mean pooling
        output = output.mean(dim=1)
        # normalize
        output = output / output.norm(dim=-1, keepdim=True)

        # compute score
        output = output - text_array
        score = self.transform_model(output)

        return score


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
        #print(x.shape)
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

        # # Pooling to get a single output
        # if self.pooling == "cls":
        #     output = output[:, 0, :]  # Use the first token’s representation
        # elif self.pooling == "mean":
        #     output = (output * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        # elif self.pooling == "max":
        #     output = (output * mask.unsqueeze(-1)).masked_fill(mask.unsqueeze(-1) == 0, float('-inf')).max(dim=1)[0]

        # mean pooling
        output = output.mean(dim=1)
        # normalize
        output = output / output.norm(dim=-1, keepdim=True)

        score = self.transform_model(output)

        return score


class MultiHeadAttentionSubtractionClass(nn.Module):
    def __init__(self, embed_dim, num_heads, class_num, dropout=0.1):
        super(MultiHeadAttentionSubtractionClass, self).__init__()
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

        self.transform_model = TwoLayerMLPClass(embed_dim, class_num)

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

        # # Pooling to get a single output
        # if self.pooling == "cls":
        #     output = output[:, 0, :]  # Use the first token’s representation
        # elif self.pooling == "mean":
        #     output = (output * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        # elif self.pooling == "max":
        #     output = (output * mask.unsqueeze(-1)).masked_fill(mask.unsqueeze(-1) == 0, float('-inf')).max(dim=1)[0]

        # mean pooling
        output = output.mean(dim=1)
        # normalize
        output = output / output.norm(dim=-1, keepdim=True)

        predicted_class = self.transform_model(output)

        return predicted_class