import torch
import torch.nn as nn
import math
import torch.nn.functional as F


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
    
class OneLayerMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(OneLayerMLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim * 2)
        # self.linear2 = torch.nn.Linear(input_dim * 2, input_dim * 2)

    def forward(self, x):
        x = self.linear1(x)
        # x = F.relu(x)
        # x = self.linear2(x)
        return x


class MultiHeadAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, class_num=1, dropout=0.1, enlarge = False):
        super(MultiHeadAttentionModel, self).__init__()
        self.enlarge = False
        if enlarge:
            
            self.enlarge = True
            self.text_enlarge_model = OneLayerMLP(embed_dim)
            self.image_enlarge_model = OneLayerMLP(embed_dim)
            embed_dim = embed_dim * 2
        


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
        if class_num == 1:
            self.transform_model = TwoLayerMLP(embed_dim)
        else:
            self.transform_model = TwoLayerMLPClass(embed_dim, class_num)


    def forward(self, x, mask, text_array):
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device)
        batch_size, seq_length, embed_dim = x.size()

        # # Apply masked positional encoding
        # x = self.pos_encoding(x, mask)
        if self.enlarge:
            
            text_array = self.text_enlarge_model(text_array)
            # x dim is batch_size, seq_length, embed_dim, convert x to batch_size * seq_length, embed_dim
            x = x.view(-1, embed_dim)
            x = self.image_enlarge_model(x)
            embed_dim = embed_dim * 2
            x = x.view(batch_size, seq_length, embed_dim)
            


        
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
        output = output - text_array
        score = self.transform_model(output)

        return score, None


class MultiHeadAttentionSubtraction(nn.Module):
    def __init__(self, embed_dim, num_heads, class_num=1, dropout=0.1):
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

        if class_num == 1:
            self.transform_model = TwoLayerMLP(embed_dim)
        else:
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

        score = self.transform_model(output)

        return score, None


class MultiHeadAttentionConcatenation(nn.Module):
    def __init__(self, embed_dim, num_heads, class_num=1, dropout=0.1):
        super(MultiHeadAttentionConcatenation, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        embed_dim = embed_dim * 2
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

        if class_num == 1:
            self.transform_model = TwoLayerMLP(embed_dim)
        else:
            self.transform_model = TwoLayerMLPClass(embed_dim, class_num)

    def forward(self, x, mask, text_array):
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device)
        batch_size, seq_length, embed_dim = x.size()
        embed_dim *= 2
        # # Apply masked positional encoding
        # x = self.pos_encoding(x, mask)

        text_array = text_array.unsqueeze(1).expand(-1, seq_length, -1)
        
        x = torch.cat([x, text_array], dim=-1)
        
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

        return score, None


class MultiHeadAttentionTwostepModel(nn.Module):
    def __init__(self, embed_dim, num_heads, class_num=1, dropout=0.1):
        super(MultiHeadAttentionTwostepModel, self).__init__()
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

        self.binary_classifier = TwoLayerMLPClass(embed_dim, 2)

        if class_num == 1:
            self.transform_model = TwoLayerMLP(embed_dim)
        else:
            self.transform_model = TwoLayerMLPClass(embed_dim, class_num)


    def forward(self, x, mask, text_array):
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device)
        batch_size, seq_length, embed_dim = x.size()

        # # Apply masked positional encoding
        # x = self.pos_encoding(x, mask)
        # if self.enlarge:
            
        #     text_array = self.text_enlarge_model(text_array)
        #     # x dim is batch_size, seq_length, embed_dim, convert x to batch_size * seq_length, embed_dim
        #     x = x.view(-1, embed_dim)
        #     x = self.image_enlarge_model(x)
        #     embed_dim = embed_dim * 2
        #     x = x.view(batch_size, seq_length, embed_dim)
            


        
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
        output = output - text_array
        score = self.transform_model(output)
        class_score = self.binary_classifier(output)

        return score, class_score
