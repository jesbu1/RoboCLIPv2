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


class CosPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=200):
        super(CosPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.positional_encoding = self._generate_positional_encoding()

    def _generate_positional_encoding(self):
        pe = torch.zeros(self.max_len, self.embed_dim)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(math.log(10000.0) / self.embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pe = self.positional_encoding[:seq_len].unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)
        x = x + pe
        return x




class DecoderOnlyBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, layer_norm):
        super(DecoderOnlyBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.depth = embed_dim // num_heads

        # Layers for Multi-Head Attention
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)


        # Layers for Feedforward Network
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

        for module in [self.query, self.key, self.value, self.fc_out, self.fc1, self.fc2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        self.layer_norm = layer_norm
        # Layer Normalization
        if layer_norm:
            self.layernorm1 = nn.LayerNorm(embed_dim)
            self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        batch_size = x.size(0)

        # Multi-Head Attention
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.depth)

        if mask is not None:
            # scores = scores.masked_fill(mask == 0, float('-inf'))
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)

        attn_output = torch.matmul(attention, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        attn_output = self.fc_out(attn_output)


        # Residual connection and layer normalization
        if self.layer_norm:
            x = self.layernorm1(x + attn_output)
        else:
            x = x + attn_output
        # Feedforward Network
        ffn_output = self.fc2(F.gelu(self.fc1(x)))

        # Residual connection and layer normalization
        if self.layer_norm:
            x = self.layernorm2(x + ffn_output)
        else:
            x = x + ffn_output

        return x

class RewardPredictor(nn.Module):
    def __init__(self, input_dim, args, class_num):
        super(RewardPredictor, self).__init__()
        self.args = args
        self.transformer_decoder = DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm)
        if class_num == 1:
            self.classifier = TwoLayerMLP(input_dim)
        else:
            self.classifier = TwoLayerMLPClass(input_dim, class_num)
        self.class_num = class_num
        self.positional_encoding = args.positional_encoding
        if args.positional_encoding:
            self.position_embedding = self._get_cosine_positional_encoding(200, input_dim)


    def _get_cosine_positional_encoding(self, max_seq_len, embed_dim):
        """
        Generate a static positional encoding matrix using sine and cosine functions.
        """
        position = torch.arange(max_seq_len).unsqueeze(1)  # Shape: [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe /= 100
        return pe.unsqueeze(0)


    def forward(self, x, triangular_mask, text_array, mask):
        batch_size, seq_len, _ = x.size()
        if self.positional_encoding:
            positional_embedding = self.position_embedding[:, seq_len, :].to(x.device)
            x = x + positional_embedding

        x = self.transformer_decoder(x, triangular_mask)
        
        x = x.view(batch_size * seq_len, -1)
        text_array = text_array.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1)
        mask = mask.view(batch_size * seq_len).bool()
        x = x[mask]
        text_array = text_array[mask]

        x = x - text_array
        x = self.classifier(x)
        if self.class_num == 1:
            x = torch.clamp(x, 0, 1)

        return x, None


class RewardTwoStepPredictor(nn.Module):
    def __init__(self, input_dim, args, class_num):
        super(RewardTwoStepPredictor, self).__init__()
        self.args = args
        self.transformer_decoder = DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm)
        if class_num == 1:
            self.classifier = TwoLayerMLP(input_dim)
        else:
            self.classifier = TwoLayerMLPClass(input_dim, class_num)
        self.twostep_classifier = TwoLayerMLPClass(input_dim, 2)

    def forward(self, x, triangular_mask, text_array, mask):
        x = self.transformer_decoder(x, triangular_mask)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)
        text_array = text_array.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1)
        mask = mask.view(batch_size * seq_len).bool()
        x = x[mask]
        text_array = text_array[mask]
        x = x - text_array
        two_step_label = self.twostep_classifier(x)
        progress = self.classifier(x)

        return progress, two_step_label


