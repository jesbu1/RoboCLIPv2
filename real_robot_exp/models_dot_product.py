import torch
import torch.nn as nn
import math
import torch.nn.functional as F



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
        normed_x = self.layernorm1(x) if self.layer_norm else x
        Q = (
            self.query(normed_x)
            .view(batch_size, -1, self.num_heads, self.depth)
            .transpose(1, 2)
        )
        K = (
            self.key(normed_x)
            .view(batch_size, -1, self.num_heads, self.depth)
            .transpose(1, 2)
        )
        V = (
            self.value(normed_x)
            .view(batch_size, -1, self.num_heads, self.depth)
            .transpose(1, 2)
        )

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
            x = self.layernorm2(x + attn_output)
        else:
            x = x + attn_output
        # Feedforward Network
        ffn_output = self.fc2(F.gelu(self.fc1(x)))

        # Residual connection again
        x = x + ffn_output

        return x




class VideoTransformerEncoder(nn.Module):
    def __init__(self, input_dim, args):
        super(VideoTransformerEncoder, self).__init__()
        self.args = args
        decoder_num = args.decoder_num
        self.transformer_decoder = nn.ModuleList([DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm) for _ in range(decoder_num)])


        self.positional_encoding = args.positional_encoding
        if args.positional_encoding:
            self.position_embedding = self._get_cosine_positional_encoding(args.max_length, input_dim)
        if args.learner_parameter:
            self.video_learner_parameter = nn.Parameter(torch.randn(1, input_dim))


    def _get_cosine_positional_encoding(self, max_seq_len, embed_dim):
        """
        Generate a static positional encoding matrix using sine and cosine functions.
        """
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe /= 10
        return pe.unsqueeze(0)
    
    def forward(self, x, triangular_mask):
        batch_size, seq_len, _ = x.size()
        if self.args.learner_parameter:
            x = x + self.video_learner_parameter.repeat(batch_size, seq_len, 1)
        if self.positional_encoding:
            positional_embedding = self.position_embedding[:, :seq_len, :].to(x.device)
            x = x + positional_embedding

        for decoder in self.transformer_decoder:
            x = decoder(x, triangular_mask)

        return x

