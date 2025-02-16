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


class TextTransformerEncoder(nn.Module):
    def __init__(self, emb_size, max_t=100, num_heads=8, num_layers=1, ff_dim=1024):
        super().__init__()
        self.emb_size = emb_size
        self.max_t = max_t  # Maximum expected sequence length
        self.project = nn.Linear(emb_size, ff_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=ff_dim, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=ff_dim,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Learnable Query Token
        self.query_token = nn.Parameter(torch.randn(1, 1, ff_dim))  # (1, 1, emb_size)

        # Precompute positional encodings
        self.positional_encoding = self._generate_positional_encoding(max_t, emb_size)  # (max_t, emb_size)

    def _generate_positional_encoding(self, t, emb_size):
        """ Generate sinusoidal positional encodings (cosine & sine). """
        pos = torch.arange(t, dtype=torch.float).unsqueeze(1)  # Shape: (t, 1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))  # Shape: (emb_size/2)

        pe = torch.zeros(t, emb_size)
        pe[:, 0::2] = torch.sin(pos * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(pos * div_term)  # Apply cosine to odd indices

        return pe.unsqueeze(0)  # Shape: (1, t, emb_size) for broadcasting

    def forward(self, x, mask):
        """
        x: (bs, t, emb_size) - Input sequence embeddings
        mask: (bs, t) - Boolean mask, True for padding, False for valid tokens
        """
        bs, t, emb_size = x.shape

        # Add positional encoding (slice to match sequence length)
        x = x + self.positional_encoding[:, :t, :].to(x.device)  # (bs, t, emb_size)

        # Expand query token for the batch
        query = self.query_token.expand(bs, -1, -1)  # Shape: (bs, 1, emb_size)

        # Pass through the linear projection

        x = self.project(x.view(bs * t, emb_size)).view(bs, t, -1)  # (bs, t, ff_dim)

        # Pass through the transformer decoder with mask
        output = self.decoder(query, x, memory_key_padding_mask=mask)  # (bs, 1, emb_size)

        return output.squeeze(1)  # (bs, emb_size)
