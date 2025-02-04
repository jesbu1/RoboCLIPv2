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
            # scores = scores.masked_fill(mask == 0, -1e9)
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




class RewardTwoStepLangTokenPositionEmbeddingPredictor(nn.Module):
    def __init__(self, input_dim, args, class_num):
        super(RewardTwoStepLangTokenPositionEmbeddingPredictor, self).__init__()
        self.args = args
        decoder_num = args.decoder_num
        self.transformer_decoder = nn.ModuleList([DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm) for _ in range(decoder_num)])

        if class_num == 1:
            self.classifier = nn.Linear(input_dim, 1)
        else:
            self.classifier = nn.Linear(input_dim, class_num)

        self.twostep_classifier = nn.Linear(input_dim, 1)

        self.class_num = class_num


    def forward(self, x, triangular_mask, mask):
        # text array with be different length token embeddings
        batch_size, seq_len, embedding_shape = x.size()

        for decoder in self.transformer_decoder:
            x = decoder(x, triangular_mask)

        x = x.contiguous().view(batch_size * seq_len, -1)
        mask = mask.view(batch_size * seq_len).bool()
        x = x[mask]
        # x = x[mask]
        # x = x.contiguous().view(batch_size, self.args.max_length, -1)
        if self.args.two_step_training:
            two_step_label = self.twostep_classifier(x)
            two_step_label = two_step_label.view(batch_size, self.args.max_length, -1)

        progress = self.classifier(x)
        progress = progress.view(batch_size, self.args.max_length, -1)

        return progress, two_step_label


class RewardOneStepLangTokenPositionEmbeddingPredictor(nn.Module):
    def __init__(self, input_dim, args, class_num):
        super(RewardOneStepLangTokenPositionEmbeddingPredictor, self).__init__()
        self.args = args
        decoder_num = args.decoder_num
        self.transformer_decoder = nn.ModuleList([DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm) for _ in range(decoder_num)])

        if class_num == 1:
            self.classifier = TwoLayerMLP(input_dim)
        else:
            self.classifier = TwoLayerMLPClass(input_dim, class_num)

        self.class_num = class_num


    def _get_cosine_positional_encoding(self, max_seq_len, embed_dim):
        """
        Generate a static positional encoding matrix using sine and cosine functions.
        """
        position = torch.arange(max_seq_len).unsqueeze(1)  # Shape: [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe /= 10 # reduce the scale of positional encoding otherwise it will dominate the input embeddings
        return pe.unsqueeze(0)


    def forward(self, x, triangular_mask, video_start, mask = None):
        # text array with be different length token embeddings
        batch_size, seq_len, _ = x.size()
        if self.positional_encoding:
            positional_embedding = self.position_embedding[:, :seq_len, :].to(x.device)
            if self.args.first_frame_embedding:
                x[:, 0] = x[:, 0] + positional_embedding[:, 0]
                x[:,video_start] = x[:,video_start] + positional_embedding[:,-1]
            else:
                x = x + positional_embedding

        for decoder in self.transformer_decoder:
            x = decoder(x, triangular_mask)
        # x = self.transformer_decoder(x, triangular_mask)

        # only take video embeddings
        x = x.contiguous().view(batch_size * seq_len, -1)
        if mask is not None:
            mask = mask.view(batch_size * seq_len).bool()
            x = x[mask]

        x = self.classifier(x)
        if self.class_num == 1:
            x = torch.clamp(x, 0, 1)
        x = x.view(batch_size, self.args.max_length, -1)
        return x, None
