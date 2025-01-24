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
        # x = F.tanh(x)
        x = torch.sigmoid(x)
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

# class TwoLayerMLP(torch.nn.Module):
#     def __init__(self, input_dim):
#         super(TwoLayerMLP, self).__init__()
#         self.linear1 = torch.nn.Linear(input_dim, 1)
#         # self.linear2 = torch.nn.Linear(input_dim // 2, 1)

#     def forward(self, x):
#         # x = F.relu(self.linear1(x))
#         # x = self.linear2(x)
#         # x = F.tanh(x)
#         x = self.linear1(x)
#         x = F.sigmoid(x)
#         x = torch.clamp(x, 0, 1)
#         return x

# class TwoLayerMLPClass(torch.nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(TwoLayerMLPClass, self).__init__()
#         self.linear1 = torch.nn.Linear(input_dim, num_classes)
#         # self.linear2 = torch.nn.Linear(input_dim // 2, num_classes)

#     def forward(self, x):
#         # x = F.relu(self.linear1(x))
#         x = self.linear1(x)
#         return x


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
            if args.cat_text:
                self.classifier = TwoLayerMLP(input_dim * 2)
            else:
                self.classifier = TwoLayerMLP(input_dim)
        else:
            if args.cat_text:
                self.classifier = TwoLayerMLPClass(input_dim * 2, class_num)
            else:
                self.classifier = TwoLayerMLPClass(input_dim, class_num)
        self.class_num = class_num
        self.positional_encoding = args.positional_encoding
        if args.positional_encoding:
            self.position_embedding = self._get_cosine_positional_encoding(50, input_dim)


    def _get_cosine_positional_encoding(self, max_seq_len, embed_dim):
        """
        Generate a static positional encoding matrix using sine and cosine functions.
        """
        position = torch.arange(max_seq_len).unsqueeze(1)  # Shape: [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe /= 100 # reduce the scale of positional encoding otherwise it will dominate the input embeddings
        return pe.unsqueeze(0)


    def forward(self, x, triangular_mask, text_array, mask = None):
        batch_size, seq_len, _ = x.size()
        if self.positional_encoding:
            positional_embedding = self.position_embedding[:, :seq_len, :].to(x.device)
            x = x + positional_embedding

        x = self.transformer_decoder(x, triangular_mask)
        
        x = x.view(batch_size * seq_len, -1)
        text_array = text_array.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1)
        if mask is not None:
            mask = mask.view(batch_size * seq_len).bool()
            x = x[mask]
            text_array = text_array[mask]

        if self.args.cat_text:
            x = torch.cat([x, text_array], dim=1)
        else:
            x = x - text_array
        x = self.classifier(x)
        if self.class_num == 1:
            x = torch.clamp(x, 0, 1)
        # x = x.view(batch_size, seq_len, -1)

        return x, None


# class RewardTwoStepPredictor(nn.Module):
#     def __init__(self, input_dim, args, class_num):
#         super(RewardTwoStepPredictor, self).__init__()
#         self.args = args
#         self.transformer_decoder = DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm)
#         if class_num == 1:
#             self.classifier = TwoLayerMLP(input_dim)
#         else:
#             self.classifier = TwoLayerMLPClass(input_dim, class_num)
#         self.twostep_classifier = TwoLayerMLPClass(input_dim, 2)

#     def forward(self, x, triangular_mask, text_array, mask):
#         x = self.transformer_decoder(x, triangular_mask)
#         batch_size, seq_len, _ = x.size()
#         x = x.view(batch_size * seq_len, -1)
#         text_array = text_array.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1)
#         mask = mask.view(batch_size * seq_len).bool()
#         x = x[mask]
#         text_array = text_array[mask]
#         x = x - text_array
#         two_step_label = self.twostep_classifier(x)
#         progress = self.classifier(x)

#         return progress, two_step_label


class RewardTwoStepPredictor(nn.Module):
    def __init__(self, input_dim, args, class_num):
        super(RewardTwoStepPredictor, self).__init__()
        self.args = args
        decoder_num = args.decoder_num
        self.transformer_decoder = nn.ModuleList([DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm) for _ in range(decoder_num)])
        # self.transformer_decoder = DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm)
        if class_num == 1:
            if args.cat_text:
                self.classifier = TwoLayerMLP(input_dim * 2)
            else:
                self.classifier = TwoLayerMLP(input_dim)
        else:
            if args.cat_text:
                self.classifier = TwoLayerMLPClass(input_dim * 2, class_num)
            else:
                self.classifier = TwoLayerMLPClass(input_dim, class_num)

        if args.cat_text:
            self.twostep_classifier = TwoLayerMLPClass(input_dim * 2, 2)
        else:
            self.twostep_classifier = TwoLayerMLPClass(input_dim, 2)

        self.class_num = class_num
        self.positional_encoding = args.positional_encoding
        if args.positional_encoding:
            self.position_embedding = self._get_cosine_positional_encoding(args.max_length, input_dim)


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


    def forward(self, x, triangular_mask, text_array, mask = None):
        batch_size, seq_len, _ = x.size()
        if self.positional_encoding:
            positional_embedding = self.position_embedding[:, :seq_len, :].to(x.device)
            if self.args.first_frame_embedding:
                x[:, 0] = x[:, 0] + positional_embedding[:, 0]
            else:
                x = x + positional_embedding

        for decoder in self.transformer_decoder:
            x = decoder(x, triangular_mask)
        # x = self.transformer_decoder(x, triangular_mask)
        
        x = x.view(batch_size * seq_len, -1)
        text_array = text_array.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1)
        if mask is not None:
            mask = mask.view(batch_size * seq_len).bool()
            x = x[mask]
            text_array = text_array[mask]

        if self.args.cat_text:
            x = torch.cat([x, text_array], dim=1)
        else:
            x = x - text_array

        two_step_label = self.twostep_classifier(x)
        x = self.classifier(x)
        if self.class_num == 1:
            x = torch.clamp(x, 0, 1)
        x = x.view(batch_size, seq_len, -1)
        two_step_label = two_step_label.view(batch_size, seq_len, -1)
        return x, two_step_label


class RewardTwoStepNewPositionEmbeddingPredictor(nn.Module):
    def __init__(self, input_dim, args, class_num):
        super(RewardTwoStepNewPositionEmbeddingPredictor, self).__init__()
        self.args = args
        decoder_num = args.decoder_num
        self.transformer_decoder = nn.ModuleList([DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm) for _ in range(decoder_num)])
        # self.transformer_decoder = DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm)
        if class_num == 1:
            self.classifier = TwoLayerMLP(input_dim)
        else:
            self.classifier = TwoLayerMLPClass(input_dim, class_num)

        # if args.cat_text:
        #     self.twostep_classifier = TwoLayerMLPClass(input_dim * 2, 2)
        # else:
        self.twostep_classifier = TwoLayerMLPClass(input_dim, 2)

        self.class_num = class_num
        self.positional_encoding = args.positional_encoding
        if args.positional_encoding:
            self.position_embedding = self._get_cosine_positional_encoding(args.max_length, input_dim)
        if args.learner_parameter:
            self.text_learner_parameter = nn.Parameter(torch.randn(1, input_dim))
            self.video_learner_parameter = nn.Parameter(torch.randn(1, input_dim))


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


    def forward(self, x, triangular_mask, text_array, mask = None):
        batch_size, seq_len, _ = x.size()
        if self.args.learner_parameter:
            text_array = text_array + self.text_learner_parameter
            x = x + self.video_learner_parameter.repeat(batch_size, seq_len, 1)
        if self.positional_encoding:
            positional_embedding = self.position_embedding[:, :seq_len, :].to(x.device)
            if self.args.first_frame_embedding:
                x[:, 0] = x[:, 0] + positional_embedding[:, 0]
            else:
                x = x + positional_embedding

        # concatenate text in front of the input
        text_array = text_array.unsqueeze(1)
        x = torch.cat([text_array, x], dim=1)

        for decoder in self.transformer_decoder:
            x = decoder(x, triangular_mask)
        # x = self.transformer_decoder(x, triangular_mask)

        # only take video embeddings
        x = x[:, 1:]
        x = x.contiguous().view(batch_size * seq_len, -1)
        if mask is not None:
            mask = mask.view(batch_size * seq_len).bool()
            x = x[mask]
            text_array = text_array[mask]


        two_step_label = self.twostep_classifier(x)
        x = self.classifier(x)
        if self.class_num == 1:
            x = torch.clamp(x, 0, 1)
        x = x.view(batch_size, seq_len, -1)
        two_step_label = two_step_label.view(batch_size, seq_len, -1)
        return x, two_step_label



class RewardTwoStepLangTokenPositionEmbeddingPredictor(nn.Module):
    def __init__(self, input_dim, args, class_num):
        super(RewardTwoStepLangTokenPositionEmbeddingPredictor, self).__init__()
        self.args = args
        decoder_num = args.decoder_num
        self.transformer_decoder = nn.ModuleList([DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm) for _ in range(decoder_num)])
        # self.transformer_decoder = DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm)
        if class_num == 1:
            self.classifier = TwoLayerMLP(input_dim)
        else:
            self.classifier = TwoLayerMLPClass(input_dim, class_num)

        # if args.cat_text:
        #     self.twostep_classifier = TwoLayerMLPClass(input_dim * 2, 2)
        # else:
        self.twostep_classifier = TwoLayerMLPClass(input_dim, 2)

        self.class_num = class_num
        self.positional_encoding = args.positional_encoding
        if args.positional_encoding:
            self.position_embedding = self._get_cosine_positional_encoding(args.max_length, input_dim)
        if args.learner_parameter:
            self.text_learner_parameter = nn.Parameter(torch.randn(1, input_dim))
            self.video_learner_parameter = nn.Parameter(torch.randn(1, input_dim))


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


    def forward(self, x, triangular_mask, text_array, mask = None):
        # text array with be different length token embeddings
        batch_size, seq_len, _ = x.size()
        if self.args.learner_parameter:
            text_array = text_array + self.text_learner_parameter
            x = x + self.video_learner_parameter.repeat(batch_size, seq_len, 1)
        if self.positional_encoding:
            positional_embedding = self.position_embedding[:, :seq_len, :].to(x.device)
            if self.args.first_frame_embedding:
                x[:, 0] = x[:, 0] + positional_embedding[:, 0]
            else:
                x = x + positional_embedding

        # concatenate text in front of the input
        text_array = text_array.unsqueeze(1)
        x = torch.cat([text_array, x], dim=1)

        for decoder in self.transformer_decoder:
            x = decoder(x, triangular_mask)
        # x = self.transformer_decoder(x, triangular_mask)

        # only take video embeddings
        x = x[:, 1:]
        x = x.contiguous().view(batch_size * seq_len, -1)
        if mask is not None:
            mask = mask.view(batch_size * seq_len).bool()
            x = x[mask]
            text_array = text_array[mask]


        two_step_label = self.twostep_classifier(x)
        x = self.classifier(x)
        if self.class_num == 1:
            x = torch.clamp(x, 0, 1)
        x = x.view(batch_size, seq_len, -1)
        two_step_label = two_step_label.view(batch_size, seq_len, -1)
        return x, two_step_label

