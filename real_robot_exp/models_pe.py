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


class RewardTwoStepNewPositionEmbeddingPredictor(nn.Module):
    def __init__(self, input_dim, args, class_num):
        super(RewardTwoStepNewPositionEmbeddingPredictor, self).__init__()
        self.args = args
        decoder_num = args.decoder_num
        self.transformer_decoder = nn.ModuleList([DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm) for _ in range(decoder_num)])
        # self.transformer_decoder = DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm)
        if class_num == 1:
            self.classifier = nn.Linear(input_dim, 1)
        else:
            self.classifier = nn.Linear(input_dim, class_num)

        # if args.cat_text:
        #     self.twostep_classifier = TwoLayerMLPClass(input_dim * 2, 2)
        # else:
        self.twostep_classifier = nn.Linear(input_dim, 1)

        self.class_num = class_num
        self.positional_encoding = args.positional_encoding
        if args.positional_encoding:
            self.position_embedding = self._get_cosine_positional_encoding(args.max_length, input_dim)
        if args.learner_parameter:
            self.text_learner_parameter = nn.Parameter(torch.randn(1, 1024))
            self.video_learner_parameter = nn.Parameter(torch.randn(1, input_dim))
        self.text_projector = nn.Linear(1024, input_dim)


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
        text_array = self.text_projector(text_array)
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
            x = torch.sigmoid(x)
            # x = torch.clamp(x, 0, 1)
        x = x.view(batch_size, seq_len, -1)
        two_step_label = two_step_label.view(batch_size, seq_len, -1)
        two_step_label = torch.sigmoid(two_step_label)
        return x, two_step_label



class RewardOneStepNewPositionEmbeddingPredictor(nn.Module):
    def __init__(self, input_dim, args, class_num):
        super(RewardOneStepNewPositionEmbeddingPredictor, self).__init__()
        self.args = args
        decoder_num = args.decoder_num
        self.transformer_decoder = nn.ModuleList([DecoderOnlyBlock(input_dim, args.attention_heads, input_dim, args.layer_norm) for _ in range(decoder_num)])

        if class_num == 1:
            self.classifier = nn.Linear(input_dim, 1)
        else:
            self.classifier = nn.Linear(input_dim, class_num)

        self.class_num = class_num

        self.positional_encoding = args.positional_encoding
        if args.positional_encoding:
            self.position_embedding = self._get_cosine_positional_encoding(args.max_length, input_dim)
        if args.learner_parameter:
            self.text_learner_parameter = nn.Parameter(torch.randn(1, 1024))
            self.video_learner_parameter = nn.Parameter(torch.randn(1, input_dim))

        self.text_projector = nn.Linear(1024, input_dim)


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
        text_array = self.text_projector(text_array)
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


        x = self.classifier(x)
        if self.class_num == 1:
            x = torch.sigmoid(x)
            x = torch.clamp(x, 0, 1)
        x = x.view(batch_size, seq_len, -1)
        two_step_label = None
        return x, two_step_label




class ClassProgressTransformer(nn.Module):
    def __init__(self, args, video_dim=768, text_dim=384, hidden_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.args = args
        
        # Project video and text to common dimension
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Position embeddings for video sequence
        if self.args.positional_encoding:
            self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))  # 32 is max_length
            if self.args.last_frame_pe:
                self.last_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))
        
        # Class token embedding
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # use a decoder-style transformer
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=hidden_dim,
        #     nhead=num_heads,
        #     dim_feedforward=hidden_dim * 4,
        #     dropout=0.1,
        #     batch_first=True
        # )
        # self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        
        # Progress prediction head (applied to each frame)
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Classification head (applied to class token)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.attention_mask = nn.Transformer.generate_square_subsequent_mask(18).to('cuda')
    
    def forward(self, video_frames, text_embed, attention_mask=None):
        batch_size = video_frames.shape[0]
        seq_len = video_frames.shape[1]
        
        # Project inputs to common dimension
        video_embed = self.video_proj(video_frames)  # [batch_size, seq_len, hidden_dim]
        text_embed = self.text_proj(text_embed).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Add positional embeddings to video]
        if self.args.positional_encoding:
            video_embed[:,0] += self.first_pos_embed
            if self.args.last_frame_pe:
                video_embed[:,-1] += self.last_pos_embed
        
        # Expand class token for batch
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        
        # Combine sequence: [class_token, video_frames, text]
        # sequence = torch.cat([class_tokens, video_embed, text_embed], dim=1)
        sequence = torch.cat([text_embed, video_embed, class_tokens], dim=1)
        
        # Create attention mask if needed
        if attention_mask is not None:
            # Add mask positions for class token and text token
            extended_mask = torch.ones((batch_size, 2), device=attention_mask.device)  # class token + text token
            attention_mask = torch.cat([extended_mask, attention_mask], dim=1)
        
        # Pass through transformer
        

        transformed = self.transformer(sequence, is_causal=True, mask = self.attention_mask)
        
        # Get class prediction from class token
        # class_pred = self.classification_head(transformed[:, 0])  # Use class token
        class_pred = self.classification_head(transformed[:, -1])  # Use class token
        
        # Get progress predictions for each frame
        progress_preds = self.progress_head(transformed[:, 1:-1])  # Exclude class token and text token
        
        return progress_preds, class_pred