import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class ClassProgressTransformer(nn.Module):
    def __init__(self, args, video_dim=768, text_dim=384, hidden_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.args = args
        
        # Project video and text to common dimension
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Position embeddings for video sequence
        # if self.args.positional_encoding:
        self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))  # 32 is max_length
            # if self.args.last_frame_pe:
            #     self.last_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))
        
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
        # seq_len = video_frames.shape[1]
        
        # Project inputs to common dimension
        video_embed = self.video_proj(video_frames)  # [batch_size, seq_len, hidden_dim]
        text_embed = self.text_proj(text_embed).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Add positional embeddings to video]
        # if self.args.positional_encoding:
        video_embed[:,0] += self.first_pos_embed
            # if self.args.last_frame_pe:
            #     video_embed[:,-1] += self.last_pos_embed
        
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