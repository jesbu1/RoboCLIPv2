import argparse
import torch
from models import MultiHeadAttentionModel, MultiHeadAttentionSubtraction, MultiHeadAttentionConcatenation

def load_reward_model(state_dict_path):

    state_dict_path = state_dict_path
    state_dict = torch.load(state_dict_path)
    saved_args = state_dict.get('args', {})

    args = saved_args

    print(args)
    ## TODO: PCA options
    embedding_dim = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cat_embedding:
        if args.catagorical_progress:
            if args.sample_neg:
                num_bins = args.catagorical_progress_bins + 1
            else:
                num_bins = args.catagorical_progress_bins
            self_attention_model = MultiHeadAttentionConcatenation(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=num_bins).to(device)
        else:
            self_attention_model = MultiHeadAttentionConcatenation(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=1).to(device)
    else:
        if args.subtract_before:
            if args.catagorical_progress:
                if args.sample_neg:
                    num_bins = args.catagorical_progress_bins + 1
                else:
                    num_bins = args.catagorical_progress_bins
                self_attention_model = MultiHeadAttentionSubtraction(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=num_bins).to(device)
            else:
                self_attention_model = MultiHeadAttentionSubtraction(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=1).to(device)
            
        else:
            if args.catagorical_progress:
                if args.sample_neg:
                    num_bins = args.catagorical_progress_bins + 1
                else:
                    num_bins = args.catagorical_progress_bins
                self_attention_model = MultiHeadAttentionModel(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=num_bins).to(device)
            else:
                self_attention_model = MultiHeadAttentionModel(embedding_dim, num_heads = args.attention_heads, dropout = args.dropout, class_num=1).to(device)


    self_attention_model.load_state_dict(state_dict['model'])
    self_attention_model.eval()


    print("Model loaded")
    return self_attention_model