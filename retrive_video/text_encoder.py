import h5py
from sentence_transformers import SentenceTransformer
import argparse
import pickle
from tqdm import tqdm
import os

def main(args):
    pickle_name = args.dataset_name + "_text.pkl"

    language_model = SentenceTransformer(args.lang_model)
    with open(pickle_name, 'rb') as file:
        # Load data from the file using pickle.load()
        loaded_data = pickle.load(file)

    anns = list(loaded_data.keys())
    
    h5_file = args.dataset_name + '_lang_emb.h5'
    file_path = os.path.join(args.save_path, h5_file)
    with h5py.File(file_path, 'w') as file:

        for ann in tqdm(anns):
            if ann == "":
                pass
            else:
                ann_latent = language_model.encode(ann, convert_to_tensor=True).squeeze().cpu().numpy()
                index = loaded_data[ann]

                group = file.create_group(ann)
                group.create_dataset('latent', data=ann_latent)
                group.create_dataset('index', data=index) # video index in the dataset




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_model", type=str, default="all-MiniLM-L6-v2", help='language model type')
    parser.add_argument("--save_path", type=str, default="/scr/jzhang96/", help='h5 saved location')
    # parser.add_argument("--top_k", type=int, default=5, help='show top k close videos')
    parser.add_argument("--dataset_name", type=str, default="droid_100", choices=["droid_100", "droid", "bridge", "fractal"], help='dataset saved location')

    args = parser.parse_args()

    main(args)