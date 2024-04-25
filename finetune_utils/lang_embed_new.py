import h5py
from sentence_transformers import SentenceTransformer
import argparse
import pickle
from tqdm import tqdm
import os

def main(args):

    file_name = args.dataset_name + "_torch_train.h5"
    file_name = os.path.join(args.save_path, file_name)
    text_file_name = args.dataset_name + "_embeddings_new.h5"
    text_file_name = os.path.join(args.save_path, text_file_name)
    language_model = SentenceTransformer(args.lang_model)

    with h5py.File(file_name, 'r') as read_file:
        with h5py.File(text_file_name, 'a') as write_file:
            for key in tqdm(read_file.keys()):
                if args.dataset_name in ["bridge", "fractal"]:
                    ann = read_file[key]["ann"][()].decode()
                    if ann != "":
                        ann_latent = language_model.encode(key, convert_to_tensor=True).squeeze().cpu().numpy()

                        write_file_keys = write_file.keys()

                        if ann not in write_file_keys:
                            group = write_file.create_group(ann)
                            group.create_dataset('latent', data=ann_latent)
                            key_name = key
                            group.create_dataset('keys', data=key_name)
                        
                        else:
                            group = write_file[ann]
                            # import pdb; pdb.set_trace()
                            keys = [group["keys"][()].decode()]
                            keys.append(key)
                            names = list(set(keys))
                            # use _ connect all element in the list
                            result = "_".join(names)
                            write_file[ann]["keys"][()] = result
                            # write_file[ann] = result


                else:
                    ann_1 = read_file[key]["ann_1"][()].decode()
                    ann_2 = read_file[key]["ann_2"][()].decode()
                    ann_3 = read_file[key]["ann_3"][()].decode()
                    ann_1_latent = language_model.encode(ann_1, convert_to_tensor=True).squeeze().cpu().numpy()
                    ann_2_latent = language_model.encode(ann_2, convert_to_tensor=True).squeeze().cpu().numpy()
                    ann_3_latent = language_model.encode(ann_3, convert_to_tensor=True).squeeze().cpu().numpy()

                    ann_pair = [(ann_1, ann_1_latent), (ann_2, ann_2_latent), (ann_3, ann_3_latent)]

                    for ann, ann_latent in ann_pair:
                        if ann != "":
                            write_file_keys = write_file.keys()
                            if ann not in write_file_keys:
                                group = write_file.create_group(ann)
                                group.create_dataset('latent', data=ann_latent)
                                key_name = key
                                group.create_dataset('keys', data=key_name)
                            else:
                                group = write_file[ann]
                                keys = [group["keys"][()].decode()]
                                keys.append(key)
                                names = list(set(keys))
                                # use _ connect all element in the list
                                result = "_".join(names)
                                write_file[ann]["keys"][()] = result
                                # write_file[ann] = result






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_model", type=str, default="all-MiniLM-L6-v2", help='language model type')
    parser.add_argument("--save_path", type=str, default="/scr/jzhang96/", help='h5 saved location')
    parser.add_argument("--dataset_name", type=str, default="droid_100", choices=["droid_100", "droid", "bridge", "fractal"], help='dataset saved location')

    args = parser.parse_args()

    main(args)