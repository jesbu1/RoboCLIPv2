import argparse
import tensorflow_datasets as tfds
import pickle
from tqdm import tqdm


'''
1. find out the top k closest language embedding from the dataset. 
2. ask human to download the corresponding video
3. download

'''



def main(args):



    ds = tfds.load(args.dataset_name, data_dir=args.dataset_path, split=args.split)

    num = 0

    text_dict = dict()


    for i, example in tqdm(enumerate(ds)):  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`

        for j, step in enumerate(example["steps"]):
            if args.dataset_name in ["droid_100", "droid"]:
                key1 = step["language_instruction"].numpy().decode()
                key2 = step["language_instruction_2"].numpy().decode()
                key3 = step["language_instruction_3"].numpy().decode()

                if key1 != "":
                    text_dict[key1] = i
                if key2 != "":
                    text_dict[key2] = i
                if key3 != "":
                    text_dict[key3] = i


            else:
            #     # "fractal, bridge": "natural_language_instruction"
                key = step['observation']['natural_language_instruction'].numpy().decode()
                text_dict[key] = i


            if j == 0:
                break
        num = i
    file_name = args.dataset_name + "_text.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(text_dict, file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_model", type=str, default="all-MiniLM-L6-v2", help='language model type')
    parser.add_argument("--dataset_path", type=str, default="/scr/jzhang96/", help='dataset saved location')
    parser.add_argument("--dataset_name", type=str, default="droid_100", choices=["droid_100", "droid", "bridge", "fractal"], help='dataset saved location')
    parser.add_argument("--split", type=str, default='train', help='training set or test set')

    args = parser.parse_args()

    main(args)



