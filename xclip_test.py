# from transformers import XCLIPTextModel, XCLIPTextConfig

# # Initializing a XCLIPTextModel with microsoft/xclip-base-patch32 style configuration
# configuration = XCLIPTextConfig()

# # Initializing a XCLIPTextConfig from the microsoft/xclip-base-patch32 style configuration
# model = XCLIPTextModel(configuration)

# # Accessing the model configuration
# configuration = model.config
# print(configuration)
# model.from_pretrained()
from finetune_utils.open_x_loader import TextVideoDataset, EmbeddingsDataset
from finetune_utils.s3d_loss import MILNCELoss


from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import numpy as np
import torch
import av
import argparse
import h5py
import os
import json



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(args):
    np.random.seed(0)
    compute_metrics = MILNCELoss()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
    model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
    training_args = TrainingArguments(
        output_dir = "/scr/jzhang96",
        num_train_epochs = args.num_epochs,
        learning_rate = args.lr,
        per_device_train_batch_size = args.batch_size,
    )


    if args.debug:
        dataset_name = "droid_100_torch_compress_test_train.h5"
    else:
        dataset_name = args.dataset_name + "_torch_compress_train" + ".h5"
    h5_path = os.path.join(args.dataset_path, dataset_name)
    print("h5_path", h5_path)
    h5_file = h5py.File(h5_path, 'r')
    split_file = args.dataset_name + "_torch_split.json"
    split = json.load(open(os.path.join("finetune_utils", split_file), "r"))
    print("Dataset", args.dataset_name, "train", len(split["train"]), "val", len(split["val"]))

    train_dataset = TextVideoDataset(args, h5_file, split, split="train")
    val_dataset = TextVideoDataset(args, h5_file, split, split="val")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("done")



if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")


    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="s3d_finetune", help="Name of the experiment")
    parser.add_argument("--run_group", type=str, default="s3d_finetune", help="Name of the run group")
    parser.add_argument("--finetune", type=str2bool, default=False, help="Whether to finetune the model, if False will train from scratch")
    parser.add_argument("--dataset_name", type=str, default="droid", choices=["droid", "droid_100", "bridge", "fractal"], help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--dataset_path", type=str, default="/scr/jzhang96/", help="Path to the dataset")
    parser.add_argument("--preprocess", type=str2bool, default=True, help="Whether to preprocess the dataset")
    parser.add_argument("--ds_frames", type=int, default=32, help="Number of frames to sample from the video")
    parser.add_argument("--debug", type=str2bool, default=True, help="Debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default="/scr/jzhang96/s3d_finetune", help="Path to save the model")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency to save the model")
    parser.add_argument("--use_amp", type=str2bool, default=True, help="Whether to use automatic mixed precision")

    args = parser.parse_args()
    main(args)