import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
# from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
# from apex import amp
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config



def main(config): 
    # train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    import pdb ; pdb.set_trace()
    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=None,
                        )
    model = model.cuda()

    import pdb ; pdb.set_trace()

    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES, 
                                       smoothing=config.AUG.LABEL_SMOOTH, 
                                       mixup_alpha=config.AUG.MIXUP, 
                                       cutmix_alpha=config.AUG.CUTMIX, 
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)

    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)


    text_labels = generate_text(train_data)
    
    if config.TEST.ONLY_TEST:
        acc1 = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn)

        acc1 = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)

    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    acc1 = validate(val_loader, text_labels, model, config)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")





if __name__ == '__main__':
    # prepare config
    args, config = parse_option()
    # init_distributed

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier(device_ids=[args.local_rank])

    # seed = config.SEED + dist.get_rank()
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    # logger
    # logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    # logger.info(f"working dir: {config.OUTPUT}")

    main(config)