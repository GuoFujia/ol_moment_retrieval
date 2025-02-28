import os
import time
import pprint
import random
import argparse
import copy
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from easydict import EasyDict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.dataset2 import StartEndDataset, start_end_collate, prepare_batch_inputs
from training.myEvaluate import eval_epoch, setup_model
from lighthouse.common.utils.myUtils import AverageMeter,write_log,save_checkpoint,rename_latest_to_best,count_parameters,ModelEMA
from torch.utils.data import DataLoader
from easydict import EasyDict
from collections import defaultdict
from tqdm import tqdm, trange
# from lighthouse.common.utils.myUtils import BaseOptions
from training.config import BaseOptions

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i):
    batch_input_fn = prepare_batch_inputs
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        model_inputs, targets = batch_input_fn(batch[1], opt.device)
        
        outputs = model(**model_inputs, targets=targets) if opt.model_name == 'cg_detr' else model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        
        optimizer.zero_grad()
        losses.backward()
            
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * criterion.weight_dict[k] if k in criterion.weight_dict else float(v))

    write_log(opt, epoch_i, loss_meters)


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    collate_fn = start_end_collate
    save_submission_filename = "latest_{}_val_preds.jsonl".format(opt.dset_name)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )

    if opt.model_ema:
        logger.info("Using model EMA...")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    prev_best_score = 0
    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i)
        lr_scheduler.step()

        if opt.model_ema:
            model_ema.update(model)

        if (epoch_i + 1) % opt.eval_epoch_interval == 0:
            with torch.no_grad():
                if opt.model_ema:
                    metrics, eval_loss_meters, latest_file_paths = \
                        eval_epoch(epoch_i, model_ema.module, val_dataset, opt, save_submission_filename, criterion)
                else:
                    metrics, eval_loss_meters, latest_file_paths = \
                        eval_epoch(epoch_i, model, val_dataset, opt, save_submission_filename, criterion)

            write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode='val')            
            logger.info("metrics {}".format(pprint.pformat(metrics["brief"], indent=4)))
            
            stop_score = metrics["brief"]["MR-full-mAP"]

            if stop_score > prev_best_score:
                prev_best_score = stop_score
                save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
                logger.info("The checkpoint file has been updated.")
                rename_latest_to_best(latest_file_paths)


def main(opt, resume=None, domain=None):
    logger.info("Setup config, data and model...")
    set_seed(opt.seed)

    # dataset & data loader
    dataset_config = EasyDict(
        dset_name=opt.dset_name,
        domain=domain,
        data_path=opt.train_path,
        ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs,
        a_feat_dirs=opt.a_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        a_feat_types=opt.a_feat_types,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
    )

    train_dataset = StartEndDataset(**dataset_config)    
    copied_eval_config = copy.deepcopy(dataset_config)
    copied_eval_config.data_path = opt.eval_path
    copied_eval_config.q_feat_dir = opt.t_feat_dir_pretrain_eval if opt.t_feat_dir_pretrain_eval is not None else opt.t_feat_dir
    eval_dataset = StartEndDataset(**copied_eval_config)

    
    # prepare model
    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    
    # load checkpoint for QVHighlight pretrain -> finetune
    if resume is not None:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["model"])
        logger.info("Loaded model checkpoint: {}".format(resume))
    
    count_parameters(model)
    logger.info("Start Training...")
    
    # start training
    train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)


def check_valid_combination(dataset, feature, domain):
    dataset_feature_map = {
        'qvhighlight': ['resnet_glove', 'clip', 'clip_slowfast', 'clip_slowfast_pann'],
        'qvhighlight_pretrain': ['resnet_glove', 'clip', 'clip_slowfast', 'clip_slowfast_pann'],
        'activitynet': ['resnet_glove', 'clip', 'clip_slowfast'],
        'charades': ['resnet_glove', 'clip', 'clip_slowfast'],
        'tacos': ['resnet_glove', 'clip', 'clip_slowfast'],
        'tvsum': ['resnet_glove', 'clip', 'clip_slowfast', 'i3d_clip'],
        'youtube_highlight': ['clip', 'clip_slowfast'],
        'clotho-moment': ['clap'],
    }

    domain_map = {
        'tvsum': ['BK', 'BT', 'DS', 'FM', 'GA', 'MS', 'PK', 'PR', 'VT', 'VU'],
        'youtube_highlight': ['dog', 'gymnastics', 'parkour', 'skating', 'skiing', 'surfing'],
    }

    if dataset in domain_map:
        return feature in dataset_feature_map[dataset] and domain in domain_map[dataset]
    else:
        return feature in dataset_feature_map[dataset]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, 
                        choices=['moment_detr', 'qd_detr', 'eatr', 'cg_detr', 'uvcom', 'tr_detr', 'taskweave_hd2mr', 'taskweave_mr2hd'],
                        help='model name. select from [moment_detr, qd_detr, eatr, cg_detr, uvcom, tr_detr, taskweave_hd2mr, taskweave_mr2hd]')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        choices=['activitynet', 'charades', 'qvhighlight', 'qvhighlight_pretrain', 'tacos', 'tvsum', 'youtube_highlight', 'clotho-moment'],
                        help='dataset name. select from [activitynet, charades, qvhighlight, qvhighlight_pretrain, tacos, tvsum, youtube_highlight, clotho-moment]')
    parser.add_argument('--feature', '-f', type=str, required=True,
                        choices=['resnet_glove', 'clip', 'clip_slowfast', 'clip_slowfast_pann', 'i3d_clip', 'clap'],
                        help='feature name. select from [resnet_glove, clip, clip_slowfast, clip_slowfast_pann, i3d_clip, clap].'
                             'NOTE: i3d_clip and clip_slowfast_pann are only for TVSum and QVHighlight, respectively.')
    parser.add_argument('--resume', '-r', type=str, help='specify model path for fine-tuning. If None, train the model from scratch.')
    parser.add_argument('--domain', '-dm', type=str,
                        choices=['BK', 'BT', 'DS', 'FM', 'GA', 'MS', 'PK', 'PR', 'VT', 'VU',
                                 'dog', 'gymnastics', 'parkour', 'skating', 'skiing', 'surfing'],
                        help='domain for highlight detection dataset (e.g., BK for TVSum, dog for YouTube Highlight).')
    args = parser.parse_args()


    is_valid = check_valid_combination(args.dataset, args.feature, args.domain)

    if is_valid:
        option_manager = BaseOptions(args.model, args.dataset, args.feature, args.resume, args.domain)
        option_manager.parse()
        option_manager.clean_and_makedirs()
        opt = option_manager.option
        main(opt, resume=args.resume, domain=args.domain)
    else:
        raise ValueError('The combination of dataset, feature, and domain is invalid: dataset={}, feature={}, domain={}'.format(args.dataset, args.feature, args.domain))
