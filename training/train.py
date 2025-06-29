"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

Moment-DETR (https://github.com/jayleicn/moment_detr)
Copyright (c) 2021 Jie Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import time
import json
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

from config import BaseOptions
from dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from training.onlineDataset import StartEndDataset, start_end_collate_ol, prepare_batch_inputs as prepare_batch_inputs_ol 
from cg_detr_dataset import CGDETR_StartEndDataset, cg_detr_start_end_collate, cg_detr_prepare_batch_inputs
from evaluate import eval_epoch, start_inference, setup_model

from lighthouse.common.utils.basic_utils import AverageMeter, dict_to_markdown, write_log, save_checkpoint, rename_latest_to_best
from lighthouse.common.utils.model_utils import count_parameters, ModelEMA

from lighthouse.common.loss_func import VTCLoss
from lighthouse.common.loss_func import CTC_Loss

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


# def additional_trdetr_losses(model_inputs, outputs, targets, opt):
#     # TR-DETR only loss
#     src_txt_mask,   src_vid_mask = model_inputs['src_txt_mask'], model_inputs['src_vid_mask']
#     pos_mask =  targets['src_pos_mask'] 

#     src_txt_ed, src_vid_ed =  outputs['src_txt_ed'], outputs['src_vid_ed']
#     loss_align = CTC_Loss()
#     loss_vid_txt_align = loss_align(src_vid_ed, src_txt_ed, pos_mask, src_vid_mask, src_txt_mask)

#     src_vid_cls_ed = outputs['src_vid_cls_ed']
#     src_txt_cls_ed = outputs['src_txt_cls_ed']
#     loss_align_VTC = VTCLoss()
#     loss_vid_txt_align_VTC = loss_align_VTC(src_txt_cls_ed, src_vid_cls_ed)

#     loss = opt.VTC_loss_coef * loss_vid_txt_align_VTC + opt.CTC_loss_coef * loss_vid_txt_align
#     return loss

# def calculate_taskweave_losses(loss_dict, weight_dict, hd_log_var, mr_log_var):
#     # TaskWeave only loss
#     grouped_losses = {"loss_mr": [], "loss_hd": []}
#     for k in loss_dict.keys():
#         if k in weight_dict:
#             if any(keyword in k for keyword in ["giou", "span", "label",'class_error']):
#                 grouped_losses["loss_mr"].append(loss_dict[k])
#             elif "saliency" in k:
#                 grouped_losses["loss_hd"].append(loss_dict[k])
#     loss_mr = sum(grouped_losses["loss_mr"])
#     loss_hd = sum(grouped_losses["loss_hd"])
#     losses = 2 * loss_hd * torch.exp(-hd_log_var) + 1 * loss_mr * torch.exp(-mr_log_var) + hd_log_var + mr_log_var
#     return losses

def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i):
    batch_input_fn = prepare_batch_inputs_ol  
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()

    if model.use_inter_memory and model.future_memory_sample_len > 0:
        print("after ",model.inter_memory.getUpdateCnt()," updates, interMemory reset in a new epoch.")
        model.inter_memory.reset()

    criterion.train()

    # init meters
    loss_meters = defaultdict(AverageMeter)

    # 初始化统计变量
    irrelevant_count = 0  # 统计 irrelevant 的预测次数
    other_count = 0       # 统计其他类别（start, mid, end）的预测次数

    num_training_examples = len(train_loader)
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        metas, batched_inputs = batch
        model_inputs, targets = batch_input_fn(metas,batched_inputs, opt.device)
        
        outputs = model(**model_inputs)  
        total_loss, loss_dict = criterion(outputs, targets)

        # # 把interMemory的loss放在这里一起优化
        # memory_loss = model.inter_memory.compute_diversity_loss_ortho()
        # total_loss += memory_loss*0.001
        # loss_dict["loss_memory"] = memory_loss.item()


        # 不用weight_dict，将loss简单相加
        losses = total_loss
        # torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        losses.backward()
            
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        # 统计预测结果
        frame_pred = outputs['frame_pred']  # (batch_size, #frames, #classes=4)
        pred_classes = torch.argmax(frame_pred, dim=-1)  # (batch_size, #frames)

        # 统计 irrelevant 和其他类别的预测次数
        irrelevant_count += (pred_classes == 3).sum().item()  # 3 是 irrelevant 的类别索引
        other_count += (pred_classes != 3).sum().item()       # 其他类别的预测次数

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            # loss_meters[k].update(float(v) * criterion.weight_dict[k] if k in criterion.weight_dict else float(v))
            loss_meters[k].update(float(v))

    # 打印统计结果
    logger.info(f"[Epoch {epoch_i+1}] Irrelevant predictions: {irrelevant_count}, Other predictions: {other_count}\t")

    write_log(opt, epoch_i, loss_meters)

def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    save_submission_filename = "latest_{}_val_preds.jsonl".format(opt.dset_name)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=lambda batch: start_end_collate_ol(batch, long_memory_sample_length=train_dataset.long_memory_sample_length),
        # collate_fn = start_end_collate_ol,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )

    if opt.model_ema:
        logger.info("Using model EMA...")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    # # train的时候，加载所有mid的标签，并set进模型中
    # mid_label_dict = train_dataset.load_all_mid_label_dict()
    # model.set_mid_label_dict(mid_label_dict)

    prev_best_score = 0
    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i)
        lr_scheduler.step()

        if opt.model_ema:
            model_ema.update(model)
        
        if (epoch_i + 1) % (opt.eval_epoch_interval * 2) == 0 and (epoch_i + 1) <= 20 or \
           (epoch_i + 1) % opt.eval_epoch_interval == 0 and (epoch_i + 1) > 20:
            # with torch.no_grad():
            if opt.model_ema:
                metrics, eval_loss_meters, latest_file_paths = \
                    eval_epoch(epoch_i, model_ema.module, val_dataset, optimizer, opt, save_submission_filename)
            else:
                metrics, eval_loss_meters, latest_file_paths = \
                    eval_epoch(epoch_i, model, val_dataset, optimizer, opt, save_submission_filename)
                    # eval_epoch(epoch_i, model, val_dataset, opt, save_submission_filename, criterion)


            # 修改日志记录部分
            write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode='val')            
            logger.info("metrics {}".format(pprint.pformat(metrics, indent=4)))
            
            stop_score = metrics["R@1,IoU=0.5"] + metrics["R@1,IoU=0.7"]
            # 如果当前分数优于之前的最佳分数，则保存模型
            if stop_score > prev_best_score:
                prev_best_score = stop_score
                # test时不优化interMemory，所以只用保存那些个parameter即可
                # save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
                # test时也优化interMemory
                if model.use_inter_memory and model.future_memory_sample_len > 0:
                    save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt, 
                        inter_memory_update_cnt=model.inter_memory.getUpdateCnt(), 
                        memory_optimizer=model.inter_memory.memory_optimizer)
                else:
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
        # OL 任务所需的参数
        chunk_interval=1,
        short_memory_sample_length=opt.short_memory_len,
        long_memory_sample_length=opt.long_memory_sample_len,
        future_memory_sample_length=opt.future_memory_sample_len,
        short_memory_stride=1,
        long_memory_stride=1,
        future_memory_stride=1,
        load_future_memory=False,
        test_mode=False,
    )

    train_dataset = StartEndDataset(**dataset_config)  
    copied_eval_config = copy.deepcopy(dataset_config)
    copied_eval_config.data_path = opt.eval_path
    copied_eval_config.q_feat_dir = opt.t_feat_dir_pretrain_eval if opt.t_feat_dir_pretrain_eval is not None else opt.t_feat_dir
    copied_eval_config.test_mode = True
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
                        choices=['ol_moment_detr','moment_detr', 'qd_detr', 'eatr', 'cg_detr', 'uvcom', 'tr_detr', 'taskweave_hd2mr', 'taskweave_mr2hd'],
                        help='model name. select from [moment_detr, qd_detr, eatr, cg_detr, uvcom, tr_detr, taskweave_hd2mr, taskweave_mr2hd]')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        choices=['activitynet', 'charades', 'qvhighlight', 'qvhighlight_pretrain', 'tacos', 'tvsum', 'youtube_highlight', 'clotho-moment'],
                        help='dataset name. select from [activitynet, charades, qvhighlight, qvhighlight_pretrain, tacos, tvsum, youtube_highlight, clotho-moment]')
    parser.add_argument('--feature', '-f', type=str, required=True,
                        choices=['resnet_glove', 'clip', 'clip_slowfast', 'clip_slowfast_pann', 'i3d_clip', 'clap'],
                        help='feature name. select from [resnet_glove, clip, clip_slowfast, clip_slowfast_pann, i3d_clip, clap].'
                             'NOTE: i3d_clip and clip_slowfast_pann are only for TVSum and QVHighlight, respectively.')
    parser.add_argument('--model_path', type=str, help='train from a saved model path')
    parser.add_argument('--froze_transformer', type=bool, default=False, help='froze transformer')
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
        
        opt.model_path = args.model_path
        opt.froze_transformer = args.froze_transformer
        main(opt, resume=args.resume, domain=args.domain)
    else:
        raise ValueError('The combination of dataset, feature, and domain is invalid: dataset={}, feature={}, domain={}'.format(args.dataset, args.feature, args.domain))
