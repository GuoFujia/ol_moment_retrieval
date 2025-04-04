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

import argparse
import pprint

from tqdm import tqdm, trange
import numpy as np
import os
from collections import OrderedDict, defaultdict
from easydict import EasyDict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lighthouse.common.utils.basic_utils import AverageMeter
from lighthouse.common.utils.span_utils import span_cxw_to_xx

from training.config import BaseOptions

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from training.onlineDataset import StartEndDataset, start_end_collate_ol, prepare_batch_inputs as prepare_batch_inputs_ol 
# from training.dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from training.cg_detr_dataset import CGDETR_StartEndDataset, cg_detr_start_end_collate, cg_detr_prepare_batch_inputs

from training.postprocessing import PostProcessorDETR
from standalone_eval.eval import eval_submission, eval_submission_ol, eval_submission_ol_2 

from lighthouse.common.utils.basic_utils import save_jsonl, save_json
from lighthouse.common.qd_detr import build_model as build_model_qd_detr
from lighthouse.common.moment_detr import build_model as build_model_moment_detr
from lighthouse.common.cg_detr import build_model as build_model_cg_detr
from lighthouse.common.eatr import build_model as build_model_eatr
from lighthouse.common.tr_detr import build_model as build_model_tr_detr
from lighthouse.common.uvcom import build_model as build_model_uvcom
from lighthouse.common.taskweave import build_model as build_model_task_weave
from lighthouse.common.ol_moment_detr import build_model as build_model_ol_moment_detr

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def eval_epoch_post_processing(submission, opt, gt_data, saliency_scores_all, save_submission_filename):
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    # 确保saliency_scores_all不为None
    if saliency_scores_all is None:
        logger.warning("saliency_scores_all is None, will use gt_data's saliency_scores")
        # saliency_scores_all = {d["vid"]: d["saliency_scores"] for d in gt_data}


    if opt.eval_split_name in ["val"]:
        #   使用修改后的评估函数
        # metrics = eval_submission_ol(submission, gt_data, saliency_scores_all)
        metrics = eval_submission_ol_2(submission, gt_data, saliency_scores_all)
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    return metrics, latest_file_paths


# for HL
# 这个用于tvsum和youtubehighlight数据集，只做HL
@torch.no_grad()
def compute_hl_results(epoch_i, model, eval_loader, opt, criterion=None):
    # batch_input_fn = cg_detr_prepare_batch_inputs  if opt.model_name == 'cg_detr' else prepare_batch_inputs
    batch_input_fn = prepare_batch_inputs_ol
    loss_meters = defaultdict(AverageMeter)

    video_ap_collected = []
    topk = 5 # top-5 map

    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        metas, batched_inputs = batch
        model_inputs, targets = batch_input_fn(metas, batched_inputs, opt.device)

        query_meta = batch[0]
        model_inputs, targets = batch_input_fn(batch[1], opt.device)

        if opt.model_name == 'taskweave':
            model_inputs['epoch_i'] = epoch_i
            outputs, _ = model(**model_inputs)
        else:
            outputs = model(**model_inputs)

        preds = outputs['saliency_scores']
        for meta, pred in zip(query_meta, preds):
            label = meta['label'] # raw label
            video_ap = []
            # Follow the UMT code "https://github.com/TencentARC/UMT/blob/main/datasets/tvsum.py"
            if opt.dset_name == 'tvsum':
                for i in range(20):
                    pred = pred.cpu()
                    cur_pred = pred[:len(label)]
                    inds = torch.argsort(cur_pred, descending=True, dim=-1)

                    # video_id = self.get_video_id(idx)
                    cur_label = torch.Tensor(label)[:, i]
                    cur_label = torch.where(cur_label > cur_label.median(), 1.0, .0)

                    cur_label = cur_label[inds].tolist()[:topk]

                    # if (num_gt := sum(cur_label)) == 0:
                    num_gt = sum(cur_label)
                    if num_gt == 0:
                        video_ap.append(0)
                        continue

                    hits = ap = rec = 0
                    prc = 1

                    for j, gt in enumerate(cur_label):
                        hits += gt

                        _rec = hits / num_gt
                        _prc = hits / (j + 1)

                        ap += (_rec - rec) * (prc + _prc) / 2
                        rec, prc = _rec, _prc

                    video_ap.append(ap)
            
            elif opt.dset_name == 'youtube_highlight':
                cur_pred = pred[:len(label)].cpu()
                inds = torch.argsort(cur_pred, descending=True, dim=-1)
                cur_label = torch.Tensor(label).squeeze()[inds].tolist()
                num_gt = sum(cur_label)
                if num_gt == 0:
                    video_ap.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(cur_label):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc
                
                video_ap.append(float(ap))

            else:
                raise NotImplementedError

            video_ap_collected.append(video_ap)  

    mean_ap = np.mean(video_ap_collected)
    submmission = dict(mAP=round(mean_ap, 5))
    
    return submmission, loss_meters


@torch.no_grad()
def compute_mr_results(epoch_i, model, eval_loader, opt, criterion=None):
    # batch_input_fn = cg_detr_prepare_batch_inputs if opt.model_name == 'cg_detr' else prepare_batch_inputs_ol
    batch_input_fn = prepare_batch_inputs_ol

    loss_meters = defaultdict(AverageMeter)

    mr_res = []
    # print(f"[DEBUG] Starting compute_mr_results with {len(eval_loader)} batches")
    for batch in tqdm(eval_loader, desc="compute probability for each frame as st ed or mid"):
         # batch 是一个元组，包含 metas 和 model_inputs

        metas, batched_inputs = batch  # 解包 batch
        model_inputs, targets = batch_input_fn(metas, batched_inputs, opt.device)  # 正确传递所有参数

        # print("batched_inputs keys:", batched_inputs.keys())
        # print("当前batch的short_memory_start和vid： ", batched_inputs["short_memory_start"])
        # input("check short_memory_start...")

        #   获得模型输出
        if opt.model_name == 'taskweave':
            model_inputs['epoch_i'] = epoch_i
            outputs, _ = model(**model_inputs)
        else:
            outputs = model(**model_inputs)

        # saliency scores
        _saliency_scores = outputs["saliency_scores"].half()  # (bsz, short_memory_length)

        # print("_saliency_scores.shape:", _saliency_scores.shape)
        # input("check _saliency_scores...")
        
        saliency_scores = []

        c, b = model_inputs["memory_len"][2], model_inputs["memory_len"][1]
        src_vid_mask_short = model_inputs["src_vid_mask"][:, -(c + b):-c] if c > 0 else model_inputs["src_vid_mask"][:, -b:]
        valid_vid_lengths = src_vid_mask_short.sum(1).cpu().tolist()

        # print("valid_vid_lengths:", valid_vid_lengths)
        # print("valid_vid_lengths len:", len(valid_vid_lengths))
        # print("src_vid_mask_short.shape:", src_vid_mask_short.shape)
        # input("check valid_vid_lengths...")

        for j in range(len(valid_vid_lengths)):
            valid_length = int(valid_vid_lengths[j])
            if valid_length > 0:  # 添加有效性检查
                saliency_scores.append(_saliency_scores[j, :valid_length].tolist())

        # compose predictions
        frame_pred = outputs["frame_pred"].cpu()    # (bsz,#queries,4), queries设置存疑
        frame_pred = torch.sigmoid(frame_pred)

        # print("in com mr res: ",frame_pred)
        # input("  ds  ")

        
        for idx, (meta, pred) in enumerate(zip(metas, frame_pred)):            
            cur_chunk_pred = dict(
                    qid=meta["qid"],
                    query=meta["query"],
                    vid=meta["vid"],
                    pred_start=meta["short_memory_start"],
                    pred_frame_prob=pred.tolist(),
                    pred_saliency_scores=saliency_scores[idx],
                )

            mr_res.append(cur_chunk_pred)

        if criterion:
            total_loss, loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

    print(f"[DEBUG] Finished compute_mr_results with {len(mr_res)} predictions")
    # input("check mr_res len...")
    return mr_res, loss_meters


def get_eval_res(epoch_i, model, eval_loader, opt, criterion):
    """compute and save query and video proposal embeddings"""
    # eval_res, eval_loss_meters = compute_mr_results(epoch_i, model, eval_loader, opt, criterion)

    #   criterion暂时还没修改，先不用
    eval_res, eval_loss_meters = compute_mr_results(epoch_i, model, eval_loader, opt, criterion) 
    return eval_res, eval_loss_meters


def eval_epoch(epoch_i, model, eval_dataset, opt, save_submission_filename, criterion=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None:
        criterion.eval()

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=lambda batch: start_end_collate_ol(batch, long_memory_sample_length=eval_dataset.long_memory_sample_length),
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    # reset mid_label_dict
    model.reset_mid_label_dict()

    # if opt.dset_name == 'tvsum' or opt.dset_name == 'youtube_highlight':
    #     metrics, eval_loss_meters = compute_hl_results(epoch_i, model, eval_loader, opt, criterion)
    #     # to match original save format
    #     submission = [{ "brief" : metrics }]
    #     save_metrics_path = os.path.join(opt.results_dir, save_submission_filename.replace('.jsonl', '_metrics.jsonl'))
    #     save_jsonl(submission, save_metrics_path)
    #     return submission[0], eval_loss_meters, [save_metrics_path]
    # else:

    submission, eval_loss_meters = get_eval_res(epoch_i, model, eval_loader, opt, criterion)  

    metrics, latest_file_paths = eval_epoch_post_processing(
        submission, opt, eval_dataset.chunk_infos, eval_dataset.saliency_scores_list, save_submission_filename)
    return metrics, eval_loss_meters, latest_file_paths

def build_model(opt):
    if opt.model_name == 'qd_detr':
        model, criterion = build_model_qd_detr(opt)
    elif opt.model_name == 'moment_detr':
        model, criterion = build_model_moment_detr(opt)
    elif opt.model_name == 'cg_detr':
        model, criterion = build_model_cg_detr(opt)
    elif opt.model_name == 'eatr':
        model, criterion = build_model_eatr(opt)
    elif opt.model_name == 'tr_detr':
        model, criterion = build_model_tr_detr(opt)
    elif opt.model_name == 'uvcom':
        model, criterion = build_model_uvcom(opt)
    elif opt.model_name == 'taskweave':
        model, criterion = build_model_task_weave(opt)
    elif opt.model_name == 'ol_moment_detr':
        model, criterion = build_model_ol_moment_detr(opt)
    else:
        raise NotImplementedError
    
    return model, criterion

def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)

    if opt.device == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)                    
        criterion.to(opt.device)

    print(opt)
    # 添加权重加载逻辑
    if 'model_path' in opt and opt.model_path:
        logger.info(f"Loading model weights from {opt.model_path}")
        try:
            # 加载 checkpoint 文件
            checkpoint = torch.load(opt.model_path, map_location=opt.device)
            # 将权重加载到模型中
            model.load_state_dict(checkpoint['model'], strict=False)  # strict=False 允许部分加载
            logger.info("Model weights loaded successfully (strict=False).")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
    else:
        logger.info("No model path provided. Using random initialization.")

    # 在加载权重后，判断是否冻结 transformer 的参数
    if "froze_transformer" in opt and opt.froze_transformer:
        for name, param in model.named_parameters():
            if "transformer" in name:  # 冻结所有 transformer 相关的参数
                param.requires_grad = False
                logger.info(f"Froze parameter: {name}")

    # 设置优化器和学习率调度器
    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    return model, criterion, optimizer, lr_scheduler


def start_inference(opt, domain=None):
    logger.info("Setup config, data and model...")

    cudnn.benchmark = True
    cudnn.deterministic = False
    load_labels = opt.eval_split_name == 'val'
    epoch_i = None # for TaskWeave.
    
    # dataset & data loader
    dataset_config = EasyDict(
        dset_name=opt.dset_name,
        domain=domain,
        data_path=opt.eval_path,
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
        load_labels=load_labels,
        #   ol任务需要的参数
        chunk_interval=1,
        short_memory_sample_length=opt.short_memory_len,
        long_memory_sample_length=opt.long_memory_sample_len,
        future_memory_sample_length=opt.future_memory_sample_len,
        short_memory_stride=1,
        long_memory_stride=1,
        future_memory_stride=1,
        load_future_memory=False,
        test_mode=True,    
    )
    
    # eval_dataset = CGDETR_StartEndDataset(**dataset_config) if opt.model_name == 'cg_detr' else StartEndDataset(**dataset_config)
    eval_dataset = StartEndDataset(**dataset_config)
    model, criterion, _, _ = setup_model(opt)

    
    logger.info("Model checkpoint: {}".format(opt.model_path))
    if not load_labels:
        criterion = None
    
    save_submission_filename = "hl_{}_submission.jsonl".format(opt.eval_split_name)

    logger.info("Starting inference...")
    with torch.no_grad():
        metrics, eval_loss_meters, latest_file_paths = \
            eval_epoch(epoch_i, model, eval_dataset, opt, save_submission_filename)

    # print("metrics is {}".format(metrics))

    if opt.eval_split_name == 'val':
        logger.info("metrics_no_nms {}".format(pprint.pformat(metrics, indent=4)))


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
                        choices=['ol_moment_detr', 'moment_detr', 'qd_detr', 'eatr', 'cg_detr', 'uvcom', 'tr_detr', 'taskweave_hd2mr', 'taskweave_mr2hd'],
                        help='model name. select from [moment_detr, qd_detr, eatr, cg_detr, uvcom, tr_detr, taskweave_hd2mr, taskweave_mr2hd]')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        choices=['activitynet', 'charades', 'qvhighlight', 'qvhighlight_pretrain', 'tacos', 'tvsum', 'youtube_highlight', 'clotho-moment', 'unav100-subset', 'tut2017'],
                        help='dataset name. select from [activitynet, charades, qvhighlight, qvhighlight_pretrain, tacos, tvsum, youtube_highlight, clotho-moment, unav100-subset, tut2017]')
    parser.add_argument('--feature', '-f', type=str, required=True,
                        choices=['resnet_glove', 'clip', 'clip_slowfast', 'clip_slowfast_pann', 'i3d_clip', 'clap'],
                        help='feature name. select from [resnet_glove, clip, clip_slowfast, clip_slowfast_pann, i3d_clip, clap].'
                             'NOTE: i3d_clip and clip_slowfast_pann are only for TVSum and QVHighlight, respectively')
    parser.add_argument('--model_path', type=str, required=True, help='saved model path')
    parser.add_argument('--split', type=str, required=True, choices=['val', 'test'], help='val or test')
    parser.add_argument('--eval_path', type=str, required=True, help='evaluation data')
    parser.add_argument('--domain', '-dm', type=str,
                        choices=['BK', 'BT', 'DS', 'FM', 'GA', 'MS', 'PK', 'PR', 'VT', 'VU',
                                 'dog', 'gymnastics', 'parkour', 'skating', 'skiing', 'surfing'],
                        help='domain for highlight detection dataset (e.g., BK for TVSum, dog for YouTube Highlight).')

    args = parser.parse_args()
    is_valid = check_valid_combination(args.dataset, args.feature, args.domain)

    if is_valid:
        resume = False
        option_manager = BaseOptions(args.model, args.dataset, args.feature, resume, args.domain)
        option_manager.parse()
        opt = option_manager.option
        os.makedirs(opt.results_dir, exist_ok=True)

        opt.model_path = args.model_path
        opt.eval_split_name = args.split
        opt.eval_path = args.eval_path
        start_inference(opt, domain=args.domain)
    
    else:
        raise ValueError('The combination of dataset and feature is invalid: dataset={}, feature={}'.format(args.dataset, args.feature))
