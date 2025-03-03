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

MIT License

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

import numpy as np
from collections import OrderedDict, defaultdict
import json
import time
import copy
import multiprocessing as mp
from standalone_eval.utils import compute_average_precision_detection, \
    compute_temporal_iou_batch_cross, compute_temporal_iou_batch_paired, load_jsonl, get_ap


def compute_average_precision_detection_wrapper(
        input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return qid, scores


def compute_mr_ap(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10),
                  max_gt_windows=None, max_pred_windows=10, num_workers=8, chunksize=50):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    for d in submission:
        pred_windows = d["pred_relevant_windows"][:max_pred_windows] \
            if max_pred_windows is not None else d["pred_relevant_windows"]
        qid = d["qid"]
        for w in pred_windows:
            pred_qid2data[qid].append({
                "video-id": d["qid"],  # in order to use the API
                "t-start": w[0],
                "t-end": w[1],
                "score": w[2]
            })

    gt_qid2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = d["relevant_windows"][:max_gt_windows] \
            if max_gt_windows is not None else d["relevant_windows"]
        qid = d["qid"]
        for w in gt_windows:
            gt_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": w[0],
                "t-end": w[1]
            })
    qid2ap_list = {}
    # start_time = time.time()
    data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]
    from functools import partial
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    # print(f"compute_average_precision_detection {time.time() - start_time:.2f} seconds.")
    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10)):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {d["qid"]: d["pred_relevant_windows"][0][:2] for d in submission}  # :2 rm scores
    # gt_qid2window = {d["qid"]: d["relevant_windows"][0] for d in ground_truth}
    gt_qid2window = {}
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_idx = 0
        if len(cur_gt_windows) > 0:  # select the GT window that has the highest IoU
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]), np.array(d["relevant_windows"])
            )[0]
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]

    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = {}
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}")
    return iou_thd2recall_at_one


def get_window_len(window):
    return window[1] - window[0]


def get_data_by_range(submission, ground_truth, len_range):
    """ keep queries with ground truth window length in the specified length range.
    Args:
        submission:
        ground_truth:
        len_range: [min_l (int), max_l (int)]. the range is (min_l, max_l], i.e., min_l < l <= max_l
    """
    min_l, max_l = len_range
    if min_l == 0 and max_l == 150:  # min and max l in dataset
        return submission, ground_truth

    # only keep ground truth with windows in the specified length range
    # if multiple GT windows exists, we only keep the ones in the range
    ground_truth_in_range = []
    gt_qids_in_range = set()
    for d in ground_truth:
        rel_windows_in_range = [
            w for w in d["relevant_windows"] if min_l < get_window_len(w) <= max_l]
        if len(rel_windows_in_range) > 0:
            d = copy.deepcopy(d)
            d["relevant_windows"] = rel_windows_in_range
            ground_truth_in_range.append(d)
            gt_qids_in_range.add(d["qid"])

    # keep only submissions for ground_truth_in_range
    submission_in_range = []
    for d in submission:
        if d["qid"] in gt_qids_in_range:
            submission_in_range.append(copy.deepcopy(d))

    return submission_in_range, ground_truth_in_range


def eval_moment_retrieval(submission, ground_truth, verbose=True):
    #length_ranges = [[0, 10], [10, 30], [30, 150], [0, 150], ]  #
    #range_names = ["short", "middle", "long", "full"]
    length_ranges = [[0, 1500]] # TODO: cover all examples?
    range_names = ["full"]

    ret_metrics = {}
    for l_range, name in zip(length_ranges, range_names):
        if verbose:
            start_time = time.time()
        _submission, _ground_truth = get_data_by_range(submission, ground_truth, l_range)
        print(f"{name}: {l_range}, {len(_ground_truth)}/{len(ground_truth)}="
              f"{100*len(_ground_truth)/len(ground_truth):.2f} examples.")
        iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth, num_workers=8, chunksize=50)
        iou_thd2recall_at_one = compute_mr_r1(_submission, _ground_truth)
        ret_metrics[name] = {"MR-mAP": iou_thd2average_precision, "MR-R1": iou_thd2recall_at_one}
        if verbose:
            print(f"[eval_moment_retrieval] [{name}] {time.time() - start_time:.2f} seconds")
    return ret_metrics


def compute_hl_hit1(qid2preds, qid2gt_scores_binary):
    qid2max_scored_clip_idx = {k: np.argmax(v["pred_saliency_scores"]) for k, v in qid2preds.items()}
    hit_scores = np.zeros((len(qid2preds), 3))
    qids = list(qid2preds.keys())
    for idx, qid in enumerate(qids):
        pred_clip_idx = qid2max_scored_clip_idx[qid]
        gt_scores_binary = qid2gt_scores_binary[qid]   # (#clips, 3)
        if pred_clip_idx < len(gt_scores_binary):
            hit_scores[idx] = gt_scores_binary[pred_clip_idx]
    # aggregate scores from 3 separate annotations (3 workers) by taking the max.
    # then average scores from all queries.
    hit_at_one = float(f"{100 * np.mean(np.max(hit_scores, 1)):.2f}")
    return hit_at_one


def compute_hl_ap(qid2preds, qid2gt_scores_binary, num_workers=8, chunksize=50):
    qid2pred_scores = {k: v["pred_saliency_scores"] for k, v in qid2preds.items()}
    ap_scores = np.zeros((len(qid2preds), 3))   # (#preds, 3)
    qids = list(qid2preds.keys())
    input_tuples = []
    for idx, qid in enumerate(qids):
        for w_idx in range(3):  # annotation score idx
            y_true = qid2gt_scores_binary[qid][:, w_idx]
            y_predict = np.array(qid2pred_scores[qid])
            input_tuples.append((idx, w_idx, y_true, y_predict))

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for idx, w_idx, score in pool.imap_unordered(
                    compute_ap_from_tuple, input_tuples, chunksize=chunksize):
                ap_scores[idx, w_idx] = score
    else:
        for input_tuple in input_tuples:
            idx, w_idx, score = compute_ap_from_tuple(input_tuple)
            ap_scores[idx, w_idx] = score

    # it's the same if we first average across different annotations, then average across queries
    # since all queries have the same #annotations.
    mean_ap = float(f"{100 * np.mean(ap_scores):.2f}")
    return mean_ap


def compute_ap_from_tuple(input_tuple):
    idx, w_idx, y_true, y_predict = input_tuple
    if len(y_true) < len(y_predict):
        # print(f"len(y_true) < len(y_predict) {len(y_true), len(y_predict)}")
        y_predict = y_predict[:len(y_true)]
    elif len(y_true) > len(y_predict):
        # print(f"len(y_true) > len(y_predict) {len(y_true), len(y_predict)}")
        _y_predict = np.zeros(len(y_true))
        _y_predict[:len(y_predict)] = y_predict
        y_predict = _y_predict

    score = get_ap(y_true, y_predict)
    return idx, w_idx, score


def mk_gt_scores(gt_data, clip_length=2):
    """gt_data, dict, """
    num_clips = int(gt_data["duration"] / clip_length)
    saliency_scores_full_video = np.zeros((num_clips, 3))
    relevant_clip_ids = np.array(gt_data["relevant_clip_ids"])  # (#relevant_clip_ids, )
    saliency_scores_relevant_clips = np.array(gt_data["saliency_scores"])  # (#relevant_clip_ids, 3)
    saliency_scores_full_video[relevant_clip_ids] = saliency_scores_relevant_clips
    return saliency_scores_full_video  # (#clips_in_video, 3)  the scores are in range [0, 4]


def eval_highlight(submission, ground_truth, verbose=True):
    """
    Args:
        submission:
        ground_truth:
        verbose:
    """
    qid2preds = {d["qid"]: d for d in submission}
    qid2gt_scores_full_range = {d["qid"]: mk_gt_scores(d) for d in ground_truth}  # scores in range [0, 4]
    # gt_saliency_score_min: int, in [0, 1, 2, 3, 4]. The minimum score for a positive clip.
    gt_saliency_score_min_list = [2, 3, 4]
    saliency_score_names = ["Fair", "Good", "VeryGood"]
    highlight_det_metrics = {}
    for gt_saliency_score_min, score_name in zip(gt_saliency_score_min_list, saliency_score_names):
        start_time = time.time()
        qid2gt_scores_binary = {
            k: (v >= gt_saliency_score_min).astype(float)
            for k, v in qid2gt_scores_full_range.items()}  # scores in [0, 1]
        hit_at_one = compute_hl_hit1(qid2preds, qid2gt_scores_binary)
        mean_ap = compute_hl_ap(qid2preds, qid2gt_scores_binary)
        highlight_det_metrics[f"HL-min-{score_name}"] = {"HL-mAP": mean_ap, "HL-Hit1": hit_at_one}
        if verbose:
            print(f"Calculating highlight scores with min score {gt_saliency_score_min} ({score_name})")
            print(f"Time cost {time.time() - start_time:.2f} seconds")
    return highlight_det_metrics

def eval_submission(submission, ground_truth, verbose=True, match_number=True):
    """
    Args:
        submission: list(dict), each dict is {
            qid: str,
            query: str,
            vid: str,
            pred_relevant_windows: list([st, ed]),
            pred_saliency_scores: list(float), len == #clips in video.
                i.e., each clip in the video will have a saliency score.
        }
        ground_truth: list(dict), each dict is     {
          "qid": 7803,
          "query": "Man in gray top walks from outside to inside.",
          "duration": 150,
          "vid": "RoripwjYFp8_360.0_510.0",
          "relevant_clip_ids": [13, 14, 15, 16, 17]
          "saliency_scores": [[4, 4, 2], [3, 4, 2], [2, 2, 3], [2, 2, 2], [0, 1, 3]]
               each sublist corresponds to one clip in relevant_clip_ids.
               The 3 elements in the sublist are scores from 3 different workers. The
               scores are in [0, 1, 2, 3, 4], meaning [Very Bad, ..., Good, Very Good]
        }
        verbose:
        match_number:

    Returns:

    """
    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    if match_number:
        assert pred_qids == gt_qids, \
            f"qids in ground_truth and submission must match. " \
            f"use `match_number=False` if you wish to disable this check"
    else:  # only leave the items that exists in both submission and ground_truth
        shared_qids = pred_qids.intersection(gt_qids)
        submission = [e for e in submission if e["qid"] in shared_qids]
        ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_relevant_windows" in submission[0]:
        moment_ret_scores = eval_moment_retrieval(
            submission, ground_truth, verbose=verbose)
        eval_metrics.update(moment_ret_scores)
        moment_ret_scores_brief = {
            "MR-full-mAP": moment_ret_scores["full"]["MR-mAP"]["average"],
            "MR-full-mAP@0.5": moment_ret_scores["full"]["MR-mAP"]["0.5"],
            "MR-full-mAP@0.75": moment_ret_scores["full"]["MR-mAP"]["0.75"],
            "MR-full-R1@0.5": moment_ret_scores["full"]["MR-R1"]["0.5"],
            "MR-full-R1@0.7": moment_ret_scores["full"]["MR-R1"]["0.7"],
        }
        eval_metrics_brief.update(
            sorted([(k, v) for k, v in moment_ret_scores_brief.items()], key=lambda x: x[0]))

    if "pred_saliency_scores" in submission[0]:
        highlight_det_scores = eval_highlight(
            submission, ground_truth, verbose=verbose)
        eval_metrics.update(highlight_det_scores)
        highlight_det_scores_brief = dict([
            (f"{k}-{sub_k.split('-')[1]}", v[sub_k])
            for k, v in highlight_det_scores.items() for sub_k in v])
        eval_metrics_brief.update(highlight_det_scores_brief)

    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics

def eval_submission_ol(submission, ground_truth, saliency_scores_all, verbose=True, match_number=True):
    """
    Args:
        submission: list(dict), each dict is {
            qid: str,
            query: str,
            vid: str,
            pred_start: int,
            pred_frame_prob: (queries, 4),  # [st, mid, ed, irre] 的概率
            pred_saliency_scores: (short_memory_sample_length),  # saliency 预测
        }
        ground_truth: list(dict), each dict is {
            "qid": str,
            "query": str,
            "duration": float,
            "duration_frame": int,
            "short_memory_start": int,
            "vid": str,
            "start_label": (short_memory_sample_length),  # 每帧是否为 st
            "end_label": (short_memory_sample_length),  # 每帧是否为 ed
            "semantic_label": (short_memory_sample_length),  # 每帧是否在查询区间内
        }
        saliency_scores_all: provide saliency scores for all video, dicted by vid 
        verbose: bool, 是否打印详细信息
        match_number: bool, 是否强制匹配 qid 数量

    Returns:
        dict: 包含分类任务和回归任务的评估指标
    """
    # 1. 对 submission 和 ground_truth 进行排序
    def get_sort_key(item):
        return (item["qid"], item["vid"], item.get("pred_start", item.get("short_memory_start")))

    submission_sorted = sorted(submission, key=get_sort_key)
    ground_truth_sorted = sorted(ground_truth, key=get_sort_key)

    # print("before matching, there're {} submissions and {} gt".format(len(submission_sorted),len(ground_truth_sorted)))

    # 2. 使用双指针法匹配 submission 和 ground_truth
    matched_submission = []
    matched_ground_truth = []
    i, j = 0, 0
    while i < len(submission_sorted) and j < len(ground_truth_sorted):
        sub = submission_sorted[i]
        gt = ground_truth_sorted[j]
        sub_key = (sub["qid"], sub["vid"], sub["pred_start"])
        gt_key = (gt["qid"], gt["vid"], gt["short_memory_start"])

        if sub_key == gt_key:
            matched_submission.append(sub)
            matched_ground_truth.append(gt)
            i += 1
            j += 1
        elif sub_key < gt_key:
            i += 1
        else:
            j += 1

    print("after matching, matched_submission's num is {}, and matched_ground_truth's is {}".format(len(matched_submission),len(matched_ground_truth)))

    # 3. 处理 [st, mid, ed, irre] 的预测（分类任务）
    frame_classification_metrics = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
    }
    if len(matched_submission) > 0:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # 将 ground truth 转换为 one-hot 编码
        gt_frame_labels = []
        for gt in matched_ground_truth:

            # print(gt.keys())

            start_label = gt["start_label"]
            end_label = gt["end_label"]
            semantic_label = gt["semantic_label"]
            frame_labels = []
            for i in range(len(start_label)):
                if start_label[i] == 1:
                    frame_labels.append([1, 0, 0, 0])  # st
                elif end_label[i] == 1:
                    frame_labels.append([0, 0, 1, 0])  # ed
                elif semantic_label[i] == 1:
                    frame_labels.append([0, 1, 0, 0])  # mid
                else:
                    frame_labels.append([0, 0, 0, 1])  # irre
            gt_frame_labels.extend(frame_labels)

        # 将预测结果展平
        pred_frame_probs = []
        for sub in matched_submission:
            pred_frame_probs.extend(sub["pred_frame_prob"])
        pred_frame_labels = [np.argmax(prob) for prob in pred_frame_probs]

        # 计算分类指标
        gt_frame_labels = np.argmax(gt_frame_labels, axis=1)
        frame_classification_metrics["accuracy"] = accuracy_score(gt_frame_labels, pred_frame_labels)
        frame_classification_metrics["precision"] = precision_score(gt_frame_labels, pred_frame_labels, average="macro")
        frame_classification_metrics["recall"] = recall_score(gt_frame_labels, pred_frame_labels, average="macro")
        frame_classification_metrics["f1_score"] = f1_score(gt_frame_labels, pred_frame_labels, average="macro")

    # 4. 处理 saliency 的预测（回归任务）
    saliency_regression_metrics = {
        "mse": 0.0,
        "mae": 0.0,
    }
    if len(matched_submission) > 0:
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        # 将 ground truth 和预测结果展平
        gt_saliency_scores = []
        pred_saliency_scores = []
        for sub, gt in zip(matched_submission, matched_ground_truth):
            pred_saliency_scores.extend(sub["pred_saliency_scores"])
            sample_st=gt["short_memory_start"]
            sample_ed=sample_st+len(gt["start_label"])
            gt_saliency_score=saliency_scores_all[gt["vid"]][sample_st:sample_ed]
            gt_saliency_scores.extend(gt_saliency_score)

        # 计算回归指标
        saliency_regression_metrics["mse"] = mean_squared_error(gt_saliency_scores, pred_saliency_scores)
        saliency_regression_metrics["mae"] = mean_absolute_error(gt_saliency_scores, pred_saliency_scores)

    # 5. 整理评估结果
    final_eval_metrics = OrderedDict()
    final_eval_metrics["frame_classification"] = frame_classification_metrics
    final_eval_metrics["saliency_regression"] = saliency_regression_metrics

    return final_eval_metrics

def eval_submission_ol_2(submission, ground_truth, saliency_scores_all, 
                      verbose=True, match_number=True, n_list=(1, 5), iou_thresholds=(0.5, 0.7)):
    """
    修改说明：
    1. 删除帧级分类指标（st/mid/ed分类）
    2. 新增基于片段定位的 R@n, IoU=m 评估逻辑
    3. 保留显著性回归评估
    """
    # ================== 1. 数据对齐 ==================
    def get_sort_key(item):
        return (item["qid"], item["vid"], item.get("pred_start", item.get("short_memory_start")))

    submission_sorted = sorted(submission, key=get_sort_key)
    ground_truth_sorted = sorted(ground_truth, key=get_sort_key)

    # 双指针对齐 submission 和 ground_truth
    matched_data = []
    i = j = 0
    while i < len(submission_sorted) and j < len(ground_truth_sorted):
        sub = submission_sorted[i]
        gt = ground_truth_sorted[j]
        sub_key = (sub["qid"], sub["vid"], sub["pred_start"])
        gt_key = (gt["qid"], gt["vid"], gt["short_memory_start"])

        if sub_key == gt_key:
            matched_data.append((sub, gt))
            i += 1
            j += 1
        elif sub_key < gt_key:
            i += 1
        else:
            j += 1
        
    #   将匹配后的gt和pred按照"qid，vid"分组,并按时间顺序排序
    matched_data_grouped = {}
    for sub, gt in matched_data:
        key = (sub["qid"], sub["vid"])
        if key not in matched_data_grouped:
            matched_data_grouped[key] = []
        matched_data_grouped[key].append((sub, gt))
    
    # 对每组内的样本按照时间顺序排序
    for key in matched_data_grouped:
        matched_data_grouped[key].sort(key=lambda x: x[1]["short_memory_start"])
        
        # 拼接每组内的预测标签
        for i in range(len(matched_data_grouped[key])-1):
            cur_sub, cur_gt = matched_data_grouped[key][i]
            next_sub, next_gt = matched_data_grouped[key][i+1]
            
            # 拼接frame_prob预测
            cur_sub["pred_frame_prob"].extend(next_sub["pred_frame_prob"])
            
            # 拼接saliency_scores预测(如果存在)
            if "pred_saliency_scores" in cur_sub and "pred_saliency_scores" in next_sub:
                cur_sub["pred_saliency_scores"].extend(next_sub["pred_saliency_scores"])

    # 打印部分匹配的gt和pred
    print("部分匹配的gt和pred:")
    for key in matched_data_grouped:
        print(f"query: {key[0]}, video: {key[1]}")
        for sub, gt in matched_data_grouped[key]:
            print(f"gt: {gt['short_memory_start']}, pred: {sub['pred_start']}")
    # input("请按回车键继续...")

    # ================== 2. 片段定位评估（R@n, IoU=m）==================
    r_at_n_metrics = OrderedDict()
    for n in n_list:
        for m in iou_thresholds:
            r_at_n_metrics[f"R@{n},IoU={m}"] = 0.0

    # 遍历每个匹配的 query-video 组
    for key, group in matched_data_grouped.items():
        # 获取该组的第一个gt来获取视频总长度
        _, first_gt = group[0]
        clip_length = first_gt["duration_frame"]

        # 收集该组内所有的预测概率
        all_st_probs = []
        all_ed_probs = []
        all_pred_starts = []
        for sub, _ in group:
            all_st_probs.extend(np.array(sub["pred_frame_prob"])[:, 0])
            all_ed_probs.extend(np.array(sub["pred_frame_prob"])[:, 2])
            all_pred_starts.extend([sub["pred_start"]] * len(sub["pred_frame_prob"]))

        # 生成跨chunk的候选片段
        candidate_moments = generate_cross_chunk_candidate_moments(
            pred_starts=all_pred_starts,
            st_probs=all_st_probs,
            ed_probs=all_ed_probs,
            clip_length=clip_length
        )

        # 获取该组的所有真实片段并去重
        gt_spans_set = set()
        for _, gt in group:
            gt_start = gt["short_memory_start"] + np.argmax(gt["start_label"])
            gt_end = gt["short_memory_start"] + np.argmax(gt["end_label"])
            if gt_start < gt_end:
                gt_spans_set.add((gt_start, gt_end))
        gt_spans = list(gt_spans_set)

        # #   打印当前的candidate_moments和gt_spans
        # print("当前的candidate_moments和gt_spans:")
        # print("candidate_moments:", candidate_moments)
        # print("gt_spans:", gt_spans)
        # print("key:", key)
        # input("请按回车键继续...")

        # 计算该组的 R@n, IoU=m
        for n in n_list:
            for m in iou_thresholds:
                topn_moments = candidate_moments[:n]
                # 对每个真实片段计算最大IoU
                for gt_span in gt_spans:
                    max_iou = max(calculate_iou(k, gt_span) for k in topn_moments)
                    if max_iou >= m:
                        r_at_n_metrics[f"R@{n},IoU={m}"] += 1
                        break  # 只要有一个真实片段满足条件就可以

    # 转换为百分比
    total_queries = len(matched_data_grouped)
    for k in r_at_n_metrics:
        r_at_n_metrics[k] = round(r_at_n_metrics[k] / total_queries * 100, 2) if total_queries > 0 else 0.0

    # ================== 3. 显著性回归评估 ==================
    if "pred_saliency_scores" in submission[0]:
        saliency_metrics = {"mse": 0.0, "mae": 0.0}
        if len(matched_data) > 0:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            pred_scores, gt_scores = [], []
            for sub, gt in matched_data:
                vid = gt["vid"]
                st = gt["short_memory_start"]
                
                # 获取预测分数
                pred_scores_chunk = sub["pred_saliency_scores"]
                
                # 获取ground truth分数
                ed = st + len(pred_scores_chunk)  # 使用预测分数的长度来确定结束位置
                gt_scores_chunk = saliency_scores_all[vid][st:ed]
                
                # 确保长度匹配
                min_len = min(len(pred_scores_chunk), len(gt_scores_chunk))
                pred_scores.extend(pred_scores_chunk[:min_len])
                gt_scores.extend(gt_scores_chunk[:min_len])

                # print("saliency: gt:",gt_scores," sub:",pred_scores)
                # input("please wait")
            
            # 再次验证长度
            assert len(pred_scores) == len(gt_scores), \
                f"Inconsistent lengths: pred_scores={len(pred_scores)}, gt_scores={len(gt_scores)}"
            
            # 计算指标
            if len(pred_scores) > 0:
                saliency_metrics["mse"] = mean_squared_error(gt_scores, pred_scores)
                saliency_metrics["mae"] = mean_absolute_error(gt_scores, pred_scores)

            

    # ================== 4. 最终指标整合 ==================
    final_metrics = OrderedDict()
    final_metrics.update(r_at_n_metrics)
    if "pred_saliency_scores" in submission[0]:
        final_metrics["saliency_mse"] = saliency_metrics["mse"]
        final_metrics["saliency_mae"] = saliency_metrics["mae"]

    return final_metrics

# ---------------------- 工具函数 ----------------------
def calculate_iou(pred_span, gt_span):
    """计算两个片段的重叠度 (IoU)"""
    pred_start, pred_end = pred_span
    gt_start, gt_end = gt_span

    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0

def generate_cross_chunk_candidate_moments(pred_starts, st_probs, ed_probs, clip_length, topk=100):
    """
    生成跨chunk的候选片段：
    1. 根据多个chunk的 st_probs 和 ed_probs 生成候选 (start, end) 对
    2. 考虑不同chunk之间的时间关系
    3. 按置信度排序（st_prob * ed_prob）
    4. 使用 NMS 去除重叠片段
    
    参数:
        pred_starts: 每个预测概率对应的起始帧位置列表
        st_probs: 所有chunk拼接后的开始概率列表
        ed_probs: 所有chunk拼接后的结束概率列表
        clip_length: 视频总帧数
        topk: 返回的候选片段数量
        
    返回:
        候选片段列表，每个元素为 (start, end) 元组
    """
    # 生成所有可能的候选
    candidates = []
    for i in range(len(st_probs)):
        for j in range(i, len(ed_probs)):
            if j - i >= 1:  # 至少持续1帧
                conf = st_probs[i] * ed_probs[j]
                start_global = pred_starts[i]  # 使用对应的全局起始位置
                end_global = pred_starts[j]    # 使用对应的全局起始位置
                
                # 计算实际的结束位置（考虑到每个位置可能是不同chunk的起始位置）
                # 这里假设每个位置的预测是针对该位置开始的帧
                start_frame = start_global + i - pred_starts[i]
                end_frame = end_global + j - pred_starts[j]
                
                # 确保片段在视频范围内
                if 0 <= start_frame < end_frame < clip_length:
                    candidates.append((start_frame, end_frame, conf))

    # 按置信度降序排序
    candidates.sort(key=lambda x: x[2], reverse=True)

    # 非极大值抑制 (NMS)
    keep = []
    while candidates:
        keep.append(candidates[0])
        candidates = [c for c in candidates if 
            calculate_iou((c[0], c[1]), (keep[-1][0], keep[-1][1])) < 0.5]
    
    res = [(start, end) for start, end, _ in keep[:topk]]

    # 并打印前5个候选片段,再打印这几个片段的置信度（prob_st,prob_ed）
    print("前5个候选片段:")
    for i, (start, end) in enumerate(res[:5]):
        print(f"  {i+1}. 起始帧: {start}, 结束帧: {end}, 持续时间: {end-start} 帧")
        print(f"   置信度: prob_st={st_probs[i]}, prob_ed={ed_probs[j]}")
    # 暂停程序
    input("请按回车键继续...")

    return res

def generate_candidate_moments(pred_start, st_probs, ed_probs, clip_length, topk=100):
    """
    生成候选片段：
    1. 根据 st_probs 和 ed_probs 生成候选 (start, end) 对
    2. 按置信度排序（st_prob * ed_prob）
    3. 使用 NMS 去除重叠片段
    """
    # 生成所有可能的候选
    candidates = []
    for i in range(len(st_probs)):
        for j in range(i, len(ed_probs)):
            if j - i >= 1:  # 至少持续1帧
                conf = st_probs[i] * ed_probs[j]
                start_global = pred_start + i
                end_global = pred_start + j
                # candidates.append((start_global, end_global, conf))
                candidates.append((start_global, end_global, conf))

    # 按置信度降序排序
    candidates.sort(key=lambda x: x[2], reverse=True)

    # 非极大值抑制 (NMS)
    keep = []
    while candidates:
        keep.append(candidates[0])
        candidates = [c for c in candidates if 
            calculate_iou((c[0], c[1]), (keep[-1][0], keep[-1][1])) < 0.5]
    
    return [(start, end) for start, end, _ in keep[:topk]]

def eval_main():
    import argparse
    parser = argparse.ArgumentParser(description="Moments and Highlights Evaluation Script")
    parser.add_argument("--submission_path", type=str, help="path to generated prediction file")
    parser.add_argument("--gt_path", type=str, help="path to GT file")
    parser.add_argument("--save_path", type=str, help="path to save the results")
    parser.add_argument("--not_verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.not_verbose
    submission = load_jsonl(args.submission_path)
    gt = load_jsonl(args.gt_path)
    results = eval_submission(submission, gt, verbose=verbose)
    if verbose:
        print(json.dumps(results, indent=4))

    with open(args.save_path, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    eval_main()
