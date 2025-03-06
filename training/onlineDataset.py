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

Copyright (c) 2022 WonJun Moon

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
import math
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from lighthouse.common.utils.basic_utils import load_jsonl, l2_normalize_np_array
from lighthouse.common.utils.tensor_utils import pad_sequences_1d
from lighthouse.common.utils.span_utils import span_xx_to_cxw
from torchtext import vocab
import torch.nn as nn

logger = logging.getLogger(__name__)

##  适应online任务做的修改
##  1.在StartEndSataset中添加了一系列属性和参数
##  2.在StartEndSataset类中添加chunk_all_videos方法
##  3.在get_item方法中：增加对短期、长期和未来记忆特征的支持；增加提案生成的逻辑
class StartEndDataset(Dataset):
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",aa
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """
    def __init__(self, dset_name, domain, data_path, v_feat_dirs, a_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state", v_feat_types="clip", a_feat_types="pann", 
                 max_q_l=32, max_v_l=75, max_a_l=75, ctx_mode="video", clip_len=2,
                 max_windows=5, span_loss_type="l1", load_labels=True,
                 chunk_interval=1, short_memory_sample_length=8, long_memory_sample_length=16,
                 future_memory_sample_length=8, short_memory_stride=1, long_memory_stride=1,
                 future_memory_stride=1, load_future_memory=False,test_mode=False,
                 use_gaussian_labels=True, alpha_s=0.25, alpha_m=0.21, alpha_e=0.25):
        self.fps = 0.5  #qv数据集定义的clip就是2s,把clip级特征看作帧级别特征用的话，就是0.5
                        #而其他数据集，follw lighthose的设置，也是2s一次采样，也可以看作0.5        
        self.dset_name = dset_name
        self.domain = domain
        self.data_path = data_path
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.a_feat_dirs = a_feat_dirs \
            if isinstance(a_feat_dirs, list) else [a_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.v_feat_types = v_feat_types
        self.a_feat_types = a_feat_types
        
        if max_v_l == -1:
            max_v_l = 100000000
        if max_a_l == -1:
            max_a_l = 100000000
        if max_q_l == -1:
            max_q_l = 100
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.max_a_l = max_a_l
        
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.use_audio = "audio" in ctx_mode
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.load_labels = load_labels

        self.saliency_scores_list=None

        ##  添加负责视频分块和short、long、future记忆的属性
        self.chunk_interval = chunk_interval
        self.short_memory_sample_length = short_memory_sample_length
        self.long_memory_sample_length = long_memory_sample_length
        self.future_memory_sample_length = future_memory_sample_length
        self.short_memory_stride = short_memory_stride
        self.long_memory_stride = long_memory_stride
        self.future_memory_stride = future_memory_stride
        self.load_future_memory = load_future_memory

        ##  加载数据集的模式
        self.test_mode=test_mode

        self.use_gaussian_labels = use_gaussian_labels
        self.alpha_s = alpha_s
        self.alpha_m = alpha_m
        self.alpha_e = alpha_e

        # data
        self.data = self.load_data()
        self.load_saliency_scores() 

        # 分块信息
        self.chunk_infos = self.chunk_all_videos()[0]

        if self.dset_name == 'tvsum' or self.dset_name == 'youtube_highlight':
            new_data = []
            for d in self.data:
                if d['domain'] == self.domain:
                    new_data.append(d)
            self.data = new_data

        self.use_glove = 'glove' in q_feat_dir
        if self.use_glove:
            self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
            self.vocab.itos.extend(['<unk>'])
            self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
            self.vocab.vectors = torch.cat(
                (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
            self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)
        

    #   这个后面要改
    def load_saliency_scores(self):        
        if len(self.data)==0:
            print("none data is loaded, cant load saliency scores")
            return
        elif self.saliency_scores_list==None:
            self.saliency_scores_list={}

        if self.dset_name == "qvhighlight" and "subs_train" not in self.data_path: 
            for line in self.data:
                duration_frame = int(line["duration"] * self.fps)
                all_vid_scores = np.zeros(duration_frame, dtype=float)
                for idx, scores in enumerate(line["saliency_scores"]):
                    all_vid_scores[line["relevant_clip_ids"][idx]] = np.mean(line["saliency_scores"][idx])
                self.saliency_scores_list[(line["vid"],line["qid"])]= all_vid_scores

                # print("all_vid_scores:", all_vid_scores)
                # input("Press Enter to continue...")

                # gt = line["relevant_windows"][0]
                # gt_start_frame = int(gt[0] * fps)
                # gt_end_frame = int(gt[1] * fps + fps)
                

                # gt = line["relevant_windows"][0]
                # # 加载 scores 并聚合分数
                # scores = np.array(line["saliency_scores"][:int((gt[1]-gt[0])/clip_len)])
                # agg_scores = np.sum(scores, 1)  # (#rel_clips, )

                # # 计算 GT 帧边界
                # vid=line["vid"]
                # rel_win_len=len(scores)  # 但是这里好像就没有逐帧提取的视频特征
                
                # fps = rel_win_len / (gt[1]-gt[0])
                # frame_len = int(line["duration"]*fps)
                # gt_start_frame = int(gt[0] * fps)
                # gt_end_frame = int(gt[1] * fps + fps)


                # # print("\n fps is {} \n".format(fps))
                # # print("frame start:{}, frame end:{}".format(gt_start_frame,gt_end_frame))

                # # 计算每一帧的显著性分数
                # gt_scores = np.repeat(agg_scores, fps * clip_len)
                # frame_scores = np.zeros(frame_len, dtype=int)
                # frame_scores[gt_start_frame:gt_end_frame]=gt_scores
                # #   将帧gt和gt对应的帧的分数返回
                # 
        else:
            #   加载sub_as_query分数
            for line in self.data:
                # 按照lighthosue用的数据集，feat也是2s一次采样，相当于fps也是0.5
                # 然后feat的长度就是duration/2，再向上取整（tacos等视频长度不一定是整数）
                
                # print("vid duration:", line["duration"])
                # print("len of this line's feat:", len(self._get_video_feat_by_vid(line["vid"])))
                # input("Press Enter to continue...")

                duration_frame = math.ceil(line["duration"] * self.fps)
                all_vid_scores = np.zeros(duration_frame, dtype=float)
                for windows in line["relevant_windows"]:
                    start_frame = math.ceil(windows[0] * self.fps)-1
                    end_frame = math.ceil(windows[1] * self.fps)-1
                    all_vid_scores[start_frame:end_frame] = 1
                self.saliency_scores_list[(line["vid"],line["qid"])]= all_vid_scores
                
                # print("line:", line)
                # print("all_vid_scores:", all_vid_scores)
                # input("Press Enter to continue...")

                # vid = line["vid"]
                # fps = len(self._get_video_feat_by_vid(vid))/line["duration"]
                # frame_len = int(line["duration"]*fps)
                # scores = np.zeros(frame_len, dtype=int)
                
                # if line["duration"] <= 0:
                #     raise ValueError("Video duration must be greater than 0.")
                    
                # gt = line["relevant_windows"][0]
                # # 将浮点数转换为整数
                # gt_start = int(round(gt[0]*fps))
                # gt_end = int(round(gt[1]*fps + fps))
                
                # # 确保索引在有效范围内
                # gt_start = max(0, gt_start)
                # gt_end = min(frame_len, gt_end)
                
                # scores[gt_start:gt_end] = 1
                # self.saliency_scores_list[vid] = scores


    
    def chunk_all_videos(self):
        """分块所有视频并生成软标签
        Returns:
            chunk_infos: list[dict], 每个dict包含一个chunk的完整信息
            total_length: int, 总chunk数量
        """
        chunk_infos = []
        total_length = 0
        
        for line in self.data:
            # 基本信息准备
            video_feat = self._get_video_feat_by_vid(line["vid"])
            duration_frame = math.ceil(line["duration"] * self.fps)

            # 计算目标片段位置
            gt_windows = line["relevant_windows"]
            for idx, windows in enumerate(gt_windows):
                    gt_windows[idx] = [math.ceil(windows[0] * self.fps)-1, math.ceil(windows[1] * self.fps)-1]
                

            # 确定chunk的起始位置
            if not self.test_mode:
                interval = self.chunk_interval - 1 + self.short_memory_sample_length
                offset = np.random.randint(interval)
            else:
                offset = 0
                interval = self.chunk_interval

            # 修改这里的格式，将range()放在同一行
            range_end = duration_frame + 1 - self.short_memory_sample_length
            for idx, short_memory_start in enumerate(range(offset, range_end, interval)):
                if (short_memory_start + self.short_memory_sample_length) <= duration_frame:
                    # 构建chunk基本信息
                    chunk_info = {
                        "chunk_idx": total_length,
                        "qid": line["qid"],
                        "query": line["query"],
                        "vid": line["vid"],
                        "duration_frame": duration_frame,
                        "gt_windows": gt_windows,
                        "short_memory_start": short_memory_start
                        # "start": short_memory_start,
                        # "end": short_memory_start + self.short_memory_sample_length
                    }

                    if not self.test_mode:
                        # 生成软标签
                        chunk_labels = self.get_chunk_labels(chunk_info)
                        if chunk_labels is None:
                            continue  # 跳过无效的chunk
                        
                        # 更新chunk信息
                        chunk_info.update({
                            'start_label': chunk_labels['start_label'],
                            'middle_label': chunk_labels['middle_label'],
                            'end_label': chunk_labels['end_label']
                        })

                        # print("chunk_info:", chunk_info)
                        # input("Press Enter to continue...")

                        # 生成saliency标签
                        boundary = [short_memory_start, 
                                  short_memory_start + self.short_memory_sample_length]
                                
                        try:
                            if 'qvhighlight' in self.dset_name:
                                if "subs_train" in self.data_path:
                                    # 处理多个窗口，取最大值作为最终标签
                                    all_saliency_labels = []
                                    for gt_window in chunk_info["gt_windows"]:
                                        labels = self.get_saliency_labels_sub_as_query(
                                            boundary, 
                                            gt_window,  # 传入单个窗口
                                            self.short_memory_sample_length
                                        )
                                        if labels is not None:
                                            all_saliency_labels.append(labels)
                                    
                                    # 如果没有有效标签，跳过这个chunk
                                    if not all_saliency_labels:
                                        continue
                                        
                                    # 合并所有窗口的标签（取最大值）
                                    saliency_labels = (
                                        max(l[0] for l in all_saliency_labels),  # pos_labels
                                        max(l[1] for l in all_saliency_labels),  # neg_labels
                                        np.maximum.reduce([l[2] for l in all_saliency_labels])  # all_labels
                                    )
                                else:
                                    scores_key = (chunk_info["vid"], chunk_info["qid"])
                                    if scores_key not in self.saliency_scores_list:
                                        print(f"Warning: No saliency scores found for {scores_key}")
                                        continue
                                    
                                    # 对每个gt window生成saliency label，取最大值
                                    all_saliency_labels = []
                                    for gt_window in chunk_info["gt_windows"]:
                                        try:
                                            labels = self.get_saliency_labels_all(
                                                gt_window,
                                                self.saliency_scores_list[scores_key],
                                                boundary
                                            )
                                            if labels is not None:
                                                all_saliency_labels.append(labels)
                                        except Exception as e:
                                            print(f"Error generating saliency labels for window {gt_window}: {str(e)}")
                                            continue
                                    
                                    if not all_saliency_labels:
                                        continue
                                        
                                    saliency_labels = (
                                        max(l[0] for l in all_saliency_labels),
                                        max(l[1] for l in all_saliency_labels),
                                        max(l[2] for l in all_saliency_labels)
                                    )
                                    
                            elif self.dset_name in ['charades', 'tacos', 'activitynet', 
                                                  'clotho-moment', 'unav100-subset', 'tut2017']:
                                # 同样处理多个窗口
                                all_saliency_labels = []
                                for gt_window in chunk_info["gt_windows"]:
                                    labels = self.get_saliency_labels_sub_as_query(
                                        boundary,
                                        gt_window,  # 使用正确的gt_window而不是不存在的gt
                                        self.short_memory_sample_length
                                    )
                                    if labels is not None:
                                        all_saliency_labels.append(labels)
                                
                                if not all_saliency_labels:
                                    continue
                                    
                                saliency_labels = (
                                    max(l[0] for l in all_saliency_labels),
                                    max(l[1] for l in all_saliency_labels),
                                    np.maximum.reduce([l[2] for l in all_saliency_labels])
                                )
                            else:
                                raise NotImplementedError
                            
                            if saliency_labels is None:
                                continue  # 跳过saliency标签无效的chunk
                            
                            chunk_info.update({
                                "saliency_pos_labels": saliency_labels[0],
                                "saliency_neg_labels": saliency_labels[1],
                                "saliency_all_labels": saliency_labels[2]
                            })
                            
                        except Exception as e:
                            print(f"Error generating saliency labels for vid={chunk_info['vid']}: {str(e)}")
                            continue

                    # print("chunk_info:", chunk_info)
                    # input("Press Enter to continue...") 

                    chunk_infos.append(chunk_info)
                    total_length += 1

        # 验证chunk_infos的完整性
        valid_chunk_infos = []
        for i, chunk_info in enumerate(chunk_infos):
            if not isinstance(chunk_info, dict):
                print(f"Warning: chunk_infos[{i}] is not a dictionary. Skipping.")
                continue
            
            if not self.test_mode:
                required_keys = {
                    'start_label', 'middle_label', 'end_label',
                    'saliency_pos_labels', 'saliency_neg_labels'
                }
                if not all(k in chunk_info for k in required_keys):
                    print(f"Warning: chunk_infos[{i}] missing required keys. Skipping.")
                    continue

            valid_chunk_infos.append(chunk_info)

        print(f"Generated {len(valid_chunk_infos)} valid chunks out of {total_length} total chunks")

        return valid_chunk_infos, len(valid_chunk_infos)

    def generate_gaussian_labels(self, video_length, start_idx, end_idx):
        """生成高斯软标签"""
        try:
            # 计算中间点位置
            middle_idx = (start_idx + end_idx) / 2
            duration = end_idx - start_idx
            
            # 计算标准差
            sigma_s = self.alpha_s * duration
            sigma_m = self.alpha_m * duration
            sigma_e = self.alpha_e * duration
            
            # 生成时间序列
            t = torch.arange(video_length, dtype=torch.float)
            
            # 使用 ** 运算符代替 pow 方法
            y_s = torch.exp(-((t - start_idx) ** 2) / (2 * sigma_s ** 2))
            y_m = torch.exp(-((t - middle_idx) ** 2) / (2 * sigma_m ** 2))
            y_e = torch.exp(-((t - end_idx) ** 2) / (2 * sigma_e ** 2))
            
            return {'start': y_s, 'middle': y_m, 'end': y_e}
        except Exception as e:
            print(f"Error in generate_gaussian_labels: {str(e)}")
            return None

    def get_chunk_labels(self, chunk_info):
        """获取当前chunk的标签，考虑多个GT片段
        Args:
            chunk_info: dict, 包含当前chunk的完整信息 {
                'vid': str,
                'qid': str,
                'short_memory_start': int,  # chunk开始帧
                'duration_frame': int,  # 视频总帧数
                'gt_windows': list,  # 多个GT片段的列表，每个元素是[start_frame, end_frame]
            }
        Returns:
            dict: 包含start_label, middle_label, end_label的软标签
        """
        vid, qid = chunk_info['vid'], chunk_info['qid']
        chunk_start = chunk_info['short_memory_start']
        chunk_end = chunk_start + self.short_memory_sample_length
        video_length = chunk_info['duration_frame']
        
        try:
            # 为每个GT片段生成软标签，取最大值作为最终标签
            start_labels = []
            middle_labels = []
            end_labels = []
            
            for gt_window in chunk_info['gt_windows']:
                # 生成当前GT片段的软标签
                soft_labels = self.generate_gaussian_labels(
                    video_length=video_length,
                    start_idx=gt_window[0],
                    end_idx=gt_window[1]
                )
                if soft_labels is None:
                    continue
                    
                # 截取当前chunk的部分
                start_labels.append(soft_labels['start'][chunk_start:chunk_end])
                middle_labels.append(soft_labels['middle'][chunk_start:chunk_end])
                end_labels.append(soft_labels['end'][chunk_start:chunk_end])
            
            # 如果没有有效的标签，返回None
            if not start_labels:
                return None
            
            # 取所有GT片段的最大值作为最终标签
            chunk_labels = {
                'start_label': torch.max(torch.stack(start_labels), dim=0)[0],
                'middle_label': torch.max(torch.stack(middle_labels), dim=0)[0],
                'end_label': torch.max(torch.stack(end_labels), dim=0)[0]
            }
            
            # 验证标签有效性
            if all(torch.sum(label) == 0 for label in chunk_labels.values()):
                return None
            
            return chunk_labels
            
        except Exception as e:
            print(f"Error generating labels for vid={vid}, qid={qid}: {str(e)}")
            return None

    def __getitem__(self, chunk_idx):
        """获取数据样本
        如果标签无效，返回None，在collate_fn中会被过滤掉
        """
        chunk_info = self.chunk_infos[chunk_idx]
        
        short_memory_start = chunk_info["short_memory_start"]
        short_memory_end = short_memory_start+self.short_memory_sample_length
        long_memory_start=None
        long_memory_end=None
        if self.long_memory_sample_length > 0:
            long_memory_start = short_memory_start - self.long_memory_sample_length
            long_memory_start=max(long_memory_start,0)
            long_memory_end = short_memory_start
        future_memory_start=None
        future_memory_end=None
        if self.load_future_memory and self.future_memory_sample_length > 0:
            future_memory_start = short_memory_end
            future_memory_end = short_memory_end + self.future_memory_sample_length
            future_memory_end=min(future_memory_end,self.data[chunk_info["chunk_idx"]]["duration"])

        model_inputs = dict()
        #         model_inputs["short_memory_start"]=short_memory_start
        model_inputs["short_memory_start"] = {"spans": short_memory_start}
        
        # 修改查询特征的处理
        if self.use_glove:
            query_feat = self.get_query(chunk_info["query"])
        else:
            query_feat = self._get_query_feat_by_qid(chunk_info["qid"])
        # 确保查询特征被正确添加到model_inputs中
        model_inputs["query_feat"] = query_feat
        #?# 这里按照一定使用视频写了，但是保留了音视频对齐的逻辑. 要不还得给short，long，future分别添加max_v_l参数。后续有需要再加上这个逻辑
        #   video feature
        if self.use_video:
            whole_video_feature = self._get_video_feat_by_vid(chunk_info["vid"])  # (Lv, Dv)
            #   short memory
            model_inputs["video_feat_short"]=whole_video_feature[short_memory_start:short_memory_end:self.short_memory_stride]
            ctx_l_short=len(model_inputs["video_feat_short"])
            #   long memory 
            if self.long_memory_sample_length > 0:
                model_inputs["video_feat_long"]=whole_video_feature[long_memory_start:long_memory_end:self.long_memory_stride]  
                ctx_l_long=len(model_inputs["video_feat_long"])
            #   future memory
            if self.load_future_memory and self.future_memory_sample_length > 0:
                model_inputs["video_feat_future"]=whole_video_feature[future_memory_start:future_memory_end:self.future_memory_stride] 
                ctx_l_future=len(model_inputs["video_feat_future"])
        #   audio feature
        if self.use_audio:
            whole_audio_feature = self._get_audio_feat_by_vid(chunk_info["vid"])  # (Lv, Dv)
            #   short memory
            model_inputs["audio_feat_short"]=whole_audio_feature[short_memory_start:short_memory_end:self.short_memory_stride]
            ctx_l_a=len(model_inputs["audio_feat_short"])
            if ctx_l_short < ctx_l_a:
                model_inputs["audio_feat_short"] = model_inputs["audio_feat_short"][:ctx_l_short]
            elif ctx_l_short > ctx_l_a:
                if self.use_video:
                    model_inputs["video_feat_short"] = model_inputs["video_feat_short"][:ctx_l_a] # TODO: Sometimes, audio length is not equal to video length.
                ctx_l_short = ctx_l_a
            #   long memory 
            if self.long_memory_sample_length > 0:
                model_inputs["audio_feat_long"]=whole_audio_feature[long_memory_start:long_memory_end:self.long_memory_stride]
                ctx_l_a=len(model_inputs["audio_feat_long"])
                if ctx_l_long < ctx_l_a:
                    model_inputs["audio_feat_long"] = model_inputs["audio_feat_long"][:ctx_l_long]
                elif ctx_l_long > ctx_l_a:
                    if self.use_video:
                        model_inputs["video_feat_long"] = model_inputs["video_feat_long"][:ctx_l_a] # TODO: Sometimes, audio length is not equal to video length.
                    ctx_l_long = ctx_l_a 
            #   future memory
            if self.load_future_memory and self.future_memory_sample_length > 0:
                model_inputs["audio_feat_future"]=whole_audio_feature[future_memory_start:future_memory_end:self.future_memory_stride] 
                ctx_l_a=len(model_inputs["audio_feat_future"])
                if ctx_l_long < ctx_l_a:
                    model_inputs["audio_feat_future"] = model_inputs["audio_feat_future"][:ctx_l_future]
                elif ctx_l_future > ctx_l_a:
                    if self.use_video:
                        model_inputs["video_feat_future"] = model_inputs["video_feat_future"][:ctx_l_a] # TODO: Sometimes, audio length is not equal to video length.
                    ctx_l_future = ctx_l_a 
        #?# 这里如果使用tef编码的话，需要根据模型如何使用feat调整：是拼接后使用tef编码，还是分别进行tef编码
        if self.use_tef:
            #   short
            tef_st_short = torch.arange(0, ctx_l_short, 1.0) / ctx_l_short
            tef_ed_short = tef_st_short + 1.0 / ctx_l_short
            tef_short = torch.stack([tef_st_short, tef_ed_short], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat_short"] = torch.cat(
                    [model_inputs["video_feat_short"], tef_short], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat_short"] = tef_short
            
            #   long
            if ctx_l_long > 0:  # 添加判断
                tef_st_long = torch.arange(0, ctx_l_long, 1.0) / ctx_l_long
                tef_ed_long = tef_st_long + 1.0 / ctx_l_long
                tef_long = torch.stack([tef_st_long, tef_ed_long], dim=1)
                if self.use_video:
                    model_inputs["video_feat_long"] = torch.cat(
                        [model_inputs["video_feat_long"], tef_long], dim=1)
                else:
                    model_inputs["video_feat_long"] = tef_long
            #   future
            if self.load_future_memory and self.future_memory_sample_length>0:
                tef_st_future = torch.arange(0, ctx_l_future, 1.0) / ctx_l_future
                tef_ed_future = tef_st_future + 1.0 / ctx_l_future
                tef_future = torch.stack([tef_st_future, tef_ed_future], dim=1)  # (Lv, 2)
                if self.use_video:
                    model_inputs["video_feat_future"] = torch.cat(
                        [model_inputs["video_feat_future"], tef_future], dim=1)  # (Lv, Dv+2)
                else:
                    model_inputs["video_feat_future"] = tef_future
        # Span Label
        if not self.test_mode:
            ## Short-term label
            chunk_labels = self.get_chunk_labels(chunk_info)
            if chunk_labels is not None:
                model_inputs['start_label'] = chunk_labels['start_label']
                model_inputs['middle_label'] = chunk_labels['middle_label']
                model_inputs['end_label'] = chunk_labels['end_label']
            else:
                return None  # 如果标签生成失败，返回None
            
            #   Saliency Label
            boundary=[short_memory_start,short_memory_end]
            if 'qvhighlight' in self.dset_name:
                if "subs_train" in self.data_path: # for pretraining
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(boundary,chunk_info["gt"], self.short_memory_sample_length)
                else:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_all(chunk_info["gt"],self.saliency_scores_list[chunk_info["vid"]],boundary)                        
            
            elif self.dset_name in ['charades', 'tacos', 'activitynet', 'clotho-moment', 'unav100-subset', 'tut2017']:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                    self.get_saliency_labels_sub_as_query(boundary,chunk_info["gt"], self.short_memory_sample_length)
            else:
                raise NotImplementedError
        #   或者这里返回的字典里可以直接用chunk_info
        return dict(meta=chunk_info, model_inputs=model_inputs)
    
    def load_data(self):
        datalist = load_jsonl(self.data_path)
        return datalist

    def __len__(self):
        return len(self.data)

    # def get_saliency_labels_sub_as_query(self, boundary, gt_frame_boundary, ctx_l, max_n=2):
    #     gt_st = gt_frame_boundary[0]
    #     gt_ed = gt_frame_boundary[1]

    #     st = boundary[0]
    #     ed = boundary[1]

    #     # 计算 boundary 和 gt_frame_boundary 的交集
    #     intersect_st = max(st, gt_st)
    #     intersect_ed = min(ed, gt_ed)

    #     # 如果交集有效（即 intersect_st <= intersect_ed），则将其作为 pos_pool
    #     if intersect_st <= intersect_ed:
    #         pos_pool = list(range(intersect_st, intersect_ed + 1))
    #     else:
    #         pos_pool = []

    #     # boundary 的其余部分作为 neg_pool
    #     neg_pool = list(range(st, intersect_st)) + list(range(intersect_ed + 1, ed + 1))

    #     # 从 pos_pool 中随机选择 max_n 个正样本
    #     if len(pos_pool) >= max_n:
    #         pos_clip_indices = random.sample(pos_pool, k=max_n)
    #     else:
    #         pos_clip_indices = pos_pool  # 如果 pos_pool 不足，则全部选择

    #     # 从 neg_pool 中随机选择 max_n 个负样本
    #     try:
    #         neg_clip_indices = random.sample(neg_pool, k=max_n)
    #     except:
    #         neg_clip_indices = pos_clip_indices  # 如果 neg_pool 不足，则使用 pos_clip_indices

    #     # 如果正样本或负样本不足，用 -1 填充
    #     if len(pos_clip_indices) < max_n:
    #         pos_clip_indices.extend([-1] * (max_n - len(pos_clip_indices)))
    #     if len(neg_clip_indices) < max_n:
    #         neg_clip_indices.extend([-1] * (max_n - len(neg_clip_indices)))

    #     # 生成显著性评分数组
    #     score_array = np.zeros(ctx_l)
    #     if pos_pool:
    #         score_array[pos_pool[0]:pos_pool[-1] + 1] = 1

    #     return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_sub_as_query(self, boundary, gt_frame_boundary, ctx_l, max_n=1):
        gt_st = gt_frame_boundary[0]
        gt_ed = gt_frame_boundary[1]

        st = boundary[0]
        ed = boundary[1]

        # 计算 boundary 和 gt_frame_boundary 的交集
        intersect_st = max(st, gt_st)
        intersect_ed = min(ed, gt_ed)

        # 如果交集有效（即 intersect_st <= intersect_ed），则将其作为 pos_pool
        if intersect_st <= intersect_ed:
            pos_pool = list(range(int(intersect_st), int(intersect_ed) + 1))  # 转换为整数
        else:
            pos_pool = []

        # boundary 的其余部分作为 neg_pool
        neg_pool = list(range(int(st), int(intersect_st))) + list(range(int(intersect_ed) + 1, int(ed) + 1))  # 转换为整数

        # 从 pos_pool 中随机选择 max_n 个正样本
        if len(pos_pool) >= max_n:
            pos_clip_indices = random.sample(pos_pool, k=max_n)
        else:
            pos_clip_indices = pos_pool  # 如果 pos_pool 不足，则全部选择

        # 从 neg_pool 中随机选择 max_n 个负样本
        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = pos_clip_indices  # 如果 neg_pool 不足，则使用 pos_clip_indices

        # 如果正样本或负样本不足，用 -1 填充
        if len(pos_clip_indices) < max_n:
            pos_clip_indices.extend([-1] * (max_n - len(pos_clip_indices)))
        if len(neg_clip_indices) < max_n:
            neg_clip_indices.extend([-1] * (max_n - len(neg_clip_indices)))

        # 生成显著性评分数组
        score_array = np.zeros(ctx_l)
        if pos_pool:
            score_array[pos_pool[0]:pos_pool[-1] + 1] = 1

        # print("在get_saliency_labels_sub_as_query中的计算结果是：")
        # print("saliency_pos_labels:", pos_clip_indices)
        # print("saliency_neg_labels:", neg_clip_indices)

        return pos_clip_indices, neg_clip_indices, score_array
    #   区别于lighthouse，这里可能存在正样本或者负样本为空的情况
    #   因为只对rel_clip进行了评分，而chunk可能和gt无交集，或者为gt的子集
    #   返回的分数列表是boundary对应的区间的分数，如果在数据集中没有这部分的分数（不是relevant cilp）则分数为0
    def get_saliency_labels_all(self, gt_boundary, gt_scores, boundary, max_n=1,):
        """
        根据 gt_boundary 和 boundary 的关系，选择正样本和负样本，并生成显著性分数数组。

        参数:
            gt_boundary (list): GT 的帧边界 [st, ed]。
            gt_scores (list): GT 范围内每一帧的显著性分数。
            boundary (list): 当前查询的帧边界 [st, ed]。
            ctx_l (int): 上下文长度（整个视频的帧数）。
            max_n (int): 每个类别（正样本和负样本）中最多选取的帧数。
            add_easy_negative (bool): 是否从非 GT 区域中随机选择负样本。

        返回:
            pos_clip_indices (list): 正样本的帧索引。
            neg_clip_indices (list): 负样本的帧索引。
            score_array (np.array): 显著性分数数组，长度为 ctx_l。
        """
        gt_st, gt_ed = gt_boundary
        gt_st, gt_ed = int(gt_st), int(gt_ed)
        st, ed = boundary

        # 初始化正样本和负样本
        pos_clip_indices = []
        neg_clip_indices = []

        # 计算 gt_boundary 和 boundary 的交集
        intersect_st = int(max(gt_st, st))
        intersect_ed = int(min(gt_ed, ed))

        # 情况 1: gt_boundary 和 boundary 有交集，且 boundary 不是 gt_boundary 的子集
        if intersect_st <= intersect_ed and (st < gt_st or ed > gt_ed):
            # 取交集中分数最高的 max_n 个帧作为正样本
            intersect_scores = gt_scores[intersect_st - gt_st:intersect_ed - gt_st + 1]
            intersect_indices = list(range(intersect_st, intersect_ed + 1))
            if len(intersect_scores) > 0:
                top_n_indices = np.argsort(intersect_scores)[-max_n:]
                pos_clip_indices = [intersect_indices[i] for i in top_n_indices]

            # 从 boundary - gt_boundary 的部分随机取样 max_n 个作为负样本
            neg_pool = list(range(st, gt_st)) + list(range(gt_ed + 1, ed + 1))
            if len(neg_pool) >= max_n:
                neg_clip_indices = random.sample(neg_pool, k=max_n)
            else:
                neg_clip_indices = neg_pool  # 如果数量不够，则取全部

        # 情况 2: boundary 是 gt_boundary 的子集
        elif st >= gt_st and ed <= gt_ed:
            # 取 boundary 中分数最高的 max_n 个帧作为正样本
            boundary_scores = gt_scores[st - gt_st:ed - gt_st + 1]
            boundary_indices = list(range(st, ed + 1))
            if len(boundary_scores) > 0:
                top_n_indices = np.argsort(boundary_scores)[-max_n:]
                pos_clip_indices = [boundary_indices[i] for i in top_n_indices]
            # 负样本为 None
            neg_clip_indices = []

        # 情况 3: gt_boundary 和 boundary 无交集
        else:
            # 从 boundary 中随机取样 max_n 个作为负样本
            neg_pool = list(range(st, ed + 1))
            if len(neg_pool) >= max_n:
                neg_clip_indices = random.sample(neg_pool, k=max_n)
            else:
                neg_clip_indices = neg_pool  # 如果数量不够，则取全部
            # 正样本为 None
            pos_clip_indices = []

        # 如果正样本或负样本不足，用 -1 填充
        if len(pos_clip_indices) < max_n:
            pos_clip_indices.extend([-1] * (max_n - len(pos_clip_indices)))
        if len(neg_clip_indices) < max_n:
            neg_clip_indices.extend([-1] * (max_n - len(neg_clip_indices)))

        # 生成显著性分数数组
        score_array = np.zeros(boundary[1] - boundary[0] + 1)
        if intersect_st <= intersect_ed:
            # 将交集的分数设置为 gt_scores 中对应的值
            score_array[intersect_st - st:intersect_ed - st + 1] = gt_scores[intersect_st - gt_st:intersect_ed - gt_st + 1]

        return pos_clip_indices, neg_clip_indices, score_array
    
    
    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def get_query(self, query):
        word_inds = torch.LongTensor(
            [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
        return self.embedding(word_inds)

    def _get_query_feat_by_qid(self, qid):
        if self.dset_name == 'tvsum' or self.dset_name == 'youtube_highlight':
            q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            q_feat = np.load(q_feat_path)
            return torch.from_numpy(q_feat['token']) if self.dset_name == 'tvsum' else torch.from_numpy(q_feat['last_hidden_state'])

        else:
            if self.dset_name == 'tacos':
                q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            elif "subs_train" in self.data_path: # for pretrain
                vid = "_".join(qid.split("_")[:-1])
                subid = qid.split("_")[-1]
                q_feat_path = join(self.q_feat_dir, f"{vid}/{subid}.npz")
            else:
                q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")

            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            q_feat = l2_normalize_np_array(q_feat)
            return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            if self.dset_name == 'tvsum' and 'i3d' in _feat_dir:
                rgb_path = join(_feat_dir, f"{vid}_rgb.npy")
                opt_path = join(_feat_dir, f"{vid}_opt.npy")
                rgb_feat = np.load(rgb_path)[:self.max_v_l].astype(np.float32)
                opt_feat = np.load(opt_path)[:self.max_v_l].astype(np.float32)
                _feat = np.concatenate([rgb_feat, opt_feat], axis=-1)
                _feat = l2_normalize_np_array(_feat) # normalize?
                v_feat_list.append(_feat)
            else:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
        
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)
    
    def _get_audio_feat_by_vid(self, vid):
        a_feat_list = []
        for _feat_dir in self.a_feat_dirs:
            if self.dset_name == 'qvhighlight' or self.dset_name == 'qvhighlight_pretrain':
                if self.a_feat_types == "pann":
                    _feat_path = join(_feat_dir, f"{vid}.npy")
                    _feat = np.load(_feat_path)[:self.max_a_l].astype(np.float32)
                else:
                    raise NotImplementedError
                _feat = l2_normalize_np_array(_feat) # normalize?
                a_feat_list.append(_feat)
            elif self.dset_name in ['clotho-moment', 'unav100-subset', 'tut2017']:
                if self.a_feat_types == "clap":
                    _feat_path = join(_feat_dir, f"{vid}.npz")
                    _feat = np.load(_feat_path)["features"][:self.max_a_l].astype(np.float32)
                else:
                    raise NotImplementedError
                _feat = l2_normalize_np_array(_feat) # normalize?
                a_feat_list.append(_feat)
            else:
                raise NotImplementedError
        
        # some features are slightly longer than the others
        min_len = min([len(e) for e in a_feat_list])
        a_feat_list = [e[:min_len] for e in a_feat_list]
        a_feat = np.concatenate(a_feat_list, axis=1)
        return torch.from_numpy(a_feat)  # (Lv, D)

def start_end_collate_ol(batch):
    """处理batch数据，过滤掉无效样本"""
    # 过滤None值
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    # 分离meta和model_inputs
    metas, model_inputs_list = [], []
    for b in batch:
        metas.append(b['meta'])
        model_inputs_list.append(b['model_inputs'])
    
    # 检查所有必需的键是否存在
    required_keys = {
        'start_label', 'middle_label', 'end_label',
        'saliency_pos_labels', 'saliency_neg_labels', 'saliency_all_labels'
    }
    valid_indices = []
    for idx, inputs in enumerate(model_inputs_list):
        if all(k in inputs for k in required_keys):
            valid_indices.append(idx)
    
    # 只保留有效样本
    metas = [metas[i] for i in valid_indices]
    model_inputs_list = [model_inputs_list[i] for i in valid_indices]
    
    if len(valid_indices) == 0:
        return None
        
    # 继续处理有效样本
    batched_model_inputs = {}
    for key in model_inputs_list[0].keys():
        if isinstance(model_inputs_list[0][key], torch.Tensor):
            batched_model_inputs[key] = torch.stack([
                inputs[key] for inputs in model_inputs_list
            ])
        elif isinstance(model_inputs_list[0][key], dict):
            # 处理嵌套字典，如short_memory_start
            batched_model_inputs[key] = {
                k: torch.stack([inputs[key][k] for inputs in model_inputs_list])
                if isinstance(model_inputs_list[0][key][k], torch.Tensor)
                else [inputs[key][k] for inputs in model_inputs_list]
                for k in model_inputs_list[0][key].keys()
            }
        else:
            batched_model_inputs[key] = [inputs[key] for inputs in model_inputs_list]
    
    return metas, batched_model_inputs

def prepare_batch_inputs(metas, batched_model_inputs, device, non_blocking=False):
    """准备模型输入
    Args:
        metas: list of dict, 每个样本的元信息
        batched_model_inputs: collate_fn输出的模型输入
        device: 计算设备
        non_blocking: 是否使用非阻塞传输
    Returns:
        model_inputs: dict, 模型输入
        targets: dict, 目标标签
    """
    model_inputs = {}
    
    # 1. 处理视频特征
    if 'video_feat_short' in batched_model_inputs:
        if 'video_feat_long' in batched_model_inputs:
            video_feats = [
                batched_model_inputs['video_feat_long'],
                batched_model_inputs['video_feat_short']
            ]
            if 'video_feat_future' in batched_model_inputs:
                video_feats.append(batched_model_inputs['video_feat_future'])
            src_vid = torch.cat(video_feats, dim=1)
        else:
            src_vid = batched_model_inputs['video_feat_short']
    
    # 2. 处理音频特征（如果有）
    if 'audio_feat_short' in batched_model_inputs:
        if 'audio_feat_long' in batched_model_inputs:
            audio_feats = [
                batched_model_inputs['audio_feat_long'],
                batched_model_inputs['audio_feat_short']
            ]
            if 'audio_feat_future' in batched_model_inputs:
                audio_feats.append(batched_model_inputs['audio_feat_future'])
            src_aud = torch.cat(audio_feats, dim=1)
        else:
            src_aud = batched_model_inputs['audio_feat_short']
        model_inputs['src_aud'] = src_aud.to(device, non_blocking=non_blocking)
    
    # 3. 基本输入处理
    model_inputs.update({
        'src_vid': src_vid.to(device, non_blocking=non_blocking),
        'src_txt': batched_model_inputs['query_feat'].to(device, non_blocking=non_blocking),
        'src_vid_mask': torch.ones(src_vid.shape[:2], dtype=torch.long, device=device),
        'src_txt_mask': torch.ones(batched_model_inputs['query_feat'].shape[:2], 
                                 dtype=torch.long, device=device)
    })
    
    # 4. 处理memory长度信息
    model_inputs['memory_len'] = [
        batched_model_inputs.get('video_feat_long', torch.tensor([])).shape[1],
        batched_model_inputs['video_feat_short'].shape[1],
        batched_model_inputs.get('video_feat_future', torch.tensor([])).shape[1]
    ]
    
    # 5. 处理标签
    targets = {}
    label_keys = [
        'start_label', 'middle_label', 'end_label',
        'saliency_pos_labels', 'saliency_neg_labels', 'saliency_all_labels'
    ]
    for key in label_keys:
        if key in batched_model_inputs:
            targets[key] = batched_model_inputs[key].to(device, non_blocking=non_blocking)
    
    # 6. 添加meta信息（如果需要）
    targets['meta'] = metas
    
    return model_inputs, targets