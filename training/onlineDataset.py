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
                 future_memory_stride=1, load_future_memory=False,test_mode=False):
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

        clip_len=2  #qv数据集定义的clip就是2s
        if self.dset_name == "qvhighlight" and "subs_train" not in self.data_path: 
            for line in self.data:
                gt = line["relevant_windows"][0]
                # 加载 scores 并聚合分数
                scores = np.array(line["saliency_scores"][:int((gt[1]-gt[0])/clip_len)])
                agg_scores = np.sum(scores, 1)  # (#rel_clips, )

                # 计算 GT 帧边界
                vid=line["vid"]
                rel_win_len=len(scores)  # 但是这里好像就没有逐帧提取的视频特征
                
                fps = rel_win_len / (gt[1]-gt[0])
                frame_len = int(line["duration"]*fps)
                gt_start_frame = int(gt[0] * fps)
                gt_end_frame = int(gt[1] * fps + fps)


                # print("\n fps is {} \n".format(fps))
                # print("frame start:{}, frame end:{}".format(gt_start_frame,gt_end_frame))

                # 计算每一帧的显著性分数
                gt_scores = np.repeat(agg_scores, fps * clip_len)
                frame_scores = np.zeros(frame_len, dtype=int)
                frame_scores[gt_start_frame:gt_end_frame]=gt_scores
                #   将帧gt和gt对应的帧的分数返回
                self.saliency_scores_list[vid]= frame_scores
        else:
            #   加载sub_as_query分数
            for line in self.data:
                vid = line["vid"]
                fps = len(self._get_video_feat_by_vid(vid))/line["duration"]
                frame_len = int(line["duration"]*fps)
                scores = np.zeros(frame_len, dtype=int)
                
                if line["duration"] <= 0:
                    raise ValueError("Video duration must be greater than 0.")
                    
                gt = line["relevant_windows"][0]
                # 将浮点数转换为整数
                gt_start = int(round(gt[0]*fps))
                gt_end = int(round(gt[1]*fps + fps))
                
                # 确保索引在有效范围内
                gt_start = max(0, gt_start)
                gt_end = min(frame_len, gt_end)
                
                scores[gt_start:gt_end] = 1
                self.saliency_scores_list[vid] = scores

            temp_frame_len = len((self._get_video_feat_by_vid(self.data[0]["vid"])))
            fps = temp_frame_len/self.data[0]["duration"]
            print("loaded saliency scores, fps is {}".format(fps))


    def chunk_all_videos(self):
        chunk_infos = []
        total_length = 0
        for line in self.data:
            duration=line["duration"]
            duration_frame=len(self._get_video_feat_by_vid(line["vid"]))
            if not self.test_mode:

                # print("在chunk中加载标签了")

                interval = self.chunk_interval - 1 + self.short_memory_sample_length    #intervals for starts of 2 adjacent chunk
                offset = np.random.randint(interval)
            else:
                offset = 0
                interval = self.chunk_interval
            for idx, short_memory_start in enumerate(
                 range(offset, 
                       duration_frame+1-self.short_memory_sample_length, 
                       interval)):
                if (short_memory_start+self.short_memory_sample_length)<=duration_frame:
                    #   lighthouse里以s的窗口应该是左右都闭合的
                    gt = list(line["relevant_windows"][0]) 
                    fps = duration_frame / duration
                    gt[0] = gt[0] * fps
                    gt[1] = gt[1] * fps + fps - 1
                    chunk_info={"anno_id": idx,
                                "qid":line["qid"],
                                "query":line["query"],
                                "vid":line["vid"],
                                "duration_frame":duration_frame,
                                "gt":gt,
                                "short_memory_start":short_memory_start,}
                    
                    # Span Label
                    if not self.test_mode:
                        ## Short-term label
                        short_start_label, short_end_label, short_semantic_label = \
                            self.get_chunk_labels([short_memory_start,short_memory_start+self.short_memory_sample_length], 
                                                chunk_info['gt'], self.short_memory_stride)
                        
                        # print("看一下get_chunk_labels函数返回的是咋样的\n start:{}\n mid:{}\n end:{}".format(short_start_label,short_semantic_label,short_end_label))

                        chunk_info['start_label'] = short_start_label
                        chunk_info['end_label'] = short_end_label
                        chunk_info['semantic_label'] = short_semantic_label 

                        # print("看一下chunk后的分块里是什么内容\n start:{}\n mid:{}\n end:{}".format(
                        #     chunk_info['start_label'],
                        #     chunk_info['semantic_label'],
                        #     chunk_info['end_label'],))

                        
                        #   Saliency Label
                        boundary=[short_memory_start,short_memory_start+self.short_memory_sample_length]
                        if 'qvhighlight' in self.dset_name:
                            if "subs_train" in self.data_path: # for pretraining
                                chunk_info["saliency_pos_labels"], chunk_info["saliency_neg_labels"], chunk_info["saliency_all_labels"] = \
                                    self.get_saliency_labels_sub_as_query(boundary,chunk_info["gt"], self.short_memory_sample_length)
                            else:
                                chunk_info["saliency_pos_labels"], chunk_info["saliency_neg_labels"], chunk_info["saliency_all_labels"] = \
                                    self.get_saliency_labels_all(chunk_info["gt"],self.saliency_scores_list[chunk_info["vid"]],boundary)                        
                        
                        elif self.dset_name in ['charades', 'tacos', 'activitynet', 'clotho-moment', 'unav100-subset', 'tut2017']:
                            chunk_info["saliency_pos_labels"], chunk_info["saliency_neg_labels"], chunk_info["saliency_all_labels"] = \
                                self.get_saliency_labels_sub_as_query(boundary,chunk_info["gt"], self.short_memory_sample_length)
                        else:
                            raise NotImplementedError

                    chunk_infos.append(chunk_info)
                    total_length+=1


        # 调试：检查 chunk_infos 的结构
        for i, chunk_info in enumerate(chunk_infos):
            if not isinstance(chunk_info, dict):
                print(f"Error: chunk_infos[{i}] is not a dictionary. Value: {chunk_info}")
                raise TypeError(f"chunk_infos[{i}] is not a dictionary")
        
        #   print some informatoin
        print("after chunk_all_videls, length of chunk_infos:{}".format(len(chunk_infos)))

        return chunk_infos,total_length
    # 对给定的chunk，获取对应的片段的st时间戳、ed时间戳、语义（整个片段）在视频特征中的位置 
    def get_chunk_labels(self, boundary, gt_boundary, stride):
        """修改后的标签生成函数，确保标签互斥"""
        s, e = boundary
        mem_len = e - s
        s_gt, e_gt = gt_boundary
        s_gt, e_gt = round(s_gt), round(e_gt)
        
        # 初始化标签
        s_label = np.zeros(mem_len)
        e_label = np.zeros(mem_len)
        se_label = np.zeros(mem_len)
        
        # 计算相对位置
        diff_s = s_gt - s
        diff_e = e_gt - s
        
        # 设置起始帧标签
        if diff_s >= 0 and diff_s < mem_len:
            s_label[diff_s] = 1
        
        # 设置结束帧标签
        if diff_e >= 0 and diff_e < mem_len:
            e_label[diff_e] = 1
        
        # 设置语义相关帧标签 - 排除起始帧和结束帧
        diff_s = max(0, diff_s)
        diff_e = min(mem_len, diff_e)
        
        if diff_s < mem_len and diff_e >= diff_s:
            # 设置中间帧（排除起始帧和结束帧）
            start_idx = diff_s + 1  # 排除起始帧
            end_idx = diff_e  # 排除结束帧（因为Python切片是左闭右开的）
            
            if start_idx < end_idx:  # 确保有中间帧
                se_label[start_idx:end_idx] = 1
        
        # 如果stride!=1，作下采样
        if stride != 1:
            s_label = s_label[::stride]
            e_label = e_label[::stride]
            se_label = se_label[::stride]
        
        return s_label, e_label, se_label

    def __getitem__(self,chunk_idx):
        #   基本信息
        chunk_info = self.chunk_infos[chunk_idx]
        # vid = chunk_info["vid"]
        # qid=chunk_info["qid"]
        #    
        short_memory_start = chunk_info["short_memory_start"]
        short_memory_end = short_memory_start+self.short_memory_sample_length
        long_memory_start=None
        long_memory_end=None
        if self.load_future_memory and self.long_memory_sample_length > 0:
            long_memory_start = short_memory_start - self.long_memory_sample_length
            long_memory_start=max(long_memory_start,0)
            long_memory_end = short_memory_start
        future_memory_start=None
        future_memory_end=None
        if self.load_future_memory and self.future_memory_sample_length > 0:
            future_memory_start = short_memory_end
            future_memory_end = short_memory_end + self.future_memory_sample_length
            future_memory_end=min(future_memory_end,self.data[chunk_info["anno_id"]]["duration"])

        model_inputs = dict()
        model_inputs["short_memory_start"]=short_memory_start
        #   query feature
        if self.use_glove:
            model_inputs["query_feat"] = self.get_query(chunk_info["query"])
        else:
            model_inputs["query_feat"] = self._get_query_feat_by_qid(chunk_info["qid"])
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
            tef_st_long = torch.arange(0, ctx_l_long, 1.0) / ctx_l_long
            tef_ed_long = tef_st_long + 1.0 / ctx_l_long
            tef_long = torch.stack([tef_st_long, tef_ed_long], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat_long"] = torch.cat(
                    [model_inputs["video_feat_long"], tef_long], dim=1)  # (Lv, Dv+2)
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

            # print("get_item 中加载start_label等了")

            ## Short-term label
            short_start_label, short_end_label, short_semantic_label = \
                self.get_chunk_labels([short_memory_start,short_memory_start+self.short_memory_sample_length], 
                                       chunk_info['gt'], self.short_memory_stride)
            model_inputs['start_label'] = short_start_label
            model_inputs['end_label'] = short_end_label
            model_inputs['semantic_label'] = short_semantic_label
            ## Future anticapation label
            if self.load_future_memory and self.future_memory_sample_length > 0:
                future_start_label, future_end_label, future_semantic_label = \
                    self.get_chunk_labels([future_memory_start,future_memory_start+self.future_memory_sample_length], 
                                           chunk_info['gt'], self.future_memory_stride)
                model_inputs['future_start_label'] = future_start_label
                model_inputs['future_end_label'] = future_end_label
                model_inputs['future_semantic_label'] = future_semantic_label
        
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
    # 提取元数据
    batch_meta = [e["meta"] for e in batch]

    # 获取 model_inputs 的所有键
    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()

    # 遍历每个键，根据键的类型进行不同的处理
    for k in model_inputs_keys:
        if k == "short_memory_start":
            # batched_data[k] = torch.tensor([e["model_inputs"][k] for e in batch], dtype=torch.long)
            batched_data[k] = [dict(short_memory_start=e["model_inputs"][k]) for e in batch]
            continue
        if k == "start_label":
            batched_data[k] = [dict(start_labels=e["model_inputs"][k]) for e in batch]
            continue
        if k == "end_label":
            batched_data[k] = [dict(end_labels=e["model_inputs"][k]) for e in batch]
            continue
        if k == "semantic_label":
            batched_data[k] = [dict(semantic_labels=e["model_inputs"][k]) for e in batch]
            continue
        if k == "future_start_label":
            batched_data[k] = [dict(future_start_labels=e["model_inputs"][k]) for e in batch]
            continue
        if k == "future_end_label":
            batched_data[k] = [dict(future_end_labels=e["model_inputs"][k]) for e in batch]
            continue
        if k == "future_semantic_label":
            batched_data[k] = [dict(future_semantic_labels=e["model_inputs"][k]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            # 跳过空样本
            labels = [e["model_inputs"][k] for e in batch if len(e["model_inputs"][k]) > 0]
            if len(labels) > 0:  # 如果有非空样本
                batched_data[k] = torch.LongTensor(labels)
            continue
        if k == "saliency_all_labels":
            # 跳过空样本
            saliency_labels = [e["model_inputs"][k] for e in batch if len(e["model_inputs"][k]) > 0]
            if len(saliency_labels) > 0:  # 如果有非空样本
                pad_data, mask_data = pad_sequences_1d(saliency_labels, dtype=np.float32, fixed_length=None)
                batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue

        # 默认处理其他特征（填充变长序列）
        # 跳过空样本
        valid_data = [e["model_inputs"][k] for e in batch if len(e["model_inputs"][k]) > 0]
        if len(valid_data) > 0:  # 如果有非空样本
            batched_data[k] = pad_sequences_1d(valid_data, dtype=torch.float32, fixed_length=None)

    return batch_meta, batched_data

def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):

    #   看一下输入的batched_model_inputs都有什么内容
    #   这里的batched_model_inputs里面的label好像也正常
    # print("see batched_model_inputs contetn:{}".format(batched_model_inputs["semantic_label"]))

    # 初始化 memory_len
    memory_len = [0, 0, 0]  # [long_memory_sample_length, short_memory_sample_length, future_memory_sample_length]

    # 拼接 src_vid 和 src_vid_mask
    src_vid_list = []
    src_vid_mask_list = []

    # 处理 long memory
    if "video_feat_long" in batched_model_inputs:
        src_vid_list.append(batched_model_inputs["video_feat_long"][0].to(device, non_blocking=non_blocking))
        src_vid_mask_list.append(batched_model_inputs["video_feat_long"][1].to(device, non_blocking=non_blocking))
        memory_len[0] = batched_model_inputs["video_feat_long"][0].shape[1]  # 获取 long memory 的长度
    else:
        memory_len[0] = 0  # 如果没有 long memory，长度为 0

    # 处理 short memory
    if "video_feat_short" in batched_model_inputs:
        src_vid_list.append(batched_model_inputs["video_feat_short"][0].to(device, non_blocking=non_blocking))
        src_vid_mask_list.append(batched_model_inputs["video_feat_short"][1].to(device, non_blocking=non_blocking))
        memory_len[1] = batched_model_inputs["video_feat_short"][0].shape[1]  # 获取 short memory 的长度
    else:
        memory_len[1] = 0  # 如果没有 short memory，长度为 0

    # 处理 future memory
    if "video_feat_future" in batched_model_inputs:
        src_vid_list.append(batched_model_inputs["video_feat_future"][0].to(device, non_blocking=non_blocking))
        src_vid_mask_list.append(batched_model_inputs["video_feat_future"][1].to(device, non_blocking=non_blocking))
        memory_len[2] = batched_model_inputs["video_feat_future"][0].shape[1]  # 获取 future memory 的长度
    else:
        memory_len[2] = 0  # 如果没有 future memory，长度为 0

    # 按顺序拼接 src_vid 和 src_vid_mask
    src_vid = torch.cat(src_vid_list, dim=1) if src_vid_list else torch.tensor([]).to(device)
    src_vid_mask = torch.cat(src_vid_mask_list, dim=1) if src_vid_mask_list else torch.tensor([]).to(device)

    # 构建 model_inputs
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=src_vid,
        src_vid_mask=src_vid_mask,
        memory_len=memory_len,  # 添加 memory_len 参数
    )

    # 处理音频特征（如果有）
    if "audio_feat_short" in batched_model_inputs:
        src_aud_list = []
        src_aud_mask_list = []

        # 处理 long memory 音频特征
        if "audio_feat_long" in batched_model_inputs:
            src_aud_list.append(batched_model_inputs["audio_feat_long"][0].to(device, non_blocking=non_blocking))
            src_aud_mask_list.append(batched_model_inputs["audio_feat_long"][1].to(device, non_blocking=non_blocking))

        # 处理 short memory 音频特征
        src_aud_list.append(batched_model_inputs["audio_feat_short"][0].to(device, non_blocking=non_blocking))
        src_aud_mask_list.append(batched_model_inputs["audio_feat_short"][1].to(device, non_blocking=non_blocking))

        # 处理 future memory 音频特征
        if "audio_feat_future" in batched_model_inputs:
            src_aud_list.append(batched_model_inputs["audio_feat_future"][0].to(device, non_blocking=non_blocking))
            src_aud_mask_list.append(batched_model_inputs["audio_feat_future"][1].to(device, non_blocking=non_blocking))

        # 按顺序拼接 src_aud 和 src_aud_mask
        src_aud = torch.cat(src_aud_list, dim=1) if src_aud_list else torch.tensor([]).to(device)
        src_aud_mask = torch.cat(src_aud_mask_list, dim=1) if src_aud_mask_list else torch.tensor([]).to(device)

        model_inputs["src_aud"] = src_aud
        model_inputs["src_aud_mask"] = src_aud_mask

    # 构建 targets
    targets = {}
    if "short_memory_start" in batched_model_inputs:
        targets["short_memory_start"] = [
            dict(spans=e["short_memory_start"])
            for e in batched_model_inputs["short_memory_start"]
            if "short_memory_start" in e  # 确保每个元素都有 "short_memory_start"
        ]
    
    if "start_label" in batched_model_inputs:
        targets["start_label"] = [
            # dict(spans=e["start_labels"].to(device, non_blocking=non_blocking))
            dict(spans=e["start_labels"])
            for e in batched_model_inputs["start_label"]
            if "start_labels" in e  # 确保每个元素都有 "start_label"
        ]

    if "end_label" in batched_model_inputs:
        targets["end_label"] = [
            # dict(spans=e["end_labels"].to(device, non_blocking=non_blocking))
            dict(spans=e["end_labels"])
            for e in batched_model_inputs["end_label"]
            if "end_labels" in e  # 确保每个元素都有 "end_label"
        ]

    if "semantic_label" in batched_model_inputs:
        targets["semantic_label"] = [
            # dict(spans=e["semantic_labels"].to(device, non_blocking=non_blocking))
            dict(spans=e["semantic_labels"])
            for e in batched_model_inputs["semantic_label"]
            if "semantic_labels" in e  # 确保每个元素都有 "semantic_label"
        ]

    if "future_start_label" in batched_model_inputs:
        targets["future_start_label"] = [
            # dict(spans=e["future_start_labels"].to(device, non_blocking=non_blocking))
            dict(spans=e["future_start_labels"])
            for e in batched_model_inputs["future_start_label"]
            if "spans" in e  # 确保每个元素都有 "future_start_label"
        ]

    if "future_end_label" in batched_model_inputs:
        targets["future_end_label"] = [
            # dict(spans=e["future_end_labels"].to(device, non_blocking=non_blocking))
            dict(spans=e["future_end_labels"])
            for e in batched_model_inputs["future_end_label"]
            if "spans" in e  # 确保每个元素都有 "future_end_label"
        ]

    if "future_semantic_label" in batched_model_inputs:
        targets["future_semantic_label"] = [
            # dict(spans=e["future_semantic_labels"].to(device, non_blocking=non_blocking))
            dict(spans=e["future_semantic_labels"])
            for e in batched_model_inputs["future_semantic_label"]
            if "spans" in e  # 确保每个元素都有 "future_semantic_label"
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        # 过滤掉没有正样本或负样本的样本
        valid_indices = [
            i for i in range(len(batched_model_inputs["saliency_pos_labels"]))
            if len(batched_model_inputs["saliency_pos_labels"][i]) > 0 and len(batched_model_inputs["saliency_neg_labels"][i]) > 0
        ]

        # print("saliency_pos_labels:", batched_model_inputs["saliency_pos_labels"])
        # print("saliency_neg_labels:", batched_model_inputs["saliency_neg_labels"])

        if len(valid_indices) > 0:
            # 只保留有效的样本
            saliency_pos_labels = torch.tensor(
                [batched_model_inputs["saliency_pos_labels"][i] for i in valid_indices],
                dtype=torch.long
            )
            saliency_neg_labels = torch.tensor(
                [batched_model_inputs["saliency_neg_labels"][i] for i in valid_indices],
                dtype=torch.long
            )

            # 确保形状一致
            if saliency_pos_labels.shape != saliency_neg_labels.shape:
                raise ValueError("saliency_pos_labels and saliency_neg_labels must have the same shape")

            targets["saliency_pos_labels"] = saliency_pos_labels.to(device, non_blocking=non_blocking)
            targets["saliency_neg_labels"] = saliency_neg_labels.to(device, non_blocking=non_blocking)
        else:
            # 如果没有有效样本，跳过 saliency score 的训练
            targets["saliency_pos_labels"] = torch.tensor([], dtype=torch.long).to(device)
            targets["saliency_neg_labels"] = torch.tensor([], dtype=torch.long).to(device)
    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)

    # print("in prepare batch inputs, content o target is {}".format(targets))

    targets = None if len(targets) == 0 else targets

    return model_inputs, targets