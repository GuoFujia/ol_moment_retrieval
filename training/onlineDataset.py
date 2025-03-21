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

class StartEndDataset(Dataset):
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17, 19, 20],
      "relevant_windows": [[26, 36], [38, 42]]
    }
    """
    def __init__(self, dset_name, domain, data_path, v_feat_dirs, a_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state", v_feat_types="clip", a_feat_types="pann", 
                 max_q_l=32, max_v_l=75, max_a_l=75, ctx_mode="video", clip_len=2,
                 max_windows=5, span_loss_type="l1", load_labels=True,
                 chunk_interval=1, short_memory_sample_length=8, long_memory_sample_length=16,
                 future_memory_sample_length=8, short_memory_stride=1, long_memory_stride=1,
                 future_memory_stride=1, load_future_memory=False,test_mode=False,
                 use_gaussian_labels=True, alpha_s=0.25, alpha_m=0.21, alpha_e=0.25,
                 pos_expansion_ratio=1., neg_pos_ratio=3,):
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

        #   加载chunk sample_flag用到的参数
        self.neg_pos_ratio = neg_pos_ratio
        self.pos_expansion_ratio = pos_expansion_ratio
        
        # data
        self.data = self.load_data()

        # 如果是activitynet或者tacos数据集，则均匀采样部分视频样本来降低内存消耗，并减少训练时间  
        if self.dset_name == "activitynet":
            self.data = self.data[:len(self.data)//4]
        elif self.dset_name == "tacos" and self.test_mode:
            self.data = self.data[:len(self.data)//4]

        self.load_saliency_scores() 

        # 分块信息
        self.chunk_infos = self.chunk_all_videos()[0]
        self.use_glove = 'glove' in q_feat_dir
        if self.use_glove:
            self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
            self.vocab.itos.extend(['<unk>'])
            self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
            self.vocab.vectors = torch.cat(
                (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
            self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)
        
    def load_saliency_scores(self):        
        if len(self.data)==0:
            print("none data is loaded, cant load saliency scores")
            return
        elif self.saliency_scores_list==None:
            self.saliency_scores_list={}

        if self.dset_name == "qvhighlight" and "subs_train" not in self.data_path: 
            for line in self.data:
                # duration_frame = int(line["duration"] * self.fps)
                duration_frame = len(self._get_video_feat_by_vid(line["vid"]))
                all_vid_scores = np.zeros(duration_frame, dtype=float)
                for idx, scores in enumerate(line["saliency_scores"]):
                    if line["relevant_clip_ids"][idx] < duration_frame:
                        all_vid_scores[line["relevant_clip_ids"][idx]] = np.mean(scores)
                self.saliency_scores_list[(line["vid"],line["qid"])]= all_vid_scores
        else:
            #   加载sub_as_query分数
            for line in self.data:
                # 按照lighthosue用的数据集，feat也是2s一次采样，相当于fps也是0.5
                # 然后feat的长度就是duration/2，再向上取整（tacos等视频长度不一定是整数）

                # duration_frame = math.ceil(line["duration"] * self.fps)
                duration_frame = len(self._get_video_feat_by_vid(line["vid"]))
                all_vid_scores = np.zeros(duration_frame, dtype=float)
                for windows in line["relevant_windows"]:
                    start_frame = math.floor(windows[0] * self.fps)
                    end_frame = math.floor(windows[1] * self.fps)
                    all_vid_scores[start_frame:end_frame] = 1
                self.saliency_scores_list[(line["vid"],line["qid"])]= all_vid_scores

    def chunk_all_videos(self):
        """分块所有视频并生成软标签
        Returns:
            chunk_infos: list[dict], 每个dict包含一个chunk的完整信息
            total_length: int, 总chunk数量
        """
        chunk_infos = []
        total_length = 0

        line_cnt = 0

        # 使用 tqdm 包装 self.data，显示进度条
        for line in tqdm(self.data, desc="Processing videos", unit="video"):
            line_cnt += 1

            # 基本信息准备
            video_feat = self._get_video_feat_by_vid(line["vid"])
            duration_frame = len(video_feat)

            # 计算目标片段位置
            gt_windows = line["relevant_windows"]
            for idx, windows in enumerate(gt_windows):
                gt_windows[idx] = [math.floor(windows[0] * self.fps), math.floor(windows[1] * self.fps)]
            
            # 确定chunk的起始位置
            if not self.test_mode:
                interval = self.chunk_interval - 1 + self.short_memory_sample_length
                # interval = self.chunk_interval - 1 + self.short_memory_sample_length + 4
                offset = np.random.randint(interval)
                while(duration_frame - offset <= 0):
                    offset = np.random.randint(interval)
            else:
                offset = 0
                # interval = self.chunk_interval
                interval = 2

            # 计算正负样本采样比例
            if not self.test_mode and self.neg_pos_ratio:
                # 计算正样本的帧范围
                pos_chunk_cnt = sum([e - s for s, e in gt_windows])  # 所有GT片段的总帧数
                alpha_pos, alpha_neg = 1 + self.pos_expansion_ratio, 1 - self.pos_expansion_ratio

                # 扩展正样本范围
                expanded_framestamps = []
                for s, e in gt_windows:
                    expanded_framestamps.append([
                        max(0, int((alpha_pos * s + alpha_neg * e) / 2)),
                        min(duration_frame, int((alpha_neg * s + alpha_pos * e) / 2))
                    ])

                # 计算负样本的帧数
                out_neg_chunk_cnt = (self.neg_pos_ratio - self.pos_expansion_ratio + 1) * pos_chunk_cnt

                # 计算理论负样本数量
                max_possible_neg_cnt = duration_frame - offset - sum(e - s for s, e in expanded_framestamps)

                # 确保计算出的负样本数量不会超过实际可能的数量
                out_neg_chunk_cnt = min(out_neg_chunk_cnt, max_possible_neg_cnt)
                
                try:
                    if(duration_frame - offset - self.pos_expansion_ratio * pos_chunk_cnt <= 0):
                        out_sample_ratio = 0
                    else:
                        out_sample_ratio = min(
                            1.0,
                            out_neg_chunk_cnt / (duration_frame - offset - self.pos_expansion_ratio * pos_chunk_cnt)
                        )
                except Exception as e:
                    print("some info: ")
                    print("out_neg_chunk_cnt: ", out_neg_chunk_cnt)
                    print("pos_chunk_cnt: ", pos_chunk_cnt)
                    print("duration_frame: ", duration_frame)
                    print("pos_expansion_ratio: ", self.pos_expansion_ratio)
                    print("offset: ", offset)

                    print("line: ",line)
                    print("exception: ",e)
                    input("^^^^^^^^^^^^^^^^^^^^^")

                # 生成采样标志
                sample_score = np.random.uniform(.0, 1.0, duration_frame - offset)
                sample_flag = (sample_score <= out_sample_ratio)

                # 强制将扩展后的正样本范围内的帧标记为 True
                for expanded_range in expanded_framestamps:
                    sample_flag[expanded_range[0]:expanded_range[1]] = True

                # 根据间隔下采样
                sample_flag = sample_flag[::interval]
            else:
                # 如果没有设置负样本比例，则默认所有帧都会被选中
                sample_flag = np.ones(max(1, duration_frame - offset) // interval + 1, dtype=bool)

            # 开始采样
            range_end = duration_frame + 1 - self.short_memory_sample_length
            for idx, short_memory_start in enumerate(range(offset, range_end, interval)):
                if (short_memory_start + self.short_memory_sample_length) <= duration_frame:
                    # 检查当前chunk是否被选中
                    if not sample_flag[idx]:
                        continue  # 跳过未被选中的chunk

                    # 构建chunk基本信息
                    chunk_info = {
                        "chunk_idx": total_length,
                        "qid": line["qid"],
                        "query": line["query"],
                        "vid": line["vid"],
                        "duration_frame": duration_frame,
                        "gt_windows": gt_windows,
                        "short_memory_start": short_memory_start
                    }

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

                    # 生成saliency标签
                    boundary = [short_memory_start, short_memory_start + self.short_memory_sample_length]
                    if 'qvhighlight' in self.dset_name and "sub_train" not in self.dset_name:
                        scores_key = (chunk_info["vid"], chunk_info["qid"])
                        if scores_key not in self.saliency_scores_list:
                            continue
                        
                        saliency_labels = self.get_saliency_labels_all(
                            chunk_info["gt_windows"],  # 直接传入所有 GT 窗口
                            self.saliency_scores_list[scores_key],
                            boundary
                        )
                    elif self.dset_name in ['charades', 'tacos', 'activitynet', 
                                            'clotho-moment', 'unav100-subset', 'tut2017']:
                        saliency_labels = self.get_saliency_labels_sub_as_query(
                            boundary,
                            chunk_info["gt_windows"],  # 直接传入所有 GT 窗口
                            self.short_memory_sample_length
                        )
                    else:
                        raise NotImplementedError
                    
                    if saliency_labels is None:
                        continue

                    chunk_info.update({
                        "saliency_pos_labels": saliency_labels[0],
                        "saliency_neg_labels": saliency_labels[1],
                        "saliency_all_labels": saliency_labels[2]
                    })

                    chunk_infos.append(chunk_info)
                    total_length += 1

        # 验证chunk_infos的完整性
        valid_chunk_infos = []
        for i, chunk_info in enumerate(chunk_infos):
            if not isinstance(chunk_info, dict):
                print(f"Warning: chunk_infos[{i}] is not a dictionary. Skipping.")
                continue
            
            required_keys = {
                'start_label', 'middle_label', 'end_label',
                'saliency_pos_labels', 'saliency_neg_labels'
            }
            if not all(k in chunk_info for k in required_keys):
                print(f"Warning: chunk_infos[{i}] missing required keys. Skipping.")
                continue

            valid_chunk_infos.append(chunk_info)

        print(f"Generated {len(valid_chunk_infos)} valid chunks out of {total_length} total chunks, line_cnt: {line_cnt}")

        return valid_chunk_infos, len(valid_chunk_infos)

    def generate_gaussian_labels(self, video_length, start_idx, end_idx):
        """生成高斯软标签"""
        # start_idx, end_idx：当前GT片段的起始帧和结束帧
        # video_length：视频总帧数
        # 返回值：完整视频对于当前GT片段的软标签，包含start, middle, end三个维度
        try:
            # 计算中间点位置
            middle_idx = (start_idx + end_idx) / 2
            
            # 处理 start_idx 等于 end_idx 的情况
            if start_idx == end_idx:
                # 为单帧事件设置一个小的有效宽度
                min_sigma = 1.0  # 最小标准差
                
                # 生成时间序列
                t = torch.arange(video_length, dtype=torch.float)
                
                # 为单帧事件生成集中的高斯分布
                y_s = torch.exp(-((t - start_idx) ** 2) / (2 * min_sigma ** 2))
                y_m = torch.exp(-((t - middle_idx) ** 2) / (2 * min_sigma ** 2))
                y_e = torch.exp(-((t - end_idx) ** 2) / (2 * min_sigma ** 2))
            else:
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

            for item in (y_s, y_m, y_e):
                for score in item:
                    if np.isnan(score):
                        print("cur vid len, st, ed:", video_length, start_idx, end_idx)
                        input("wait! nan occur!") 
            
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
        model_inputs["chunk_idx"] = chunk_idx  # 添加 chunk_idx
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
                    self.get_saliency_labels_sub_as_query(boundary,chunk_info["gt_windows"], self.short_memory_sample_length)
            else:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                    self.get_saliency_labels_all(chunk_info["gt_windows"],self.saliency_scores_list[chunk_info["vid"],chunk_info["qid"]],boundary)                        
        
        elif self.dset_name in ['charades', 'tacos', 'activitynet', 'clotho-moment', 'unav100-subset', 'tut2017']:
            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                self.get_saliency_labels_sub_as_query(boundary,chunk_info["gt_windows"], self.short_memory_sample_length)
        else:
            raise NotImplementedError
        #   或者这里返回的字典里可以直接用chunk_info
        return dict(meta=chunk_info, model_inputs=model_inputs)
    
    def load_data(self):
        datalist = load_jsonl(self.data_path)
        return datalist

    def __len__(self):
        return len(self.chunk_infos)


    def get_saliency_labels_sub_as_query(self, boundary, gt_windows, ctx_l, max_n=1):
        """处理多个GT片段的显著性标签生成 (左闭右开)
        Args:
            boundary (list): [st, ed] 当前chunk的边界 (左闭右开)
            gt_windows (list): 多个GT片段的列表，每个元素为 [st, ed] (左闭右开)
            ctx_l (int): chunk的长度
            max_n (int): 每类（正/负样本）最多选择的样本数
        """
        all_pos_pools = []
        all_neg_pools = []
        st, ed = boundary

        # 对每个GT窗口计算正负样本池
        for gt_st, gt_ed in gt_windows:
            # 计算交集（左闭右开）
            intersect_st = max(st, gt_st)
            intersect_ed = min(ed, gt_ed)

            # 收集正样本池（左闭右开区间）
            if intersect_st < intersect_ed:  # 只有在有交集的情况下才添加
                all_pos_pools.extend(range(int(intersect_st), int(intersect_ed)))

            # 计算负样本池（排除正样本的区域）
            curr_neg_pool = list(range(int(st), int(intersect_st))) + \
                            list(range(int(intersect_ed), int(ed)))  # `intersect_ed` 直接用，不 +1
            all_neg_pools.extend(curr_neg_pool)

        # 去重并保证负样本不包含正样本
        all_pos_pools = list(set(all_pos_pools))
        all_neg_pools = list(set(all_neg_pools) - set(all_pos_pools))

        # 选择样本
        pos_clip_indices = random.sample(all_pos_pools, k=min(max_n, len(all_pos_pools))) if all_pos_pools else []
        neg_clip_indices = random.sample(all_neg_pools, k=min(max_n, len(all_neg_pools))) if all_neg_pools else []

        # 填充到固定长度
        pos_clip_indices.extend([-1] * (max_n - len(pos_clip_indices)))
        neg_clip_indices.extend([-1] * (max_n - len(neg_clip_indices)))

        # 生成分数数组（左闭右开，索引不能超过 ctx_l - 1）
        score_array = np.zeros(ctx_l)
        for pos_range in all_pos_pools:
            if 0 <= pos_range - st < ctx_l:  # 允许索引范围 [0, ctx_l-1]
                score_array[pos_range - st] = 1

        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all(self, gt_windows, gt_scores, boundary, max_n=1):
        st, ed = map(int, boundary)
        all_pos_indices = []
        all_pos_scores = []
        all_neg_indices = []
        
        # 生成分数数组
        score_array = np.zeros(ed - st)
        
        try:
            for gt_window in gt_windows:
                gt_st, gt_ed = map(int, gt_window)
                
                # 计算交集
                intersect_st = int(max(gt_st, st))
                intersect_ed = int(min(gt_ed, ed))
                
                if intersect_st < intersect_ed:
                    # 确保索引范围有效
                    src_start = max(0, intersect_st - gt_st)
                    src_end = min(len(gt_scores), intersect_ed - gt_st)
                    dst_start = max(0, intersect_st - st)
                    dst_end = min(len(score_array), intersect_ed - st)
                    
                    length = src_end - src_start
                    if length > 0:
                        score_array[dst_start:dst_start + length] = \
                            gt_scores[src_start:src_start + length]
                        
                        # 收集正样本及其分数
                        curr_scores = gt_scores[src_start:src_start + length]
                        curr_indices = list(range(intersect_st, intersect_st + length))
                        all_pos_indices.extend(curr_indices)
                        all_pos_scores.extend(curr_scores)
                    
                    # 收集负样本
                    curr_neg = list(range(st, gt_st)) + list(range(gt_ed, ed))
                    all_neg_indices.extend(curr_neg) 
        except Exception as e:
            print(f"Error in get_saliency_labels_all: {str(e)}")
            print(f"Boundary: {st}, {ed}")
            print(f"GT window: {gt_st}, {gt_ed}")
            print(f"Intersection: {intersect_st}, {intersect_ed}")
            print(f"Array shapes - score_array: {len(score_array)}, gt_scores: {len(gt_scores)}")
            input("press enter to continue, or ctrl-c to exit")
            return [], [], np.zeros(ed - st)
        
        # 去重并选择分数最高的正样本
        if all_pos_indices:
            pos_pairs = list(zip(all_pos_indices, all_pos_scores))
            pos_pairs = sorted(set(pos_pairs), key=lambda x: x[1], reverse=True)
            pos_clip_indices = [p[0] for p in pos_pairs[:max_n]]
        else:
            pos_clip_indices = []
        
        # 去重并随机选择负样本
        all_neg_indices = list(set(all_neg_indices) - set(all_pos_indices))
        neg_clip_indices = random.sample(all_neg_indices, k=min(max_n, len(all_neg_indices))) if all_neg_indices else []
        
        # 填充到固定长度
        pos_clip_indices.extend([-1] * (max_n - len(pos_clip_indices)))
        neg_clip_indices.extend([-1] * (max_n - len(neg_clip_indices)))
        
        return pos_clip_indices, neg_clip_indices, score_array

    def get_query(self, query):
        word_inds = torch.LongTensor(
            [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
        return self.embedding(word_inds)

    def _get_query_feat_by_qid(self, qid):
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

        # print("in func _get_query_feat_by_qid(): ")

        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            _feat_path = join(_feat_dir, f"{vid}.npz")
            _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
            _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
            # print("feat type: ",_feat_dir)
            # print("len: ",len(_feat))
        
        # some features are slightly longer than the others
        # 对齐所有特征的长度（采用多特征的时候）
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)

        # print("min_len: ",min_len)
        # print("output_len: ",len(torch.from_numpy(v_feat)))


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
        print("batch is empty")
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
            # batched_model_inputs[key] = torch.stack([
            #     inputs[key] for inputs in model_inputs_list
            # ])
             # 获取当前key下所有tensor的shape
            tensors = [inputs[key] for inputs in model_inputs_list]
            shapes = [t.shape for t in tensors]
            
            # 检查是否需要padding
            if len(shapes[0]) == 2:  # 只处理2D张量
                max_dim0 = max(s[0] for s in shapes)
                max_dim1 = max(s[1] for s in shapes)
                
                padded_tensors = []
                for tensor in tensors:
                    if tensor.shape[0] < max_dim0 or tensor.shape[1] < max_dim1:
                        # 创建填充后的张量
                        padded = torch.zeros(max_dim0, max_dim1,
                                          dtype=tensor.dtype,
                                          device=tensor.device)
                        # 复制原始数据
                        padded[:tensor.shape[0], :tensor.shape[1]] = tensor
                        padded_tensors.append(padded)
                    else:
                        padded_tensors.append(tensor)
                
                try:
                    batched_model_inputs[key] = torch.stack(padded_tensors)
                except Exception as e:
                    print(f"Error stacking tensors for key {key}")
                    print(f"Tensor shapes: {[t.shape for t in padded_tensors]}")
                    raise e
            else:
                # 对于非2D张量，直接尝试stack
                batched_model_inputs[key] = torch.stack(tensors)

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
    
    # 添加 chunk_idx 到 batched_model_inputs
    batched_model_inputs["chunk_idx"] = torch.tensor(
        [inputs["chunk_idx"] for inputs in model_inputs_list]
    )
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
                                 dtype=torch.long, device=device),
        'chunk_idx': batched_model_inputs['chunk_idx'].to(device, non_blocking=non_blocking)  # 添加 chunk_idx
    })
    
    # 4. 处理memory长度信息

    vid_feat_long = batched_model_inputs.get('video_feat_long', None)
    vid_feat_future = batched_model_inputs.get('video_feat_future', None)

    model_inputs['memory_len'] = [
        vid_feat_long.shape[1] if vid_feat_long is not None else 0,
        batched_model_inputs['video_feat_short'].shape[1],
        vid_feat_future.shape[1] if vid_feat_future is not None else 0,
    ]
    
    # 5. 处理标签
    targets = {}
    label_keys = [
        'start_label', 'middle_label', 'end_label',"short_memory_start",
        'saliency_pos_labels', 'saliency_neg_labels', 'saliency_all_labels'
    ]
    for key in label_keys:
        if key in batched_model_inputs:
            # 检查数据类型并相应处理
            if isinstance(batched_model_inputs[key], torch.Tensor):
                targets[key] = batched_model_inputs[key].to(device, non_blocking=non_blocking)
            elif isinstance(batched_model_inputs[key], list):
                targets[key] = torch.tensor(np.array(batched_model_inputs[key]), device=device)
            else:
                targets[key] = batched_model_inputs[key]
    
    # 6. 添加meta信息
    targets['meta'] = metas
    
    return model_inputs, targets