import math
import sys
import torch
from torch.utils.data import Dataset
import numpy as np
from torchtext.models.t5 import model
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
                 short_memory_sample_length, long_memory_sample_length, future_memory_sample_length, 
                 q_feat_type="last_hidden_state", v_feat_types="clip", a_feat_types="pann", 
                 max_q_l=500, max_v_l=2000, max_a_l=75, ctx_mode="video", clip_len=2,
                 max_windows=5, span_loss_type="l1", load_labels=True,
                 chunk_interval=1, short_memory_stride=1, long_memory_stride=1,
                 future_memory_stride=1, load_future_memory=False,test_mode=False,
                 use_gaussian_labels=True, asym_gaussian_label=False,
                 alpha_s=0.25, alpha_m=0.21, alpha_e=0.25,
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
        self.asym_gaussian_label = asym_gaussian_label
        self.alpha_s = alpha_s
        self.alpha_m = alpha_m
        self.alpha_e = alpha_e

        #   加载chunk sample_flag用到的参数
        self.neg_pos_ratio = neg_pos_ratio
        self.pos_expansion_ratio = pos_expansion_ratio

        print("for dataset ", self.dset_name, " the short_memory_sample_length is ", 
            self.short_memory_sample_length, " the long_memory_sample_length is ", 
        self.long_memory_sample_length, " the future_memory_sample_length is ", 
        self.future_memory_sample_length)
        
        # data
        self.data = self.load_data()
        # sample_size = len(self.data) // 4
        # self.data = random.sample(self.data, sample_size)  # 随机抽取
        # self.data = self.data[:sample_size] # 只取前sample_size个
        self.load_saliency_scores() 

        # 高斯标签
        self.st_label_dict, self.mid_label_dict, self.end_label_dict = self.load_all_gaussian_label_dict()

        # 分块信息
        self.chunk_infos = self.chunk_all_videos()[0]

        # 创建(索引, short_memory_start, qid)的元组列表
        indexed_data = [(i, item['short_memory_start'], item['qid']) for i, item in enumerate(self.chunk_infos)]
        
        # 按照short_memory_start和vid排序
        sorted_indexed_data = sorted(indexed_data, key=lambda x: (x[1], x[2]))
        
        # 提取排序后的索引，这就是采样顺序
        self.sample_seq = [item[0] for item in sorted_indexed_data]

        # # 检查采样是否符合设想的顺序
        # for idx in self.sample_seq:
        #     print(indexed_data[idx]) # chunk_idx, short_memory_start, vid
        # input("wait! check sample seq!\n")

        self.use_glove = 'glove' in q_feat_dir
        if self.use_glove:
            self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
            self.vocab.itos.extend(['<unk>'])
            self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
            self.vocab.vectors = torch.cat(
                (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
            self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)



        
    # train的时候，加载所有mid的标签，test的时候不能用这个
    def load_all_gaussian_label_dict(self):
        st_label_dict = {}
        mid_label_dict = {}
        end_label_dict = {}
        if len(self.data) == 0:
            print("No data is loaded, can't load gaussian labels")
            return     
        
        for line in tqdm(self.data, desc="Loading gaussian labels"):
            duration_frame = len(self._get_video_feat_by_vid(line["vid"]))
            mid_scores = np.zeros(duration_frame, dtype=float)
            # 利用generate_gaussian_labels，每个视频根据每个relevant_window生成高斯标签
            # 然后取每个位置上的max值
            all_mid_scores = []
            all_st_scores = []
            all_end_scores = []
            for window in line["relevant_windows"]:
                if self.dset_name == "qvhighlight" and "subs_train" not in self.data_path:
                    start_frame = math.floor(window[0]*self.fps)
                    end_frame = math.floor(window[1]*self.fps)
                else:
                    start_frame = math.floor(window[0]*self.fps)
                    end_frame = math.ceil(window[1]*self.fps)
                # generate_gaussian_labels
                # gaussian_label = self.generate_gaussian_labels(duration_frame, start_frame, end_frame)
                # TSGSV高斯生成代码中，ge_end是包含在gtnei的，所以这里要-1
                gaussian_label = self.generate_gaussian_labels(duration_frame, start_frame, end_frame -1 ,self.short_memory_stride)

                st_score, mid_scores, end_score = gaussian_label["start"], gaussian_label["middle"], gaussian_label["end"]

                all_mid_scores.append(mid_scores)
                all_st_scores.append(st_score)
                all_end_scores.append(end_score)

            mid_label_dict[(line["vid"], line["qid"])] = np.max(all_mid_scores, axis=0)
            st_label_dict[(line["vid"], line["qid"])] = np.max(all_st_scores, axis=0)
            end_label_dict[(line["vid"], line["qid"])] = np.max(all_end_scores, axis=0)
        return st_label_dict, mid_label_dict, end_label_dict

    def load_saliency_scores(self):        
        if len(self.data) == 0:
            print("No data is loaded, can't load saliency scores")
            return
        elif self.saliency_scores_list is None:
            self.saliency_scores_list = {}

        if self.dset_name == "qvhighlight" and "subs_train" not in self.data_path: 
            # 第一个循环添加tqdm
            for line in tqdm(self.data, desc="Loading qvhighlight saliency scores"):
                duration_frame = len(self._get_video_feat_by_vid(line["vid"]))
                all_vid_scores = np.zeros(duration_frame, dtype=float)
                
                for idx, scores in enumerate(line["saliency_scores"]):
                    if line["relevant_clip_ids"][idx] < duration_frame:
                        all_vid_scores[line["relevant_clip_ids"][idx]] = np.mean(scores)
                
                self.saliency_scores_list[(line["vid"], line["qid"])] = all_vid_scores
        else:
            # 第二个循环添加tqdm
            for line in tqdm(self.data, desc="Loading sub-as-query saliency scores"):
                duration_frame = len(self._get_video_feat_by_vid(line["vid"]))
                all_vid_scores = np.zeros(duration_frame, dtype=float)
                
                for windows in line["relevant_windows"]:
                    start_frame = math.floor(windows[0] * self.fps)
                    end_frame = math.floor(windows[1] * self.fps)
                    all_vid_scores[start_frame:end_frame] = 1
                
                self.saliency_scores_list[(line["vid"], line["qid"])] = all_vid_scores

    def calculate_gt_frames_in_long_memory(self, short_memory_start, gt_windows, long_memory_sample_length):
        """计算历史帧与GT窗口相交的帧数量
        
        Args:
            short_memory_start (int): 当前chunk的起始位置，也是历史帧的结束位置
            gt_windows (list): GT窗口列表，每个窗口格式为[start, end)，左闭右开
            long_memory_sample_length (int): 历史帧的最大长度
        
        Returns:
            int: 历史帧与所有GT窗口相交的帧数量总和
        """
        # 计算历史帧的范围 [long_memory_start, short_memory_start)
        long_memory_start = max(0, short_memory_start - long_memory_sample_length)
        long_memory_end = short_memory_start
        
        # 如果历史帧长度为0，直接返回0
        if long_memory_start >= long_memory_end:
            return 0
        
        total_intersection_frames = 0
        
        # 遍历所有GT窗口，计算与历史帧的相交帧数
        for gt_start, gt_end in gt_windows:
            # 计算两个区间的交集长度
            # 区间1: [long_memory_start, long_memory_end)
            # 区间2: [gt_start, gt_end)
            intersection_start = max(long_memory_start, gt_start)
            intersection_end = min(long_memory_end, gt_end)
            
            # 如果有交集，累加交集长度
            if intersection_start < intersection_end:
                intersection_frames = intersection_end - intersection_start
                total_intersection_frames += intersection_frames
        
        return total_intersection_frames


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
            # 这个计算出来的直接就是0开头的index，且是左闭右开的
            if self.dset_name == "qvhighlight":
                for idx, windows in enumerate(gt_windows):
                    gt_windows[idx] = [math.floor(windows[0] * self.fps), math.floor(windows[1] * self.fps)]
            elif self.dset_name in ["tacos", "activitynet"]:
                for idx, windows in enumerate(gt_windows):
                    gt_windows[idx] = [math.floor(windows[0] * self.fps), math.ceil(windows[1] * self.fps)]
            # 确定chunk的起始位置
            if not self.test_mode:
                interval = self.chunk_interval - 1 + self.short_memory_sample_length
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

                    # 计算历史帧与GT窗口相交的帧数量
                    gt_frames_in_long_memory = self.calculate_gt_frames_in_long_memory(
                        short_memory_start, 
                        gt_windows, 
                        self.long_memory_sample_length
                    )
                    chunk_info["gt_frames_in_long_memory"] = gt_frames_in_long_memory

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
                'saliency_pos_labels', 'saliency_neg_labels',
                'gt_frames_in_long_memory'
            }
            if not all(k in chunk_info for k in required_keys):
                print(f"Warning: chunk_infos[{i}] missing required keys. Skipping.")
                continue

            valid_chunk_infos.append(chunk_info)

        print(f"Generated {len(valid_chunk_infos)} valid chunks out of {total_length} total chunks, line_cnt: {line_cnt}")

        return valid_chunk_infos, len(valid_chunk_infos)
    
    def __gaussian_label__(self, t, mu, sigma, label_type):
        if not self.asym_gaussian_label or label_type == 'semantic':
            t = - np.power( (t - mu) / sigma, 2) / 2
        else:
            # ratio_ori = 0.2
            # ratio_new = 0.6
            ratio_ori = 0.1
            ratio_new = 0.3
            if label_type == 'start':
                mask = (t < mu)
                ratio_ori *= -1
                ratio_new *= -1
            else:
                mask = (t > mu)
            s_ori = ratio_ori * (mu - t) + sigma
            s_new = ratio_new * (t - mu) + sigma
            s = np.ma.array(s_ori, mask=mask)
            s = s.filled(s_new)
            t = - np.power((t - mu) / s, 2) / 2
        res = np.exp(t)
        return res

    def generate_gaussian_labels(self, duration_frame, s_gt, e_gt, stride):
        """Generate Gaussian labels for start and end.
        Args:
            duration_frame (int): duration of memory features in frames.
            s_gt, e_gt (float): start and end time of the annotation.
            stride (int):
        """
        s, e = 0, duration_frame
        
        span_length = e_gt - s_gt + 1
        t = np.arange(s, e, stride)

        sigma_s = self.alpha_s * span_length
        sigma_e = self.alpha_e * span_length
        sigma_se = self.alpha_m * span_length
        s_label = self.__gaussian_label__(t, s_gt, sigma_s, 'start')
        e_label = self.__gaussian_label__(t, e_gt, sigma_e, 'end')
        se_label = self.__gaussian_label__(t, (s_gt+e_gt)/2, sigma_se, 'semantic')
        # # GT区间内强制为1（左闭右开）
        # gt_mask = (t >= s_gt) & (t < e_gt)
        # se_label[gt_mask] = 1.0
        return {'start': s_label, 'middle': se_label, 'end': e_label}

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

        # 转换为torch张量
        chunk_labels = {
            'start_label': torch.tensor(self.st_label_dict[(vid, qid)][chunk_start:chunk_end]),
            'middle_label': torch.tensor(self.mid_label_dict[(vid, qid)][chunk_start:chunk_end]),
            'end_label': torch.tensor(self.end_label_dict[(vid, qid)][chunk_start:chunk_end])
        }
        
        # 验证标签有效性
        if all(torch.sum(label) == 0 for label in chunk_labels.values()):
            return None
        
        return chunk_labels

    def __getitem__(self, idx):
        """获取数据样本
        如果标签无效，返回None，在collate_fn中会被过滤掉
        """
        if self.test_mode:
            chunk_info = self.chunk_infos[self.sample_seq[idx]]
        else:
            chunk_info = self.chunk_infos[idx]
        # train和val都顺序输入
        # chunk_info = self.chunk_infos[self.sample_seq[idx]]
        # chunk_info = self.chunk_infos[idx]
    
        
        short_memory_start = chunk_info["short_memory_start"]
        short_memory_end = short_memory_start+self.short_memory_sample_length

        # long_memory_start=0
        # long_memory_end=short_memory_start
        long_memory_start=None
        long_memory_end=None

        # 取全部history的话，就没有long_memory_sample_length这个概念了
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
        model_inputs["chunk_idx"] = chunk_info["chunk_idx"]  # 添加 chunk_idx
        model_inputs["short_memory_start"] = short_memory_start
        
        # 修改查询特征的处理
        if self.use_glove:
            query_feat = self.get_query(chunk_info["query"])
        else:
            query_feat = self._get_query_feat_by_qid(chunk_info["qid"])
        # 确保查询特征被正确添加到model_inputs中
        model_inputs["query_feat"] = query_feat
        #   video feature
        if self.use_video:
            whole_video_feature = self._get_video_feat_by_vid(chunk_info["vid"])  # (Lv, Dv)
            #   short memory
            model_inputs["video_feat_short"]=whole_video_feature[short_memory_start:short_memory_end:self.short_memory_stride]
            ctx_l_short=len(model_inputs["video_feat_short"])
            #   long memory 
            # if self.long_memory_sample_length > 0:
            model_inputs["video_feat_long"]=whole_video_feature[long_memory_start:long_memory_end:self.long_memory_stride]  
            ctx_l_long=len(model_inputs["video_feat_long"])
            model_inputs["qid_vid"] = [chunk_info['qid'], chunk_info['vid']]
            model_inputs["short_memory_start"] = short_memory_start
            if not self.test_mode:
                # 从 mid_label_dict 获取权重
                vid, qid = chunk_info['vid'], chunk_info['qid']
                # 截取对应区间并下采样
                # 实现1：将mid_label作为权重
                # long_memory_weights = mid_labels[long_memory_start:long_memory_end:self.long_memory_stride]
                # 实现2：将显著性分数作为权重
                long_memory_weights = self.saliency_scores_list[(vid, qid)][long_memory_start:long_memory_end:self.long_memory_stride]
                if self.dset_name == "qvhighlight" and len(long_memory_weights) > 0 :
                    # 当long_memory_weights长度不为0时，将显著性分数归一化到0-1
                    long_memory_weights = long_memory_weights / 4
                model_inputs["long_memory_weight"] = torch.tensor(long_memory_weights)


            # # train/val均从数据集获得外部权重，评估upbound
            # # 从 mid_label_dict 获取权重
            # vid, qid = chunk_info['vid'], chunk_info['qid']
            # # 截取对应区间并下采样
            # # 实现1：将mid_label作为权重
            # # long_memory_weights = mid_labels[long_memory_start:long_memory_end:self.long_memory_stride]
            # # 实现2：将显著性分数作为权重
            # long_memory_weights = self.saliency_scores_list[(vid, qid)][long_memory_start:long_memory_end:self.long_memory_stride]
            # if self.dset_name == "qvhighlight" and len(long_memory_weights) > 0 :
            #     # 当long_memory_weights长度不为0时，将显著性分数归一化到0-1
            #     long_memory_weights = long_memory_weights / 4
            # model_inputs["long_memory_weight"] = torch.tensor(long_memory_weights)

            # # train/val都用之前预测的结果
            # model_inputs["qid_vid"] = [chunk_info['qid'], chunk_info['vid']]
            # model_inputs["short_memory_start"] = short_memory_start



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
            if ctx_l_long > 0:
                # 正常情况：计算时间编码
                tef_st_long = torch.arange(0, ctx_l_long, 1.0) / ctx_l_long
                tef_ed_long = tef_st_long + 1.0 / ctx_l_long
                tef_long = torch.stack([tef_st_long, tef_ed_long], dim=1)  # shape: [L, 2]
            else:
                tef_long = torch.zeros(0, 2)  # shape: [0, 2]（与空视频特征对齐）
            if self.use_video:
                model_inputs["video_feat_long"] = torch.cat(
                    [model_inputs["video_feat_long"], tef_long], 
                    dim=1
                )
            else:
                model_inputs["video_feat_long"] = tef_long
                
        # Span Label
        ## Short-term label
        model_inputs["start_label"] = chunk_info["start_label"]
        model_inputs["middle_label"] = chunk_info["middle_label"]
        model_inputs["end_label"] = chunk_info["end_label"]

        #   Saliency Label
        model_inputs["saliency_pos_labels"] = chunk_info["saliency_pos_labels"]
        model_inputs["saliency_neg_labels"] = chunk_info["saliency_neg_labels"]
        model_inputs["saliency_all_labels"] = chunk_info["saliency_all_labels"]

        # 历史帧与GT窗口的交集长度
        model_inputs["gt_frames_in_long_memory"] = chunk_info["gt_frames_in_long_memory"]
        
        # if chunk_info["qid"] == 9539:
        #     print("wait! check a sample for model input")
        #     for key, value in model_inputs.items():
        #         #如果是tensor类型
        #         if isinstance(value, torch.Tensor):
        #             print(f"{key}:shape {value.shape}:\n {value}")
        #         else:
        #             print(f"{key}: {value}")
        #     print("chunk_info: ",chunk_info)
        #     input("Press Enter to continue...")
        # if chunk_info["qid"] == 6083:
        #     print("wait! check a sample for model input")
        #     for key, value in model_inputs.items():
        #         #如果是tensor类型
        #         if isinstance(value, torch.Tensor):
        #             print(f"{key}:shape {value.shape}:{value}")
        #         else:
        #             print(f"{key}: {value}")
        #     print("chunk_info: ",chunk_info)
        #     input("Press Enter to continue...")


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

        pos_clip_indices = []
        neg_clip_indices = []
        if all_pos_pools and all_neg_pools:
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
        score_array = gt_scores[st:ed]     
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
                    
                    length = src_end - src_start
                    if length > 0:             
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

        # print("start from: ",st)
        # print("generated score_array: ", score_array)
        # print("true score_array: ", gt_scores)
        # input("press enter to continue, or ctrl-c to exit")
        
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
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            _feat_path = join(_feat_dir, f"{vid}.npz")
            _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
            _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        
        # some features are slightly longer than the others
        # 对齐所有特征的长度（采用多特征的时候）
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)

        return torch.from_numpy(v_feat)  # (Lv, D)

def start_end_collate_ol(batch, long_memory_sample_length = None):
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
        'saliency_pos_labels', 'saliency_neg_labels', 'saliency_all_labels',
        'gt_frames_in_long_memory'
    }
    valid_indices = []
    for idx, inputs in enumerate(model_inputs_list):
        if all(k in inputs for k in required_keys):
            valid_indices.append(idx)

    if len(valid_indices) < len(model_inputs_list):
        print(f"{len(model_inputs_list) - len(valid_indices)} samples are invalid and filtered out")
    
    # 只保留有效样本
    metas = [metas[i] for i in valid_indices]
    model_inputs_list = [model_inputs_list[i] for i in valid_indices]
    
    if len(valid_indices) == 0:
        return None
    # 继续处理有效样本
    batched_model_inputs = dict()

    for k in model_inputs_list[0].keys():
        if k in ["saliency_pos_labels", "saliency_neg_labels", "chunk_idx", "short_memory_start"]:
            batched_model_inputs[k] = torch.LongTensor([model_inputs_list[i][k] for i in valid_indices])
            continue
        if k == "qid_vid" or k == "gt_frames_in_long_memory":
            batched_model_inputs[k] = [model_inputs_list[i][k] for i in valid_indices]  # 保持原样
            continue
        if k in ["saliency_all_labels"]:
            pad_data, mask_data = pad_sequences_1d([model_inputs_list[i][k] for i in valid_indices], dtype=np.float32, fixed_length=None)
            batched_model_inputs[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue
        if k in ["start_label", "middle_label", "end_label"]:
            pad_data, mask_data = pad_sequences_1d([model_inputs_list[i][k] for i in valid_indices], dtype=torch.float32, fixed_length=None)
            batched_model_inputs[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue
        if k in ["video_feat_long"]:
            input_list = [model_inputs_list[i][k] for i in valid_indices]
            batched_model_inputs[k] = pad_sequences_1d(
                input_list, dtype=torch.float32, fixed_length=long_memory_sample_length)
            continue
        if k in ["long_memory_weight"]:
            batched_model_inputs[k] = pad_sequences_1d(
                [model_inputs_list[i][k] for i in valid_indices], dtype=torch.float32, fixed_length=long_memory_sample_length)
            continue
        if k in ["video_feat_short"]:
            input_list = [model_inputs_list[i][k] for i in valid_indices]
            batched_model_inputs[k] = pad_sequences_1d(
                input_list, dtype=torch.float32, fixed_length=None)
            continue

        batched_model_inputs[k] = pad_sequences_1d(
            [model_inputs_list[i][k] for i in valid_indices], dtype=torch.float32, fixed_length=None)
    
    # # 逐个检查batched_model_inputs，从键盘读入控制停止的信息    
    # flag = 1
    # while flag:
    #     for key, value in batched_model_inputs.items():
    #         if key in ["video_feat_long", "video_feat_short"]:
    #             print(f"{key}: {len(value[0])}*{len(value[0][0])}\n{value}")
    #         elif isinstance(value, torch.Tensor):
    #             print(f"{key}: {value.shape}\n{value}")
    #         else:
    #             print(f"{key}: {type(value)}\n{value}")
            
    #     flag = int(input("是否继续检查batched_model_inputs？1-是，0-否："))

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
                batched_model_inputs['video_feat_long'][0],
                batched_model_inputs['video_feat_short'][0]
            ]
            video_masks = [
                batched_model_inputs['video_feat_long'][1],
                batched_model_inputs['video_feat_short'][1]
            ]
            if 'video_feat_future' in batched_model_inputs:
                video_feats.append(batched_model_inputs['video_feat_future'][0])
                video_masks.append(batched_model_inputs['video_feat_future'][1])
            try:    
                src_vid = torch.cat(video_feats, dim=1)
                src_vid_mask = torch.cat(video_masks, dim=1)
            except Exception as e:
                print("video_feats: ", video_feats)
                print("video_feats.shape: ", video_feats[0].shape, video_feats[1].shape)
                print("Error concatenating video features:", e)
                raise
        else:
            src_vid = batched_model_inputs['video_feat_short'][0]
            src_vid_mask = batched_model_inputs['video_feat_short'][1]
    
    # 3. 基本输入处理
    model_inputs.update({
        'src_vid': src_vid.to(device, non_blocking=non_blocking),
        'src_txt': batched_model_inputs['query_feat'][0].to(device, non_blocking=non_blocking),
        'src_vid_mask': src_vid_mask.to(device, non_blocking=non_blocking),
        'src_txt_mask': batched_model_inputs['query_feat'][1].to(device, non_blocking=non_blocking),
        'chunk_idx': batched_model_inputs['chunk_idx'].to(device, non_blocking=non_blocking)  # 添加 chunk_idx
    })
    if 'long_memory_weight' in batched_model_inputs:
        model_inputs['long_memory_weight'] = batched_model_inputs['long_memory_weight'][0].to(device, non_blocking=non_blocking)
    if 'qid_vid' in batched_model_inputs:
        model_inputs['qid_vid'] = batched_model_inputs['qid_vid']
    if 'short_memory_start' in batched_model_inputs:
        model_inputs['short_memory_start'] = batched_model_inputs['short_memory_start']

    # 4. 处理memory长度信息

    vid_feat_long = batched_model_inputs.get('video_feat_long', None)[0]

    model_inputs['memory_len'] = [
        len(batched_model_inputs['video_feat_long'][0][0]) if vid_feat_long is not None else 0,
        len(batched_model_inputs['video_feat_short'][0][0]),
        0,
    ]
    
    # 5. 处理标签
    targets = {}
    label_keys = [
        'start_label', 'middle_label', 'end_label',"short_memory_start",
        'saliency_pos_labels', 'saliency_neg_labels', 'saliency_all_labels',
        'gt_frames_in_long_memory'
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

    # # 检查每个样本的内容
    # print("memory_len:", model_inputs['memory_len'])
    # flag = 1
    # for i in tqdm(range(len(model_inputs['src_vid']))):
    #     for key in model_inputs:
    #         if key == 'memory_len':
    #             continue
    #         if isinstance(model_inputs[key], torch.Tensor):
    #             print(f"{key}: {model_inputs[key][i].shape}\n{model_inputs[key][i]}")
    #         else:
    #             print(f"{key}: {type(model_inputs[key][i])}\n{model_inputs[key][i]}")
    #     flag = int(input("是否继续检查batched_model_inputs？1-是，0-否："))
    #     if flag == 0:
    #         break
    return model_inputs, targets