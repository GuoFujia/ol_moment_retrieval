import math
from multiprocessing import process

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lighthouse.common.matcher import build_matcher
from lighthouse.common.misc import accuracy
from lighthouse.common.moment_transformer import build_transformer
from lighthouse.common.position_encoding import build_position_encoding
from lighthouse.common.utils.span_utils import temporal_iou, generalized_temporal_iou, span_cxw_to_xx
from lighthouse.common.compressor import MultimodalTokenCompressor
from lighthouse.common.compressor import SimpleCompressor, TextGuidedCompressor, CompressorWithExternalWeights, CompressorWithWeightedKV, CompressorWithPostWeighting, CompressorWithSmoothMechanism, ResidualCompressor, CrossAttentionResidualCompressor, CompressorWithPositionWeights, CrossAttentionResidualCompressorWithQuery


class OLMomentDETR(nn.Module):
    """ This is the Moment-DETR module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False, max_v_l=75, span_loss_type="l1", 
                 use_txt_pos=False, n_input_proj=2, aud_dim=0, compress_len=10, use_vid_compression=True, weight_alpha=0.5):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Moment-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.frame_class_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 输出维度仍为4，但表示4个二分类的logits
        self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.compress_len = compress_len

        # 一个字典，记录每个[qid_vid]对对应的视频的显著性分数值
        # 用saliency_score_dict[qid_vid]可以快速访问
        # saliency_score_dict结构：每个[qid_vid]索引两个元素：1.start：当前vid对于qid从start·开始采样；
        #                                              2.score：二维float数组，score[i][j]代表从start开始的第i帧的第j个预测saliency_score
        self.saliency_score_dict = {}

        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        # self.foreground_thd = foreground_thd
        # self.background_thd = background_thd

        self.use_vid_compression = use_vid_compression
        if self.use_vid_compression:
            # self.memory_compressor = TextGuidedCompressor(t_dim = hidden_dim,v_dim = hidden_dim,hidden_dim = hidden_dim,compress_len=self.compress_len,weight_alpha=weight_alpha)
            # self.memory_compressor =CompressorWithExternalWeights(hidden_dim, compress_len=compress_len, weight_alpha=weight_alpha)
            # self.memory_compressor = MultimodalTokenCompressor(hidden_dim, compress_len=compress_len)
            # self.text_guided_video_attention = TextGuidedVideoAttention(t_dim = hidden_dim,v_dim = hidden_dim,hidden_dim = hidden_dim)
            # self.memory_compressor = CompressorWithSmoothMechanism(hidden_dim, compress_len=compress_len)
            # self.memory_compressor = CompressorWithWeightedKV(hidden_dim, compress_len=self.compress_len)
            # self.memory_compressor = CompressorWithPostWeighting(hidden_dim, compress_len=self.compress_len)
            # self.memory_compressor = ResidualCompressor(dimension=hidden_dim, compress_len=self.compress_len, weight_alpha=weight_alpha)
            # self.memory_compressor = CrossAttentionResidualCompressor(dimension=hidden_dim, compress_len=self.compress_len, weight_alpha=weight_alpha)
            self.memory_compressor = CrossAttentionResidualCompressorWithQuery(dimension=hidden_dim, compress_len=self.compress_len, weight_alpha=weight_alpha, attn_weight=0.3)
            # self.memory_compressor = CompressorWithPositionWeights(dimension=hidden_dim, compress_len=self.compress_len)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.aux_loss = aux_loss

    def forward(self, src_txt, src_txt_mask, src_vid, memory_len, src_vid_mask, chunk_idx, src_aud=None, src_aud_mask=None,
                long_memory_weight=None, qid_vid=None, short_memory_start=None):
        """
        在线模型forward需要的输入:
                -src_txt: [batch_size, L_txt, D_txt]
                -src_txt_mask: [batch_size, L_txt],
                -src_vid: [batch_size, L_vid, D_vid], 
                    L_vid = long_memory_sample_length + short_memory_sample_length + future_memory_sample_length
                - src_vid_mask: [batch_size, L_vid]
                -memory_len: [], 三种记忆的采样长度
                -long_memory_weight: [batch_size, L_vid],在train的时候使用,用于提供extern_weight
                    -qid_vid: [batch_size,],在eval/val的时候使用的，用于将样本和saliency_score_dict中的标签匹配
            返回值为包含以下内容的dict:
                -"pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
                -"pred_logits": The class prediction for all queries.
                -"frame_pred": short记忆的逐帧预测概率
                -"saliency_scores": short记忆的逐帧saliency score
                -"chunk_idx": chunk index

        """

        # if qid_vid is not None:
        #     # 捕获6083，"tsOkWgzgW-o_60.0_210.0"
        #     for i in range(len(qid_vid)):
        #         if qid_vid[i][0] == 6083:
        #             print("catch 6083")
        #             print("src_txt:", src_txt[i])
        #             print("src_txt_mask:", src_txt_mask[i])
        #             print("src_vid:", src_vid[i])
        #             print("src_vid_mask:", src_vid_mask[i])
        #             print("memory_len:", memory_len)
        #             print("chunk_idx:", chunk_idx[i])
        #             if  long_memory_weight is not None:
        #                 print("long_memory_weight:", long_memory_weight[i])
        #             if qid_vid is not None:
        #                 print("qid_vid:", qid_vid[i])
        #             print("short_memory_start:", short_memory_start[i])
        #             input("Press Enter to continue...")
        #     input("捕获6083结束")
            


        #   拼接音频特征
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
        #   投影输入特征
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)

        # 取长期记忆视频，长期记忆mask
        long_vid = src_vid[:, :memory_len[0]]
        long_vid_mask = src_vid_mask[:, :memory_len[0]]

        # # 逐个样本的检查mask
        # flag = 1
        # for i in range(long_vid_mask.shape[0]):
        #     if(long_vid_mask[i][0]!=0): 
        #         print("long_vid_mask[", i, "]:\n", long_vid_mask[i])
        #         print("src_txt_mask[", i, "]:\n", src_txt_mask[i])
        #         flag = input("1 to continue, 0 to exit")
        #         if flag == '0':
        #             break
            

        # 压缩长期记忆
        if self.use_vid_compression:
            # 训练/val均使用数据集提供的显著性分数作为extern_weight，验证方法的upbound
            # extern_weight = long_memory_weight

            # 如果是train，使用数据集提供的mid分数作为extern_weight
            if self.training:
                extern_weight = long_memory_weight
            else: # inference时，使用过去推理的mid结果
                all_weights = []
                extern_weight = None  # 显式初始化

                for batch_idx in range(len(qid_vid)):
                    output = self.get_long_memory_weight_by_qid_vid(qid_vid[batch_idx][0], qid_vid[batch_idx][1])
                    
                    if output is None:
                        # print(f"No long memory weight found for qid {qid_vid[batch_idx][0]} and vid {qid_vid[batch_idx][1]}")
                        # 清空已收集的权重，确保一致性
                        all_weights = []  
                        extern_weight = None
                        break
                    indexed_weight = self.index_weight(output[1], output[0], short_memory_start[batch_idx], memory_len[0])
                    all_weights.append(torch.from_numpy(indexed_weight).float())

                # 只有在所有batch都成功获取权重时才stack
                if len(all_weights) == len(qid_vid):
                    extern_weight = torch.stack(all_weights, dim=0)
                else:
                    extern_weight = None
                    
            # 确保extern_weight在和long_vid在相同的设备上
            if extern_weight is not None:
                extern_weight = extern_weight.to(long_vid.device)

            if torch.isnan(long_vid).any():
                print("long_vid contains NaN!")

            # long_vid = self.text_guided_video_attention(Ft = src_txt, Fv = long_vid, vid_mask = long_vid_mask, text_mask = src_txt_mask)
            # compress_long_vid, compress_long_mask = self.memory_compressor(Fv = long_vid, mask=long_vid_mask, extern_weight=extern_weight)
            # compress_long_vid, compress_long_mask = self.memory_compressor(Fv = long_vid, mask=long_vid_mask)
            # compress_long_vid, compress_long_mask = self.memory_compressor(Fv = long_vid, mask=long_vid_mask, query=src_txt, query_mask=src_txt_mask)
            # compress_long_vid, compress_long_mask = self.memory_compressor(Fv = long_vid, vid_mask=long_vid_mask, Ft=src_txt, text_mask=src_txt_mask)
            compress_long_vid, compress_long_mask = self.memory_compressor(Fv = long_vid, vid_mask=long_vid_mask, Ft=src_txt, text_mask=src_txt_mask,extern_weight=extern_weight)

            # print("shape of compress_long_vid:", compress_long_vid.shape)
            # print("shape of compress_long_mask:", compress_long_mask.shape)
            # print("compress_long_vid:", compress_long_vid)
            # print("compress_long_mask:", compress_long_mask)
            # input("Press Enter to continue...")

            if torch.isnan(compress_long_vid).any():
                print("compress_long_vid contains NaN!")
                print("long_vid:", long_vid)
                print("src_txt:", src_txt)
                print("src_txt_mask:", src_txt_mask)
                print("long_vid_mask:", long_vid_mask)
                print("extern_weight:", extern_weight)
                input("Press Enter to continue...")

            #   将压缩后的长期记忆和短期，future记忆拼接得到新的src_vid, src_vid_mask
            src_vid, src_vid_mask = torch.cat([compress_long_vid, src_vid[:, memory_len[0]:]], dim=1), \
                torch.cat([compress_long_mask, src_vid_mask[:, memory_len[0]:]], dim=1)
            memory_len[0] = self.compress_len
            # print("after compress, src_vid shape and src_vid_mask shape:", src_vid.shape, src_vid_mask.shape)
            # # 逐个样本的查看mask
            # flag = 1
            # for i in range(src_vid_mask.shape[0]):
            #     print("src_vid_mask[", i, "]:\n", src_vid_mask[i])
            #     print("src_txt_mask[", i, "]:\n", src_txt_mask[i])
            #     flag = input("1 to continue, 0 to exit")
            #     if flag == '0':
            #         break

        #   拼接视频和文本的特征，掩码
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        #   分别生成位置编码并拼接
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)

        # print("after compress, src_vid shape and src_vid_mask shape:", src_vid.shape, src_vid_mask.shape)
        # print("shape of src, mask, pos_vid:", src.shape, mask.shape, pos_vid.shape)
        # input("press enter to continue...")


        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)
        #   通过transformer
        # hs, memory, compress_memory = self.transformer(src, ~mask, self.query_embed.weight, pos, use_memory_compression=False)
        hs, memory= self.transformer(src, ~mask, self.query_embed.weight, pos)
        # hs.shape = (num_decoder_layers, batch_size, num_queries, hidden_dim)
        # memory.shape = (batch_size, L_vid+L_txt, hidden_dim)
        

        out = {}

        # follow moment-detr
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        outputs_coord = self.span_embed(hs)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out['pred_logits'] = outputs_class[-1]
        out['pred_spans'] = outputs_coord[-1]
        
        # 帧预测结果和显著性分数
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        short_start = memory_len[0]
        short_end = short_start + memory_len[1]
        
        out['frame_pred'] = self.frame_class_embed(vid_mem)[:, short_start:short_end]    # (bsz, short_memory_length, 4)
        out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)[:, short_start:short_end]  # (bsz, short_memory_length)

        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        # 添加 chunk_idx 到输出
        out["chunk_idx"] = chunk_idx  # [batch_size]

        # 维护和更新saliency_score_dict的逻辑
        if self.use_vid_compression and not self.training and qid_vid is not None and short_memory_start is not None:
            qid_vid_list = qid_vid # 转换为Python列表
            short_memory_start_list = short_memory_start
            batch_size = src_vid.shape[0]

            for i in range(batch_size):
                qid,vid = qid_vid_list[i]
                current_short_start = short_memory_start_list[i]
                current_saliency_score = out["saliency_scores"][i, :]
                current_length = current_saliency_score.size(0)
                current_saliency_score_np = current_saliency_score.cpu().detach().numpy()

                weight_col = 1 if self.training else memory_len[1]
                if (qid,vid) not in self.saliency_score_dict:
                    # 初始化新条目
                    weight = np.full((current_length, weight_col), np.nan)
                    # 将当前预测结果填充到weight中
                    weight[:current_length, 0] = current_saliency_score_np
                    self.saliency_score_dict[(qid,vid)] = {
                        'start': current_short_start,
                        'weight': weight
                    }
                else:
                    entry = self.saliency_score_dict[(qid,vid)]
                    existing_start = entry['start']
                    existing_weight = entry['weight']
                    new_L = current_short_start - existing_start + current_length
                    new_weight = np.full((new_L, weight_col), np.nan)
                    # 将existing_weight复制到new_weight

                    # print("existing_weight:", existing_weight)
                    # print("new_weight:", new_weight)
                    # input("press enter to continue...")

                    new_weight[:existing_weight.shape[0], :weight_col] = existing_weight
                    # 将现有数据复制到新数组
                    for i, saliency_score in enumerate(current_saliency_score_np):
                        for insert_idx, latest_label in enumerate(new_weight[current_short_start + i - existing_start]):
                            if np.isnan(latest_label) :

                                if np.isnan(saliency_score):
                                    input("nan occur in saliency score")

                                new_weight[current_short_start + i - existing_start, insert_idx] = saliency_score
                                break
                    # 更新条目
                    entry['weight'] = new_weight

                    # print("new_weight:", new_weight)
                    # input("press enter to continue...")
        return out

    def reset_saliency_score_dict(self):
        self.saliency_score_dict = {}

    def process_probs_by_avg(self,probs):
        return np.mean(probs, axis=0)

    # 通过saliency_score_dict获取long memory的weight
    # 返回值:1.startl2.从start开始的全部的long_memory_weight,形状为(batch_size, long_memory_length)
    def get_long_memory_weight_by_qid_vid(self,qid,vid):
        if (qid,vid) not in self.saliency_score_dict:
            return None
        dict = self.saliency_score_dict[(qid,vid)]
        dict_weight = dict["weight"]
        res = []
        for frame in dict_weight:
            # 对于frame中的prob，先筛除nan

            # print(frame.dtype)  # 如果 frame 是 NumPy 数组
            # print([type(x) for x in frame])  # 如果 frame 是 Python 列表
            # print("frame:", frame)
            # print("dict_weight:", dict_weight)
            # print("dict_start:", dict_start)
            # print("dict:", self.saliency_score_dict[(qid,vid)])
            # input("press enter to continue...")

            valid_frame = frame[~np.isnan(frame)]

            if np.isnan(valid_frame).any():
                input("all nan occur in valid_frame")

            frame_weight = self.process_probs_by_avg(valid_frame)
            # 对于每一帧的权重，归一化到0-1
            # 如果是qv数据集，先把分数缩放
            if ".0" in vid:
                frame_weight = frame_weight / 4
            if frame_weight > 1:
                frame_weight = 1
            if frame_weight < 0:
                frame_weight = 0
                
            res.append(frame_weight)
        return dict["start"], np.array(res)
    # 从get_long_memory_weight_by_qid_vid返回的weight_list中获取[st,ed]之间的分数
    # 返回的结果应该是pad到return_len的长度
    def index_weight(self,weight_list,weight_start,short_start,return_len):
        # 从weight_list中，取出[weight_start,short_start]之间的分数（不包括short_start对应的帧）作为raw_res
        # 如果raw_res比return_len小，有效值在低位索引处，用0填充
        # 如果raw_res比return_len大，则从后向前阶段，确保返回值长度
        weight_list = np.pad(weight_list, (weight_start, 0), 'constant', constant_values=0)
        if weight_start > short_start:
            return None
        raw_res = weight_list[:short_start]
        if len(weight_list)<short_start:
            raw_res = np.pad(weight_list, (0, short_start - len(weight_list)), 'constant', constant_values=0)
        if len(raw_res) < return_len:
            raw_res = np.pad(raw_res, (0, return_len - len(raw_res)), 'constant', constant_values=0)
        else:
            raw_res = raw_res[-return_len:]
        return raw_res
        
class SetCriterionOl(nn.Module):
    """ This class computes the loss for ol_DETR.
    主要计算两部分的loss:
        1) 帧级别的分类损失 [start, mid, end, irrelevant].
        2) saliency score 预测结果的损失.
    还考虑添加一个时序一致性loss来约束saliency score, 让相邻帧的saliency score预测结果变化更加平滑
    """

    def __init__(self, weight_dict, losses, saliency_margin=1, gamma=.0):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            saliency_margin: float, margin for saliency loss.
            gamma: float, weight factor for hard sample mining in label loss.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.saliency_margin = saliency_margin
        self.gamma = gamma  # 添加 gamma 作为初始化参数

    # 根据frame_pred结果做的span loss
    def loss_spans(self, outputs, targets):
        frame_pred = outputs['frame_pred']  # (batch_size, short_memory_len, 4)
        chunk_idx = outputs.get('chunk_idx', None)
        
        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs")
        
        device = frame_pred.device
        batch_size = frame_pred.shape[0]
        seq_len = frame_pred.shape[1]
        
        # 初始化 loss_v1
        loss_v1 = torch.tensor(0.0, device=device, requires_grad=True)
        valid_samples = 0
        
        for i, idx in enumerate(chunk_idx):
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue
                    
            short_memory_start = target_meta['short_memory_start']
            start_prob = torch.sigmoid(frame_pred[i, :, 0])  
            end_prob = torch.sigmoid(frame_pred[i, :, 2])    
            
            span_scores = []  # 存储(start_idx, end_idx, confidence)
            for st in range(seq_len):
                for ed in range(st, seq_len):
                    conf = start_prob[st] * end_prob[ed]
                    global_st = st + short_memory_start
                    global_ed = ed + short_memory_start
                    span_scores.append((global_st, global_ed, conf))
            
            span_scores.sort(key=lambda x: x[2], reverse=True)
            top_k = min(5, len(span_scores))
            top_spans = span_scores[:top_k]
            
            gt_windows = target_meta['gt_windows']  # list of [start, end] pairs
            
            span_ious = []
            pred_confs = []
            for st, ed, conf in top_spans:
                pred_span = torch.tensor([[st, ed]], device=device)
                max_iou = float('-inf')
                
                for gt_start, gt_end in gt_windows:
                    gt_span = torch.tensor([[gt_start, gt_end]], device=device)
                    iou = generalized_temporal_iou(pred_span, gt_span)
                    max_iou = max(max_iou, iou.item())
                
                span_ious.append(max_iou)
                pred_confs.append(conf.item())
            
            if not span_ious:  # 如果没有有效的span
                continue
                    
            span_ious = torch.tensor(span_ious, device=device, requires_grad=True)  # 需要梯度
            pred_confs = torch.tensor(pred_confs, device=device, requires_grad=True)  # 需要梯度
            
            # 使用非原地操作
            loss_v1 = loss_v1 + F.mse_loss(pred_confs, span_ious)
            valid_samples += 1
        
        if valid_samples > 0:
            loss_v1 = loss_v1 / valid_samples
        
        return {
            'loss_spans_v1': loss_v1
        }

    def new_loss_spans(self, outputs, targets):
        pred_spans = outputs['pred_spans']  # (batch_size, num_queries, 2)
        chunk_idx = outputs.get('chunk_idx', None)

        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs")

        device = pred_spans.device
        batch_size = pred_spans.shape[0]
        num_queries = pred_spans.shape[1]

        # 初始化损失
        loss_span = torch.tensor(0.0, device=device, requires_grad=True)
        loss_giou = torch.tensor(0.0, device=device, requires_grad=True)
        valid_samples = 0

        # 遍历每个样本
        for i, idx in enumerate(chunk_idx):
            # 找到 targets 中对应的样本
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue

            # 获取真实的时间片段 (center_x, width)
            gt_spans = target_meta['gt_windows']  # list of [st, ed]
            if not gt_spans:
                continue  # 如果没有真实片段，跳过

            # 将真实片段转换为张量
            gt_spans = torch.tensor(gt_spans, dtype=torch.float32, device=device)  # (num_gt_spans, 2)

            # 当前样本的预测片段
            sample_pred_spans = pred_spans[i]  # (num_queries, 2)
            short_start = targets['short_memory_start']
            sample_pred_spans = span_cxw_to_xx(sample_pred_spans)
            sample_pred_spans = sample_pred_spans * target_meta["start_label"].shape[0]
            sample_pred_spans = sample_pred_spans + torch.tensor(short_start[i], dtype=torch.float32, device=device)

            # 计算所有预测和真实片段对的L1距离
            l1_dist = torch.cdist(sample_pred_spans, gt_spans, p=1)  # (num_queries, num_gt_spans)
            min_l1_loss = l1_dist.min(dim=1)[0].mean()  # 每个查询到最近真实片段的最小L1距离

            # 计算广义IoU
            giou_matrix = 1 - generalized_temporal_iou(sample_pred_spans, gt_spans)  # (num_queries, num_gt_spans)
            max_giou_per_query = giou_matrix.max(dim=1)[0]  # 每个查询的最大GIoU
            mean_max_giou = max_giou_per_query.mean()  # 所有查询的平均最大GIoU

            # 累加损失
            loss_span = loss_span + min_l1_loss
            loss_giou = loss_giou + mean_max_giou
            valid_samples += 1

        if valid_samples > 0:
            loss_span = loss_span / valid_samples
            loss_giou = loss_giou / valid_samples

        return {
            'loss_span': loss_span / 40,  # 将loss_span保持在1附近
            'loss_giou': loss_giou
        }

    def loss_labels(self, outputs, targets, log=True):
        """帧分类损失函数，先筛选有效样本，再批量计算损失"""
        frame_pred = outputs['frame_pred']  # (batch_size, short_memory_len, 4)
        chunk_idx = outputs.get('chunk_idx', None)  # 获取 chunk_idx

        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs. Ensure it is passed correctly.")

        # 筛选有效样本
        valid_preds = []
        valid_start_labels = []
        valid_middle_labels = []
        valid_end_labels = []
        
        # flag = 1
        for i, idx in enumerate(chunk_idx):
            # 找到 targets 中对应的样本
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue  # 如果找不到对应的 target，跳过该样本

            # 获取当前样本的标签
            start_label = target_meta['start_label']  # (num_frames,)
            middle_label = target_meta['middle_label']  # (num_frames,)
            end_label = target_meta['end_label']  # (num_frames,)

            # if flag == 1:
            #     # 当前样本的meta信息，主要是视频时长，short_memory_start，relevant_windows
            #     print("duration_frame: ", target_meta['duration_frame'])
            #     print("short_memory_start: ", target_meta['short_memory_start'])
            #     print("gt_windows: ", target_meta['gt_windows'])
            #     # 当前样本的预测结果，按st，mid，ed分别显示
            #     pred_start = frame_pred[i, :, 0]
            #     pred_middle = frame_pred[i, :, 1]
            #     pred_end = frame_pred[i, :, 2]
            #     print("pred_start: ", torch.sigmoid(pred_start))
            #     print("pred_middle: ", torch.sigmoid(pred_middle))
            #     print("pred_end: ", torch.sigmoid(pred_end))
            #     print("start_label: ", start_label)
            #     print("middle_label: ", middle_label)
            #     print("end_label: ", end_label)

            #     flag = input("1 to continue, 0 to exit")

            # 获取当前样本的预测
            pred = frame_pred[i]  # (short_memory_len, 4)

            # 添加到有效样本列表
            valid_preds.append(pred)
            valid_start_labels.append(start_label)
            valid_middle_labels.append(middle_label)
            valid_end_labels.append(end_label)

        if not valid_preds:
            return {"loss_label": torch.tensor(0.0, device=frame_pred.device)}

        # 将有效样本组织成批量张量
        valid_preds = torch.stack(valid_preds)  # (num_valid_samples, num_frames, 4)
        valid_start_labels = torch.stack(valid_start_labels)  # (num_valid_samples, num_frames)
        valid_middle_labels = torch.stack(valid_middle_labels)  # (num_valid_samples, num_frames)
        valid_end_labels = torch.stack(valid_end_labels)  # (num_valid_samples, num_frames)

        # 将标签移动到与预测相同的设备上
        valid_start_labels = valid_start_labels.to(valid_preds.device)
        valid_middle_labels = valid_middle_labels.to(valid_preds.device)
        valid_end_labels = valid_end_labels.to(valid_preds.device)

        # 批量计算损失
        losses = {}
        total_loss = 0

        label_types = ['start_label', 'middle_label', 'end_label']
        pred_indices = [0, 1, 2]  # pred 的索引对应 start, middle, end
        for label_type, pred_idx in zip(label_types, pred_indices):
            target = valid_start_labels if label_type == 'start_label' else \
                    valid_middle_labels if label_type == 'middle_label' else \
                    valid_end_labels  # (num_valid_samples, num_frames)
            pred_slice = valid_preds[:, :, pred_idx]  # (num_valid_samples, num_frames)

            # 将 logits 转换为概率值
            pred_probs = torch.sigmoid(pred_slice)  # (num_valid_samples, num_frames)

            # 检查 pred_probs 是否有效
            assert not torch.isnan(pred_probs).any(), "pred_probs contains NaN"
            assert not torch.isinf(pred_probs).any(), "pred_probs contains inf"
            assert (pred_probs >= 0).all() and (pred_probs <= 1).all(), "pred_probs must be in [0, 1]"

            # 计算加权BCE损失
            loss = F.binary_cross_entropy(
                pred_probs.to(torch.float64), target, reduction='none'
            )

            # # 计算加权BCE损失
            # loss = F.binary_cross_entropy_with_logits(
            #     pred_slice, target, reduction='none'
            # )

            # 动态权重，赋予较难的类别较高的权重
            weight = torch.abs(pred_probs - target)  # 使用 gamma 计算权重
            weight = torch.pow(weight, self.gamma)
            loss = (loss * weight).mean()
            # loss = loss.mean()  # 先不用weight

            if torch.isnan(loss).any():
                print(f"NaN detected in {label_type} loss. Target: {target}, Pred: {pred_slice}")
                input("Press Enter to continue...")

            losses[f'loss_{label_type}'] = loss
            total_loss += loss

        # 平均损失
        losses['loss_label'] = total_loss

        return losses

    def loss_saliency(self, outputs, targets, log=True):
        """显著性损失函数，先筛选有效样本，再批量计算损失"""
        if ("saliency_scores" not in outputs or 
            "saliency_pos_labels" not in targets or 
            "short_memory_start" not in targets):
            print("info lacked to compute loss saliency")
            return {"loss_saliency": 0}

        saliency_scores = outputs["saliency_scores"]  # (batch_size, num_frames)
        chunk_idx = outputs.get('chunk_idx', None)  # 获取 chunk_idx

        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs. Ensure it is passed correctly.")

        # 筛选有效样本
        valid_scores = []
        valid_pos_labels = []
        valid_neg_labels = []
        valid_tgt_scores = []

        for i, idx in enumerate(chunk_idx):
            # 找到 targets 中对应的样本
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue  # 如果找不到对应的 target，跳过该样本

            # 检查 saliency_pos_labels 和 saliency_neg_labels 是否有效
            pos_labels = target_meta['saliency_pos_labels']
            neg_labels = target_meta['saliency_neg_labels']
            if -1 in pos_labels or -1 in neg_labels:
                continue  # 如果其中一个为 -1，跳过该样本

            # 获取当前样本的预测和标签
            pred_scores = saliency_scores[i]  # (num_frames,)， 因为是模型的输出，所以已经是张量
            short_memory_start = target_meta['short_memory_start']
            tgt_scores = torch.tensor(target_meta['saliency_all_labels'], dtype=torch.float32, device=pred_scores.device)   # 因为是从targets来的，所以需要手动转换为张量

            # 转换为相对于 short memory 的局部索引
            pos_indices = [pos_idx - short_memory_start for pos_idx in pos_labels]
            neg_indices = [neg_idx - short_memory_start for neg_idx in neg_labels]

            if not pos_indices or not neg_indices:
                continue  # 如果没有有效索引，跳过该样本

            # 将索引列表转换为张量
            pos_indices = torch.tensor([pos_idx - short_memory_start for pos_idx in pos_labels], 
                                       dtype=torch.long, device=pred_scores.device)
            neg_indices = torch.tensor([neg_idx - short_memory_start for neg_idx in neg_labels], 
                                       dtype=torch.long, device=pred_scores.device)

            # 添加到有效样本列表
            valid_scores.append(pred_scores)
            valid_tgt_scores.append(tgt_scores)
            valid_pos_labels.append(pos_indices)
            valid_neg_labels.append(neg_indices)

        if not valid_scores:
            return {"loss_saliency": 0}

        # 将有效样本组织成批量张量
        valid_scores = torch.stack(valid_scores)  # (num_valid_samples, num_frames)
        valid_tgt_scores = torch.stack(valid_tgt_scores)  # (num_valid_samples, num_frames)

        # 1.批量计算对比损失
        base_losses = []
        for i in range(len(valid_scores)):
            pos_indices = valid_pos_labels[i]  # 可能是多个索引的 tensor
            neg_indices = valid_neg_labels[i]  # 可能是多个索引的 tensor

            pos_scores = valid_scores[i, pos_indices]  
            neg_scores = valid_scores[i, neg_indices]  

            # 基础对比损失
            base_loss = torch.clamp(
                self.saliency_margin + neg_scores - pos_scores, 
                min=0
            ).mean()

            base_losses.append(base_loss)

        # 平均所有样本的损失
        loss_contrastive = torch.mean(torch.stack(base_losses))

        # 2.计算celoss
        # 创建二分类标签：valid_tgt_scores > 0 为正样本，否则为负样本
        binary_targets = (valid_tgt_scores > 0).float()

        # 使用 sigmoid 函数将分数转换为概率
        pred_probs = torch.sigmoid(valid_scores)
        
        # 计算二元交叉熵损失
        ce_loss_fn = torch.nn.BCELoss()
        loss_ce = ce_loss_fn(pred_probs, binary_targets)

        # 3.计算平滑L1损失
        # 使用 smooth L1 损失 (Huber loss)
        smooth_l1_loss_fn = torch.nn.SmoothL1Loss()
        loss_smooth_l1 = smooth_l1_loss_fn(valid_scores, valid_tgt_scores)

        # 合并所有损失
        final_loss = loss_contrastive + loss_ce + loss_smooth_l1
        return {"loss_saliency": final_loss}

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "span": self.new_loss_spans,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices=None))

        # total_loss = sum(losses[k] * self.weight_dict.get(k, 1.0) for k in losses.keys())
        total_loss = sum(losses.values())

        return total_loss, losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)

def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = OLMomentDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        aud_dim=args.a_feat_dim if "a_feat_dim" in args else 0,
        aux_loss=args.aux_loss,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        span_loss_type=args.span_loss_type,
        n_input_proj=args.n_input_proj,
        compress_len=args.compress_len,
        use_vid_compression=args.use_vid_compression,
        weight_alpha=args.weight_alpha
    )

    matcher = build_matcher(args)

    # 定义 weight_dict，确保与 SetCriterionOl 中的损失项匹配
    weight_dict = {
        "loss_start_label": args.start_label_loss_coef,  # 帧分类损失的权重
        "loss_middle_label": args.middle_label_loss_coef,  
        "loss_end_label": args.end_label_loss_coef,  
        "loss_saliency": args.saliency_loss_coef,   # 显著性分数损失的权重
        # "loss_spans": args.span_loss_coef,  # span损失的权重
    }
    
    # 条件性添加损失项
    losses = ['labels', 'saliency', 'span']
    
    # loss_label消融实验
    # losses = ['labels', 'saliency']

    
    criterion = SetCriterionOl(
        weight_dict=weight_dict, 
        losses=losses,  # 传入条件性的损失列表
        saliency_margin=args.saliency_margin,
    )

    criterion.to(device)
    return model, criterion