import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lighthouse.common.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx
from lighthouse.common.position_encoding import build_position_encoding
from lighthouse.common.matcher import build_matcher
from lighthouse.common.misc import accuracy
from lighthouse.common.moment_transformer import build_transformer


class OLMomentDETR(nn.Module):
    """ This is the Moment-DETR module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False, max_v_l=75, span_loss_type="l1", 
                 use_txt_pos=False, n_input_proj=2, aud_dim=0):
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
        # self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground

        self.frame_class_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 输出维度仍为4，但表示4个二分类的logits

        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        # self.foreground_thd = foreground_thd
        # self.background_thd = background_thd

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

        self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.aux_loss = aux_loss

    def forward(self, src_txt, src_txt_mask, src_vid, memory_len, src_vid_mask, chunk_idx, src_aud=None, src_aud_mask=None):
        """
        在线模型forward需要的输入:
                -src_txt: [batch_size, L_txt, D_txt]
                -src_txt_mask: [batch_size, L_txt],
                -src_vid: [batch_size, L_vid, D_vid], 
                    L_vid = long_mempry_sample_length + short_mempry_sample_length + future_mempry_sample_length
                -memory_len: [], 三种记忆的采样长度
        返回值为包含以下内容的dict:
                -"frame_pred": short记忆的逐帧预测概率
                -"saliency_scores": short记忆的逐帧saliency score

        """

        print(f"[DEBUG] OLMomentDETR forward - Batch size: {src_txt.shape[0]}, Memory lengths: {memory_len}")
        # 修改后：使用long+short memory的queries
        self.num_queries = memory_len[0] + memory_len[1]  # long + short

        #   拼接音频特征
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
        #   投影输入特征
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        #   拼接视频和文本的特征，掩码

        # print("src_vid:",src_vid.shape,"src_txt:",src_txt.shape)
        # input("请按回车键继续...")
        
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        #   分别生成位置编码并拼接
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)
        #   通过transformer
        hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
        # hs.shape = (num_decoder_layers, batch_size, num_queries, hidden_dim)


        #   使用掩码确定实际的long memory长度
        long_memory_mask = src_vid_mask[:, :memory_len[0]]
        actual_long_len = long_memory_mask.sum(dim=1).max().item()  # 获取batch中最长的有效长度
        
        # 使用实际长度进行切片
        short_start = min(memory_len[0], actual_long_len)
        short_end = short_start + memory_len[1]

        out = {'frame_pred': self.frame_class_embed(hs)[-1][:, short_start:short_end]}
        
        print("hs shape:", hs.shape)
        print(self.frame_class_embed(hs)[-1].shape)
        input("in forward")
        print(f"[DEBUG] Frame predictions shape: {out['frame_pred'].shape}")

        # print("actual_long_len:",actual_long_len,"memory_len:",memory_len)
        # print("short_memory_start:",short_start)
        # print("short_memory_end:",short_end)
        # print("out['frame_pred']:",out['frame_pred'])
        # input("请按回车键继续...")

        #   saliency score的计算可以沿用
        # txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)

        st=memory_len[0]
        ed=st + memory_len[1]
        vid_mem_short = vid_mem[:, st:ed, :]  # (bsz, ed - st, d)

        out["saliency_scores"] = self.saliency_proj(vid_mem_short).squeeze(-1)  # (bsz, short_memory_length)

        if not self.training:
            out["frame_pred"] = out["frame_pred"][:, -1, :].unsqueeze(1)  # 只取最后一个帧的预测,变成 (bsz, 1, 4)
            # out["saliency_scores"] = out["saliency_scores"][:,-1]   
            # saliency不能只取最后一帧，否则更没法做对比loss之类的了

        # 添加 chunk_idx 到输出
        out["chunk_idx"] = chunk_idx  # [batch_size]

        # if self.aux_loss:
        #     # assert proj_queries and proj_txt_mem
        #     out['aux_outputs'] = [
        #         {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        # print("out:",out)

        return out

class SetCriterionOl(nn.Module):
    """ This class computes the loss for ol_DETR.
    主要计算两部分的loss:
        1) 帧级别的分类损失 [start, mid, end, irrelevant].
        2) saliency score 预测结果的损失.
    还考虑添加一个时序一致性loss来约束saliency score, 让相邻帧的saliency score预测结果变化更加平滑
    """

    def __init__(self, weight_dict, losses, saliency_margin=1, use_consistency_loss=True, gamma=2):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            saliency_margin: float, margin for saliency loss.
            use_consistency_loss: bool, whether to use consistency loss.
            gamma: float, weight factor for hard sample mining in label loss.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.saliency_margin = saliency_margin
        self.use_consistency_loss = use_consistency_loss
        self.gamma = gamma  # 添加 gamma 作为初始化参数

    # def loss_labels(self, outputs, targets, indices, log=True):
    #     """修改后的帧分类损失函数"""
    #     frame_pred = outputs['frame_pred']  # (batch_size, num_frames, 4)
        
    #     losses = {}
    #     total_loss = 0

    #     # 使用middle_label替代semantic_label
    #     label_types = ['start_label', 'middle_label', 'end_label']
    #     for i, label_type in enumerate(label_types):
    #         target = targets[label_type].float()
    #         frame_probs = torch.sigmoid(frame_pred)
    #         pred = frame_probs[:, :, i]

    #         # 计算加权BCE损失
    #         loss = F.binary_cross_entropy_with_logits(
    #             pred, target, reduction='none'
    #         )
    #         # 动态权重,赋予较难的类别较高的权重
    #         weight = (target * (1 - target)).pow(self.gamma)  # 使用 gamma 计算权重
    #         # weight = 1
    #         loss = (loss * weight).mean()
    #         if(np.isnan(loss.item())):
    #             print("target:", target)
    #             print("pred:", pred)
    #             print("loss:", loss)
    #             input("wait")
            
    #         losses[f'loss_{label_type}'] = loss
    #         total_loss += loss
        
    #     # 条件性添加时序平滑损失
    #     if self.use_consistency_loss:
    #         pred_probs = torch.sigmoid(pred)  # 使用sigmoid获取概率
    #         temporal_smooth = torch.mean((pred_probs[:, 1:] - pred_probs[:, :-1]).pow(2))
    #         losses['loss_label'] = total_loss / 3.0 + 0.1 * temporal_smooth

    #     return losses

    # def loss_saliency(self, outputs, targets, indices, log=True):
    #     """显著性损失"""
    #     print("targets:", targets)
    #     print("outputs:", outputs)
    #     input("in loss saliency: wait! ")

    #     if ("saliency_scores" not in outputs or 
    #         "saliency_pos_labels" not in targets or 
    #         "short_memory_start" not in targets):
    #         print("info lacked to compute loss saliency")
    #         return {"loss_saliency": 0}

    #     saliency_scores = outputs["saliency_scores"]
    #     pos_indices = targets["saliency_pos_labels"]
    #     neg_indices = targets["saliency_neg_labels"]
        
    #     # 获取每个样本的short_memory_start
    #     short_memory_starts = targets["short_memory_start"]["spans"]
        
    #     # 收集有效样本
    #     valid_pos_scores = []
    #     valid_neg_scores = []
    #     for i in range(len(pos_indices)):
    #         if i >= len(short_memory_starts):  # 安全检查
    #             continue
            
    #         current_start = short_memory_starts[i]  # 获取当前样本的short_memory_start
    #         pos_idx = pos_indices[i]
    #         neg_idx = neg_indices[i]
            
    #         # 跳过无效的索引
    #         if pos_idx == -1 or neg_idx == -1:
    #             continue
            
    #         # 转换为相对于short memory的局部索引
    #         pos_idx_local = pos_idx - current_start
    #         neg_idx_local = neg_idx - current_start
            
    #         if (0 <= pos_idx_local < saliency_scores.shape[1] and 
    #             0 <= neg_idx_local < saliency_scores.shape[1]):
    #             valid_pos_scores.append(saliency_scores[i, pos_idx_local])
    #             valid_neg_scores.append(saliency_scores[i, neg_idx_local])

    #     if not valid_pos_scores or not valid_neg_scores:
    #         return {"loss_saliency": 0}

    #     valid_pos_scores = torch.stack(valid_pos_scores)
    #     valid_neg_scores = torch.stack(valid_neg_scores)
        
    #     # 1. 基础对比损失
    #     base_loss = torch.clamp(
    #         self.saliency_margin + valid_neg_scores - valid_pos_scores, 
    #         min=0
    #     ).mean()

    #     # 2. 添加L2正则化
    #     l2_reg = 0.01 * (valid_pos_scores.pow(2).mean() + valid_neg_scores.pow(2).mean())

    #     # 3. 条件性添加平滑损失
    #     loss = base_loss + l2_reg
    #     if self.use_consistency_loss:
    #         smooth_loss = 0.1 * torch.mean(
    #             (saliency_scores[:, 1:] - saliency_scores[:, :-1]).pow(2)
    #         )
    #         loss += smooth_loss

    #     # print("sal loss ",loss)
    #     # input("wait")

    #     return {
    #         "loss_saliency": loss
    #     }

    def loss_labels(self, outputs, targets, indices, log=True):
        """帧分类损失函数，先筛选有效样本，再批量计算损失"""
        frame_pred = outputs['frame_pred']  # (batch_size, num_frames, 4)
        chunk_idx = outputs.get('chunk_idx', None)  # 获取 chunk_idx

        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs. Ensure it is passed correctly.")

        # # 根据训练模式调整 frame_pred 的形状
        # if not self.training:
        #     # 非训练模式：frame_pred 的形状是 (batch_size, 4)
        #     frame_pred = frame_pred.unsqueeze(1)  # 扩展为 (batch_size, 1, 4)

        print("first 5 frame_pred:", frame_pred[:5])
        print("first 5 targets:", targets['meta'][:5])
        input("wait")

        # 筛选有效样本
        valid_preds = []
        valid_start_labels = []
        valid_middle_labels = []
        valid_end_labels = []

        for i, idx in enumerate(chunk_idx):
            # 找到 targets 中对应的样本
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue  # 如果找不到对应的 target，跳过该样本

            # 获取当前样本的标签
            start_label = target_meta['start_label']  # (num_frames,)
            middle_label = target_meta['middle_label']  # (num_frames,)
            end_label = target_meta['end_label']  # (num_frames,)

            # 在非训练模式下，仅取最后一帧的标签
            if not self.training:
                start_label = start_label[-1:]  # 取最后一帧
                middle_label = middle_label[-1:]  # 取最后一帧
                end_label = end_label[-1:]  # 取最后一帧

            # 获取当前样本的预测
            pred = frame_pred[i]  # (num_frames, 4)

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


            # # 看下这些变量都在什么设备上
            # print(f"{label_type} loss: {pred_slice.device}, {target.device}")
            # print("pred_slice:",pred_slice)
            # print("target:",target)
            # input("Press Enter to continue...")

            # 计算加权BCE损失
            loss = F.binary_cross_entropy_with_logits(
                pred_slice, target, reduction='none'
            )

            # 动态权重，赋予较难的类别较高的权重
            weight = (target * (1 - target)).pow(self.gamma)  # 使用 gamma 计算权重
            loss = (loss * weight).mean()

            if torch.isnan(loss).any():
                print(f"NaN detected in {label_type} loss. Target: {target}, Pred: {pred_slice}")
                input("Press Enter to continue...")

            losses[f'loss_{label_type}'] = loss
            total_loss += loss

        # 平均损失
        losses['loss_label'] = total_loss / 3  # 3 表示 start, middle, end

        # 条件性添加时序平滑损失
        if self.use_consistency_loss:
            pred_probs = torch.sigmoid(valid_preds)  # 使用sigmoid获取概率
            temporal_smooth = torch.mean((pred_probs[:, :, 1:] - pred_probs[:, :, :-1]).pow(2))
            losses['loss_label'] += 0.1 * temporal_smooth

        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
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

        # # 根据训练模式调整 saliency_scores 的形状
        # if not self.training:
        #     # 非训练模式：saliency_scores 的形状是 (batch_size,)
        #     saliency_scores = saliency_scores.unsqueeze(-1)  # 扩展为 (batch_size, 1)

        # 筛选有效样本
        valid_scores = []
        valid_pos_labels = []
        valid_neg_labels = []

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
            pred_scores = saliency_scores[i]  # (num_frames,)
            short_memory_start = target_meta['short_memory_start']

            # 转换为相对于 short memory 的局部索引
            pos_indices = [pos_idx - short_memory_start for pos_idx in pos_labels]
            neg_indices = [neg_idx - short_memory_start for neg_idx in neg_labels]

            # 确保索引在有效范围内
            pos_indices = [idx for idx in pos_indices if 0 <= idx < pred_scores.shape[0]]
            neg_indices = [idx for idx in neg_indices if 0 <= idx < pred_scores.shape[0]]

            if not pos_indices or not neg_indices:
                continue  # 如果没有有效索引，跳过该样本

            # 添加到有效样本列表
            valid_scores.append(pred_scores)
            valid_pos_labels.append(pos_indices)
            valid_neg_labels.append(neg_indices)

        if not valid_scores:
            return {"loss_saliency": 0}

        # 将有效样本组织成批量张量
        valid_scores = torch.stack(valid_scores)  # (num_valid_samples, num_frames)

        # 批量计算损失
        base_losses = []
        for i in range(len(valid_scores)):
            pos_scores = valid_scores[i, valid_pos_labels[i]]  # 正样本分数
            neg_scores = valid_scores[i, valid_neg_labels[i]]  # 负样本分数

            # 基础对比损失
            base_loss = torch.clamp(
                self.saliency_margin + neg_scores - pos_scores, 
                min=0
            ).mean()

            # 添加L2正则化
            l2_reg = 0.01 * (pos_scores.pow(2).mean() + neg_scores.pow(2).mean())
            base_loss += l2_reg

            base_losses.append(base_loss)

        # 平均所有样本的损失
        final_loss = torch.mean(torch.stack(base_losses))

        # 条件性添加平滑损失
        if self.use_consistency_loss:
            smooth_loss = 0.1 * torch.mean(
                (valid_scores[:, 1:] - valid_scores[:, :-1]).pow(2)
            )
            final_loss += smooth_loss

        return {"loss_saliency": final_loss}

    def loss_temporal_consistency(self, outputs, targets, indices):
        """Temporal consistency loss for smooth predictions"""
        if "saliency_scores" not in outputs:
            return {"loss_temporal_consistency": 0}
       
        saliency_scores = outputs["saliency_scores"]  # (batch_size, #frames)
        saliency_probs = torch.sigmoid(saliency_scores)  # (batch_size, #frames)

        # Compute L2 loss between adjacent frames
        loss_temporal = torch.mean((saliency_probs[:, 1:] - saliency_probs[:, :-1]) ** 2)
        return {"loss_temporal_consistency": loss_temporal}

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "temporal_consistency": self.loss_temporal_consistency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices=None))

        # Add temporal consistency loss if needed
        if "temporal_consistency" in self.losses:
            losses.update(self.loss_temporal_consistency(outputs, targets, indices=None))

        total_loss = sum(losses[k] * self.weight_dict.get(k, 1.0) for k in losses.keys())
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
    )

    matcher = build_matcher(args)

    # 定义 weight_dict，确保与 SetCriterionOl 中的损失项匹配
    weight_dict = {
        "loss_start_label": args.start_label_loss_coef,  # 帧分类损失的权重
        "loss_middle_label": args.middle_label_loss_coef,  
        "loss_end_label": args.end_label_loss_coef,  
        "loss_saliency": args.saliency_loss_coef,   # 显著性分数损失的权重
    }
    # 如果使用一致性损失，为weight_dict添加"loss_temporal_consistency"
    if args.use_consistency_loss:
        weight_dict["loss_temporal_consistency"] = args.temporal_consistency_loss_coef

    # 根据参数决定是否使用一致性损失
    use_consistency = args.use_consistency_loss if hasattr(args, 'use_consistency_loss') else True
    
    # 条件性添加损失项
    losses = ['labels', 'saliency']
    if use_consistency:
        losses.append('temporal_consistency')
    
    criterion = SetCriterionOl(
        weight_dict=weight_dict, 
        losses=losses,  # 传入条件性的损失列表
        saliency_margin=args.saliency_margin,
        use_consistency_loss=use_consistency
    )

    criterion.to(device)
    return model, criterion