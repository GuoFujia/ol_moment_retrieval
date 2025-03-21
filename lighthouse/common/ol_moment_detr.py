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
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.frame_class_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 输出维度仍为4，但表示4个二分类的logits
        self.saliency_proj = nn.Linear(hidden_dim, 1)

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
                -"pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
                -"pred_logits": The class prediction for all queries.
                -"frame_pred": short记忆的逐帧预测概率
                -"saliency_scores": short记忆的逐帧saliency score
                -"chunk_idx": chunk index

        """


        #   拼接音频特征
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
        #   投影输入特征
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        #   拼接视频和文本的特征，掩码
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        #   分别生成位置编码并拼接
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)
        #   通过transformer
        hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
        # hs.shape = (num_decoder_layers, batch_size, num_queries, hidden_dim)

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
        # 如果是测试阶段，只取最后一个帧的预测,并保持形状
        if not self.training:
            out["frame_pred"] = out["frame_pred"][:, -1, :].unsqueeze(1)            # (bsz, 1, 4)
            out["saliency_scores"] = out["saliency_scores"][:,-1].unsqueeze(1)      # (bsz, 1)

        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        # 添加 chunk_idx 到输出
        out["chunk_idx"] = chunk_idx  # [batch_size]

        return out

class SetCriterionOl(nn.Module):
    """ This class computes the loss for ol_DETR.
    主要计算两部分的loss:
        1) 帧级别的分类损失 [start, mid, end, irrelevant].
        2) saliency score 预测结果的损失.
    还考虑添加一个时序一致性loss来约束saliency score, 让相邻帧的saliency score预测结果变化更加平滑
    """

    def __init__(self, weight_dict, losses, saliency_margin=1, gamma=2):
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

    def loss_spans(self, outputs, targets):
        frame_pred = outputs['frame_pred']  # (batch_size, short_memory_len, 4)
        chunk_idx = outputs.get('chunk_idx', None)
        
        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs")
        
        device = frame_pred.device
        batch_size = frame_pred.shape[0]
        seq_len = frame_pred.shape[1]
        
        # 初始化 loss_v1，不需要梯度
        loss_v1 = torch.tensor(0.0, device=device, requires_grad=False)
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

            # 计算加权BCE损失
            loss = F.binary_cross_entropy_with_logits(
                pred_slice, target, reduction='none'
            )

            # 动态权重，赋予较难的类别较高的权重
            weight = (target * (1 - target)).pow(self.gamma)  # 使用 gamma 计算权重
            loss = (loss * weight).mean()
            # loss = loss.mean()  # 先不用weight

            if torch.isnan(loss).any():
                print(f"NaN detected in {label_type} loss. Target: {target}, Pred: {pred_slice}")
                input("Press Enter to continue...")

            losses[f'loss_{label_type}'] = loss
            total_loss += loss

        # 平均损失
        losses['loss_label'] = total_loss / 3  # 3 表示 start, middle, end

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
            "span": self.loss_spans,
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
    # losses = ['labels', 'saliency', 'span']
    losses = ['saliency', 'span']
    
    criterion = SetCriterionOl(
        weight_dict=weight_dict, 
        losses=losses,  # 传入条件性的损失列表
        saliency_margin=args.saliency_margin,
    )

    criterion.to(device)
    return model, criterion