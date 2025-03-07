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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
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
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
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

    def forward(self, src_txt, src_txt_mask, src_vid, memory_len, src_vid_mask, src_aud=None, src_aud_mask=None):
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

        # # 将logits转换为概率
        # frame_logits = self.frame_class_embed(hs)[-1][:, short_start:short_end]  # (batch_size, num_frames, 4)
        # frame_probs = torch.sigmoid(frame_logits)
        
        # out = {'frame_pred': frame_probs}

        out = {'frame_pred': self.frame_class_embed(hs)[-1][:, short_start:short_end]}

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
        要用这个的话, 就得改一下saliency部分的逻辑
    """

    def __init__(self, weight_dict, losses, saliency_margin=1, use_consistency_loss=True, gamma=3):
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

    def loss_labels(self, outputs, targets, indices, log=True):
        """修改后的帧分类损失函数"""
        frame_pred = outputs['frame_pred']  # (batch_size, num_frames, 4)
        
        losses = {}
        total_loss = 0
        
        # 使用middle_label替代semantic_label
        label_types = ['start_label', 'middle_label', 'end_label']
        for i, label_type in enumerate(label_types):
            target = targets[label_type].float()
            frame_probs = torch.sigmoid(frame_pred)
            pred = frame_probs[:, :, i]
            


            # print("loss type: ", label_type)
            # print("pred: ", pred)
            # print("target: ",target)
            # input("wait")

            # 计算加权BCE损失
            loss = F.binary_cross_entropy_with_logits(
                pred, target, reduction='none'
            )
            
            # 动态权重
            weight = (target * (1 - target)).pow(self.gamma)  # 使用 gamma 计算权重
            loss = (loss * weight).mean()
            
            losses[f'loss_{label_type}'] = loss
            total_loss += loss
        
        # 条件性添加时序平滑损失
        if self.use_consistency_loss:
            pred_probs = torch.sigmoid(pred)  # 使用sigmoid获取概率
            temporal_smooth = torch.mean((pred_probs[:, 1:] - pred_probs[:, :-1]).pow(2))
            losses['loss_label'] = total_loss / 3.0 + 0.1 * temporal_smooth
        
        if log:

            # print("pred: ",torch.sigmoid(pred))
            # print("target: ",target)

            # 计算每个二分类的预测概率率
            pred_binary = torch.sigmoid(pred)

            # print("pred_bin: ",pred_binary)
            # print("target: ",target)

            # accuracy = (pred_binary == target).float().mean() * 100
            # losses['class_error'] = 100 - accuracy

        
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """显著性损失"""

        # print("target",targets.keys())
        # input("wait")

        if ("saliency_scores" not in outputs or 
            "saliency_pos_labels" not in targets or 
            "short_memory_start" not in targets):
            print("infolacked to compute loss saliency")
            return {"loss_saliency": 0}

        saliency_scores = outputs["saliency_scores"]
        pos_indices = targets["saliency_pos_labels"]
        neg_indices = targets["saliency_neg_labels"]
        
        # 获取每个样本的short_memory_start
        short_memory_starts = targets["short_memory_start"]["spans"]
        
        # 收集有效样本
        valid_pos_scores = []
        valid_neg_scores = []
        for i in range(len(pos_indices)):
            if i >= len(short_memory_starts):  # 安全检查
                continue

            # print("short_memory_starts",short_memory_starts)
            # input("wait")
            
            current_start = short_memory_starts[i]  # 获取当前样本的short_memory_start
            pos_idx = pos_indices[i]
            neg_idx = neg_indices[i]
            
            # 跳过无效的索引
            if pos_idx == -1 or neg_idx == -1:
                continue
            
            # 转换为相对于short memory的局部索引
            pos_idx_local = pos_idx - current_start
            neg_idx_local = neg_idx - current_start
            
            if (0 <= pos_idx_local < saliency_scores.shape[1] and 
                0 <= neg_idx_local < saliency_scores.shape[1]):
                valid_pos_scores.append(saliency_scores[i, pos_idx_local])
                valid_neg_scores.append(saliency_scores[i, neg_idx_local])

        if not valid_pos_scores or not valid_neg_scores:
            return {"loss_saliency": 0}

        valid_pos_scores = torch.stack(valid_pos_scores)
        valid_neg_scores = torch.stack(valid_neg_scores)
        
        # 1. 基础对比损失
        base_loss = torch.clamp(
            self.saliency_margin + valid_neg_scores - valid_pos_scores, 
            min=0
        ).mean()

        # 2. 添加L2正则化
        l2_reg = 0.01 * (valid_pos_scores.pow(2).mean() + valid_neg_scores.pow(2).mean())

        # 3. 条件性添加平滑损失
        loss = base_loss + l2_reg
        if self.use_consistency_loss:
            smooth_loss = 0.1 * torch.mean(
                (saliency_scores[:, 1:] - saliency_scores[:, :-1]).pow(2)
            )
            loss += smooth_loss

        # print("sal loss ",loss)
        # input("wait")

        return {
            "loss_saliency": loss
        }

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

        # Weighted sum of losses

        # print("weight_dict",self.weight_dict)
        # print("losses.keys",losses.keys())
        # input("wait")

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