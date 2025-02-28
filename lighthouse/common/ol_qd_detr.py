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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from lighthouse.common.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx
from lighthouse.common.matcher import build_matcher
from lighthouse.common.qd_detr_transformer import build_transformer
from lighthouse.common.position_encoding import build_position_encoding
from lighthouse.common.misc import accuracy
import numpy as np

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class QDDETR(nn.Module):
    """ QD DETR. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=True, max_v_l=75, span_loss_type="l1", 
                 use_txt_pos=False, n_input_proj=2, aud_dim=0):
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

        # 添加逐帧预测头
        self.frame_class_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 预测每个帧的类别（start, mid, end, irrelevant）

        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # 修改为 hidden_dim
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

        # 显著性分数计算
        self.saliency_proj = nn.Linear(hidden_dim, 1)  # 直接预测显著性分数

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

    def forward(self, src_txt, src_txt_mask, src_vid, memory_len, src_vid_mask, src_aud=None, src_aud_mask=None):
        """The forward expects:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt]
               - src_vid: [batch_size, L_vid, D_vid], 
                    L_vid = long_memory_length + short_memory_length + future_memory_length
               - memory_len: [], 三种记忆的采样长度
               - src_vid_mask: [batch_size, L_vid]
        """
        # 设置 num_queries 为短时记忆的长度
        self.num_queries = memory_len[1]

        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        # 添加全局 token
        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        video_length = src_vid.shape[1]
        
        hs, reference, memory, memory_global = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length)

        # 逐帧预测
        out = {'frame_pred': self.frame_class_embed(hs)[-1]}  # (bsz, num_queries, 4)

        # 显著性分数计算
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        st = memory_len[0]
        ed = st + memory_len[1]
        vid_mem_short = vid_mem[:, st:ed, :]  # (bsz, short_memory_length, d)
        out["saliency_scores"] = self.saliency_proj(vid_mem_short).squeeze(-1)  # (bsz, short_memory_length)

        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        return out

class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, eos_coef, losses, 
                 span_loss_type, max_v_l, saliency_margin=1, temporal_consistency_weight=0.1):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin
        self.temporal_consistency_weight = temporal_consistency_weight

        # 逐帧分类的类别权重
        self.frame_class_weights = torch.tensor([10.0, 10.0, 10.0, 0.1], dtype=torch.float32)  # start, mid, end, irrelevant

    def loss_frame_labels(self, outputs, targets, indices, log=True):
        """逐帧分类损失"""
        frame_pred = outputs['frame_pred']  # (batch_size, num_queries, 4)
        frame_labels = targets['frame_labels']  # (batch_size, num_queries)

        loss_ce = F.cross_entropy(frame_pred.transpose(1, 2), frame_labels, weight=self.frame_class_weights, reduction="none")
        losses = {'loss_frame_label': loss_ce.mean()}

        if log:
            pred_classes = torch.argmax(frame_pred, dim=-1)
            correct = (pred_classes == frame_labels).float().sum(dim=1)
            accuracy = (correct / frame_labels.shape[1]).mean() * 100
            losses['frame_class_error'] = 100 - accuracy

        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """显著性分数损失"""
        if "saliency_scores" not in outputs or "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        saliency_scores = outputs["saliency_scores"]  # (batch_size, num_frames)
        pos_indices = targets["saliency_pos_labels"]  # (batch_size, num_pos_pairs)
        neg_indices = targets["saliency_neg_labels"]  # (batch_size, num_neg_pairs)

        # 计算对比损失
        pos_scores = saliency_scores.gather(1, pos_indices)
        neg_scores = saliency_scores.gather(1, neg_indices)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).mean()

        return {"loss_saliency": loss_saliency}

    def forward(self, outputs, targets):
        """计算总损失"""
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices=None))

        # 加权总损失
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
    device = torch.device(args.device)
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = QDDETR(
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

    # 定义 weight_dict，确保与 SetCriterion 中的损失项匹配
    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef,
        "loss_frame_label": args.frame_loss_coef,  # 新增逐帧分类损失权重
        "loss_saliency": args.saliency_loss_coef,  # 显著性分数损失权重
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    # 定义损失项
    losses = ['spans', 'labels', 'frame_labels', 'saliency']  # 新增 'frame_labels'

    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, span_loss_type=args.span_loss_type, 
        max_v_l=args.max_v_l, saliency_margin=args.saliency_margin,
    )
    criterion.to(device)
    return model, criterion