import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGuidedVideoAttention(nn.Module):
    """
    基于文本引导的视频特征注意力模块
    
    输入:
        Ft: 文本特征，形状为 (batch_size, num_token, t_dim)
        Fv: 视频特征，形状为 (batch_size, vid_len, v_dim)
        vid_mask: 视频掩码，形状为 (batch_size, vid_len)，True/1表示有效位置
        text_mask: 文本掩码，形状为 (batch_size, num_token)，True/1表示有效位置
        
    输出:
        Fv_enhanced: 增强后的视频特征，保持原始形状 (batch_size, vid_len, v_dim)
    """
    
    def __init__(self, t_dim, v_dim, hidden_dim=512, num_heads=8, dropout=0.1):
        """
        初始化文本引导视频注意力模块
        
        参数:
            t_dim: 文本特征维度
            v_dim: 视频特征维度
            hidden_dim: 注意力机制内部隐藏维度
            num_heads: 多头注意力的头数
            dropout: Dropout率
        """
        super(TextGuidedVideoAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 确保hidden_dim可以被num_heads整除
        assert hidden_dim % num_heads == 0, "隐藏维度必须能被头数整除"
        
        # 投影层 - 视频特征作为query，文本特征作为key和value
        self.v_proj = nn.Linear(v_dim, hidden_dim)
        self.t_key_proj = nn.Linear(t_dim, hidden_dim)
        self.t_val_proj = nn.Linear(t_dim, hidden_dim)
        
        # 输出投影层
        self.out_proj = nn.Linear(hidden_dim, v_dim)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.norm = nn.LayerNorm(v_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(v_dim, 4 * v_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * v_dim, v_dim),
            nn.Dropout(dropout)
        )
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(v_dim)
        
    def forward(self, Ft, Fv, vid_mask=None, text_mask=None):
        """
        前向传播
        
        参数:
            Ft: 文本特征，形状为 (batch_size, num_token, t_dim)
            Fv: 视频特征，形状为 (batch_size, vid_len, v_dim)
            vid_mask: 视频掩码，形状为 (batch_size, vid_len)，True/1表示有效位置
            text_mask: 文本掩码，形状为 (batch_size, num_token)，True/1表示有效位置
            
        返回:
            Fv_enhanced: 增强后的视频特征，形状为 (batch_size, vid_len, v_dim)
        """
        batch_size, vid_len, _ = Fv.size()
        _, num_token, _ = Ft.size()
        
        # 1. 投影query, key, value
        q = self.v_proj(Fv)           # (batch_size, vid_len, hidden_dim)
        k = self.t_key_proj(Ft)       # (batch_size, num_token, hidden_dim)
        v = self.t_val_proj(Ft)       # (batch_size, num_token, hidden_dim)
        
        # 2. 分割多头
        head_dim = self.hidden_dim // self.num_heads
        
        # 重塑为多头格式
        q = q.view(batch_size, -1, self.num_heads, head_dim)  
        k = k.view(batch_size, -1, self.num_heads, head_dim)
        v = v.view(batch_size, -1, self.num_heads, head_dim)
        
        # 交换维度以便于计算
        q = q.transpose(1, 2)  # (batch_size, num_heads, vid_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, num_token, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, num_token, head_dim)
        
        # 3. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # 应用文本掩码（如果提供）
        if text_mask is not None:
            # 扩展文本掩码以匹配注意力分数的形状
            # (batch_size, num_token) -> (batch_size, 1, 1, num_token)
            expanded_text_mask = text_mask.unsqueeze(1).unsqueeze(2)
            
            # 在不有效的文本位置应用大的负值
            scores = scores.masked_fill(~expanded_text_mask.bool(), -1e9)
        
        # 4. 应用softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, vid_len, num_token)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 5. 加权聚合文本信息
        weighted_v = torch.matmul(attn_weights, v)
        
        # 6. 合并多头结果
        weighted_v = weighted_v.transpose(1, 2).contiguous()
        weighted_v = weighted_v.view(batch_size, vid_len, self.hidden_dim)
        
        # 7. 应用输出投影
        attn_output = self.out_proj(weighted_v)
        attn_output = self.proj_dropout(attn_output)
        
        # 8. 残差连接和层归一化
        attn_output = self.norm(Fv + attn_output)
        
        # 9. 前馈网络
        ff_output = self.feed_forward(attn_output)
        
        # 10. 残差连接和最终层归一化
        Fv_enhanced = self.final_norm(attn_output + ff_output)
        
        # 11. 应用视频掩码（如果提供）
        if vid_mask is not None:
            # 扩展掩码维度以匹配输出形状
            mask_expanded = vid_mask.unsqueeze(-1).float()  # (batch_size, vid_len, 1)
            Fv_enhanced = Fv_enhanced * mask_expanded
        
        return Fv_enhanced

class VideoMemoryCompressorWithExternalWeights(nn.Module):
    """
    基于注意力机制的视频记忆压缩模块，支持融合外部权重
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        mask: 指示有效位置的掩码，形状为 (batch_size, vid_len)。这个掩码同时应用于Fv和extern_weight
        extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
        
    输出:
        Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
        combined_weights: 注意力权重与外部权重的平均，形状为 (batch_size, compress_len, vid_len)
    """

    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1):
        """
        初始化压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 压缩后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
        """
        super(VideoMemoryCompressorWithExternalWeights, self).__init__()

        self.dimension = dimension
        self.compress_len = compress_len
        self.num_heads = num_heads
        
        # 确保dimension可以被num_heads整除
        assert dimension % num_heads == 0, "维度必须能被头数整除"

        # 可学习的查询参数，将作为cross-attention中的query
        # 形状: (1, compress_len, dimension)
        self.query = nn.Parameter(torch.randn(1, compress_len, dimension))
        
        # 初始化查询参数
        nn.init.xavier_uniform_(self.query)
        
        # 多头注意力层 - 不直接使用nn.MultiheadAttention，因为需要获取并修改注意力权重
        # 因此需要自己实现多头注意力机制的核心部分

        # 为query, key, value创建线性变换
        self.q_proj = nn.Linear(dimension, dimension)
        self.k_proj = nn.Linear(dimension, dimension)
        self.v_proj = nn.Linear(dimension, dimension)
        
        # 输出投影
        self.out_proj = nn.Linear(dimension, dimension)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # 层归一化和残差连接
        self.norm = nn.LayerNorm(dimension)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
            nn.Dropout(dropout)
        )
        
        # 最终的层归一化
        self.final_norm = nn.LayerNorm(dimension)

    def forward(self, Fv, extern_weight=None, mask=None, weight_alpha=0.5):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            weight_alpha: 注意力权重的权重，1-weight_alpha为外部权重的权重
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            combined_weights: 注意力权重与外部权重的平均，形状为 (batch_size, compress_len, vid_len)
        """
        
        batch_size, vid_len, _ = Fv.size()
        
        # 扩展查询以匹配批次大小
        # 从 (1, compress_len, dimension) 到 (batch_size, compress_len, dimension)
        query = self.query.expand(batch_size, -1, -1)

        # 1. 投影query, key, value
        q = self.q_proj(query)  # (batch_size, compress_len, dimension)
        k = self.k_proj(Fv)     # (batch_size, vid_len, dimension)
        v = self.v_proj(Fv)     # (batch_size, vid_len, dimension)
        
        # 2. 分割多头
        head_dim = self.dimension // self.num_heads
        
        # 重塑为多头格式
        # (batch_size, seq_len, dimension) -> (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, -1, self.num_heads, head_dim)
        k = k.view(batch_size, -1, self.num_heads, head_dim)
        v = v.view(batch_size, -1, self.num_heads, head_dim)
        
        # 交换维度以便于计算
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)  # (batch_size, num_heads, compress_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, vid_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, vid_len, head_dim)
        
        # 3. 计算注意力分数
        # (batch_size, num_heads, compress_len, head_dim) @ (batch_size, num_heads, head_dim, vid_len)
        # -> (batch_size, num_heads, compress_len, vid_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # 应用掩码（如果提供）
        if mask is not None:
            # 扩展掩码以适应多头注意力的形状
            # (batch_size, vid_len) -> (batch_size, 1, 1, vid_len)
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            
            # 在不有效的位置应用大的负值
            scores = scores.masked_fill(~expanded_mask.bool(), -1e9)
        
        # 应用softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, compress_len, vid_len)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 4. 如果提供了外部权重，将其与注意力权重融合
        combined_weights = None
        if extern_weight is not None:
            # 确保extern_weight的形状正确
            assert extern_weight.shape == (batch_size, vid_len), \
                f"外部权重形状应为 {(batch_size, vid_len)}, 但得到了 {extern_weight.shape}"
            
            # 首先将extern_weight形状由(batch_size, vid_len)变为(batch_size, compress_len, vid_len)
            # 扩展后的维度1应该为compress_len，每个维度2的值相同
            extern_weight = extern_weight.unsqueeze(1).expand(-1, self.compress_len, -1)

            # 首先处理外部权重的掩码，同样使用传入的mask
            masked_extern_weight = extern_weight.clone()

            if mask is not None:
                # 将mask应用于extern_weight的vid_len维度
                # mask: (batch_size, vid_len)
                # extern_weight: (batch_size, compress_len, vid_len)
                mask_expanded = mask.unsqueeze(1)  # (batch_size, 1, vid_len)

                masked_extern_weight = masked_extern_weight.masked_fill(~mask_expanded.bool(), 0)
                
                # 重新归一化外部权重，使每行的和为1
                # 对于全为0的行（即所有位置都被掩码的情况），添加一个小的epsilon以避免除以0
                row_sums = masked_extern_weight.sum(dim=-1, keepdim=True)
                masked_extern_weight = masked_extern_weight / (row_sums + 1e-9)

            # 扩展外部权重以匹配多头注意力的形状
            # (batch_size, compress_len, vid_len) -> (batch_size, 1, compress_len, vid_len)
            extern_weight_expanded = masked_extern_weight.unsqueeze(1)

            # 融合两种权重
            combined_weights = weight_alpha * attn_weights + (1 - weight_alpha) * extern_weight_expanded
            
            # 确保权重总和为1
            # 由于已经对两个输入分别进行了softmax/归一化，并且是线性组合，
            # 所以combined_weights的每一行的和应该接近1，但为了确保精确性，我们再次归一化
            combined_weights_sum = combined_weights.sum(dim=-1, keepdim=True)
            combined_weights = combined_weights / (combined_weights_sum + 1e-9)
            
            # 使用融合后的权重计算加权和
            # (batch_size, num_heads, compress_len, vid_len) @ (batch_size, num_heads, vid_len, head_dim)
            # -> (batch_size, num_heads, compress_len, head_dim)
            weighted_v = torch.matmul(combined_weights, v)
        else:
            # 如果没有外部权重，就使用原始注意力权重
            weighted_v = torch.matmul(attn_weights, v)
            combined_weights = attn_weights

        # 5. 合并多头结果
        # (batch_size, num_heads, compress_len, head_dim) -> (batch_size, compress_len, num_heads, head_dim)
        weighted_v = weighted_v.transpose(1, 2).contiguous()
        
        # (batch_size, compress_len, num_heads, head_dim) -> (batch_size, compress_len, dimension)
        weighted_v = weighted_v.view(batch_size, self.compress_len, self.dimension)

        # 6. 应用输出投影
        attn_output = self.out_proj(weighted_v)
        
        # 7. 残差连接和层归一化
        attn_output = self.norm(query + attn_output)
        
        # 8. 前馈网络
        ff_output = self.feed_forward(attn_output)
        
        # 9. 残差连接和最终层归一化
        Fv_compress = self.final_norm(attn_output + ff_output)

        # 10. 将多头注意力权重平均为单头形式以便返回
        # (batch_size, num_heads, compress_len, vid_len) -> (batch_size, compress_len, vid_len)
        combined_weights_single = combined_weights.mean(dim=1)

        # 创建压缩后的掩码，形状为 (batch_size, compress_len)，全为True
        compress_mask = torch.ones((batch_size, self.compress_len), dtype=torch.bool, device=Fv.device)
        
        return Fv_compress, compress_mask, combined_weights_single


class MultimodalTokenCompressor(nn.Module):
    def __init__(self, v_dim, compress_len, num_heads=8, dropout=0.1):
        super().__init__()
        self.compress_len = compress_len
        self.token_learner = MultimodalTokenLearner(
            hidden_dim=v_dim,
            num_tokens=compress_len,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, video_feats, query=None):
        """
        Args:
            video_feats: (bsz, v_len, v_dim)
            query: (bsz, q_len, v_dim) 或 None（若无需多模态）
        Returns:
            compressed_feats: (bsz, compress_len, v_dim)
            weights: (bsz, compress_len, v_len)
        """
        if query is None:
            # 若无查询，用均值作为伪查询
            query = video_feats.mean(dim=1, keepdim=True)  # (bsz, 1, v_dim)
        
        compressed_feats = self.token_learner(video_feats, query)
        
        # 若需返回注意力权重（需修改MultimodalTokenLearner以暴露权重）
        weights = None  # 可自行实现
        return compressed_feats, weights

class MultimodalTokenLearner(nn.Module):
    def __init__(self, hidden_dim, num_tokens, num_heads, dropout):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        self.build_tokenlearner(hidden_dim, num_tokens, dropout)

    def build_tokenlearner(self, hidden_dim, num_tokens, dropout):
        # Vision tokenlearner
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens),
            nn.Dropout(dropout)
        )
        self.video_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim<<1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        # Linguistic tokenlearner
        self.fc_feat = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_query = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.w = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, num_tokens, bias=False)
        )
        self.query_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim<<1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def tokenlearner(self, input, query):
        # [B, T, h]
        selected = self.input_mlp(input)
        # [B, T, n]
        selected = selected.transpose(1, 2)
        selected = F.softmax(selected, -1)
        # [B, n, T] 
        feat_selected = torch.einsum('...nt,...td->...nd', selected, input)
        # [B, n, h]

        query = query.mean(1, keepdim=True)
        # [B, T, h] [B, 1, h]
        attn = self.fc_feat(input) + self.fc_query(query)
        # [B, T, h]
        attn = self.w(attn).transpose(1, 2)
        attn = F.softmax(attn, dim=-1)
        # [B, n, T]
        feat_query = torch.einsum('...nt,...td->...nd', attn, input)
        # [B, n, h]

        feat = self.norm(
            self.video_gate(input.mean(1, keepdim=True)) * feat_selected + 
            self.query_gate(query) * feat_query)
        
        return feat

    def forward(self, input, query):
        feat = self.tokenlearner(input, query)
        return feat
