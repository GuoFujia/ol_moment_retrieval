import numpy as np
import torch
import math
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

class SimpleCompressor(nn.Module):
    """
    基于注意力机制的视频记忆压缩模块
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        mask: 指示有效位置的掩码，形状为 (batch_size, vid_len)。这个掩码同时应用于Fv和extern_weight
        
    输出:
        Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
        compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
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
        super(SimpleCompressor, self).__init__()

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
        
        # 多头注意力层 

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

    def forward(self, Fv, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
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
        
        # 4. 原始注意力权重
        weighted_v = torch.matmul(attn_weights, v)
            
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

        # 创建压缩后的掩码，形状为 (batch_size, compress_len)，全为True
        compress_mask = torch.ones((batch_size, self.compress_len), dtype=torch.bool, device=Fv.device)
        
        return Fv_compress, compress_mask

class TextGuidedCompressor(nn.Module):
    """
    基于文本引导的视频记忆压缩模块，使用门控机制融合两个分支的结果
    """
    def __init__(self, t_dim, v_dim, hidden_dim=512, num_heads=8, dropout=0.1, compress_len=10, weight_alpha=None):
        """
        初始化文本引导的视频压缩器
        
        参数:
            t_dim: 文本特征维度
            v_dim: 视频特征维度
            hidden_dim: 注意力机制内部隐藏维度
            num_heads: 多头注意力的头数
            dropout: Dropout率
            compress_len: 压缩后的序列长度
        """
        super().__init__()

        self.compress_len = compress_len

        self.text_guided_video_attention = TextGuidedVideoAttention(
            t_dim=t_dim,
            v_dim=v_dim,
            hidden_dim=hidden_dim)
            
        if weight_alpha is None:
            # 直接视频特征压缩器
            self.direct_memory_compressor = SimpleCompressor(
                v_dim, 
                compress_len=compress_len)
                
            # 文本增强视频特征压缩器
            self.text_enhanced_memory_compressor = SimpleCompressor(
                v_dim, 
                compress_len=compress_len)
        else:
            self.direct_memory_compressor =CompressorWithExternalWeights(
                v_dim, 
                compress_len=compress_len,
                weight_alpha=weight_alpha)
            self.text_enhanced_memory_compressor = CompressorWithExternalWeights(
                v_dim, 
                compress_len=compress_len,
                weight_alpha=weight_alpha)
        # 直接特征门控 - 控制原始视频特征的贡献
        self.direct_gate = nn.Sequential(
            nn.Linear(v_dim, hidden_dim<<1),  # v_dim -> hidden_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),      # hidden_dim*2 -> 1
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        
        # 文本增强特征门控 - 控制文本增强视频特征的贡献
        self.text_enhanced_gate = nn.Sequential(
            nn.Linear(v_dim, hidden_dim<<1),  # v_dim -> hidden_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),      # hidden_dim*2 -> 1
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        
        # 添加层归一化
        self.norm = nn.LayerNorm(v_dim, elementwise_affine=False)

    def forward(self, Ft, Fv, vid_mask=None, text_mask=None, extern_weight=None):
        """
        前向传播
        
        参数:
            Ft: 文本特征，形状为 (batch_size, num_token, t_dim)
            Fv: 视频特征，形状为 (batch_size, vid_len, v_dim)
            vid_mask: 视频掩码，形状为 (batch_size, vid_len)
            text_mask: 文本掩码，形状为 (batch_size, num_token)
            
        返回:
            feat: 融合后的压缩视频特征，形状为 (batch_size, compress_len, v_dim)
            combined_weights: 注意力权重，形状为 (batch_size, num_heads, compress_len, vid_len)
        """
        # 文本引导增强的视频特征 - (batch_size, vid_len, v_dim)
        Fv_enhanced = self.text_guided_video_attention(Ft, Fv, vid_mask, text_mask)

        if extern_weight is None:
            # 压缩文本增强的视频特征 - (batch_size, compress_len, v_dim)
            Fv_compress_enhanced, combined_weights = self.text_enhanced_memory_compressor(Fv_enhanced, mask=vid_mask)
            
            # 压缩原始视频特征 - (batch_size, compress_len, v_dim)
            Fv_compress, _ = self.direct_memory_compressor(Fv, mask=vid_mask)
        else:
            # 压缩文本增强的视频特征 - (batch_size, compress_len, v_dim)
            Fv_compress_enhanced, combined_weights = self.text_enhanced_memory_compressor(Fv_enhanced, mask=vid_mask, extern_weight=extern_weight)
            
            # 压缩原始视频特征 - (batch_size, compress_len, v_dim)
            Fv_compress, _ = self.direct_memory_compressor(Fv, mask=vid_mask, extern_weight=extern_weight)

        # 计算门控值
        # Fv_compress.mean(1, keepdim=True) - (batch_size, 1, v_dim)
        # self.direct_gate(Fv_compress.mean(1, keepdim=True)) - (batch_size, 1, 1)
        direct_gate_value = self.direct_gate(Fv_compress.mean(1, keepdim=True))
        
        # Fv_compress_enhanced.mean(1, keepdim=True) - (batch_size, 1, v_dim)
        # self.text_enhanced_gate(...) - (batch_size, 1, 1)
        text_enhanced_gate_value = self.text_enhanced_gate(Fv_compress_enhanced.mean(1, keepdim=True))

        # --- 融合两个分支特征 ---
        # direct_gate_value * Fv_compress - (batch_size, compress_len, v_dim)
        # text_enhanced_gate_value * Fv_compress_enhanced - (batch_size, compress_len, v_dim)
        feat = self.norm(
            direct_gate_value * Fv_compress + 
            text_enhanced_gate_value * Fv_compress_enhanced
        )

        return feat, combined_weights

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

    def forward(self, Fv, query=None, mask=None, query_mask=None):
        """
        Args:
            Fv: (bsz, v_len, v_dim)
            query: (bsz, q_len, v_dim)
            mask: [B, T]
        Returns:
            compressed_feats: (bsz, compress_len, v_dim)
            weights: (bsz, compress_len, v_len)
        """
        # if query is None:
        #     # 若无查询，用均值作为伪查询
        #     query = video_feats.mean(dim=1, keepdim=True)  # (bsz, 1, v_dim)
        assert query is not None, "Query must be provided for multimodal compression"
        
        compressed_feats = self.token_learner(Fv, query, mask, query_mask)
        # 创建压缩后的掩码，形状为 (batch_size, compress_len)，全为True
        compress_mask = torch.ones((Fv.shape[0], self.compress_len), dtype=torch.bool, device=Fv.device)
        return compressed_feats, compress_mask

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


    def tokenlearner(self, input, query, input_mask=None, query_mask=None):
        """Args:
            input:  [B, T, h] 视频特征
            query:  [B, q_len, h] 查询特征
            input_mask: [B, T]
            query_mask: [B, q_len]
        """
        # --- 视觉TokenLearner分支 ---
        selected = self.input_mlp(input)  # [B, T, n]
        selected = selected.transpose(1, 2)  # [B, n, T]
        
        # 应用掩码：将无效位置的权重设为负无穷
        if input_mask is not None:
            input_mask = input_mask.unsqueeze(1)  # [B, 1, T]
            selected = selected.masked_fill(~input_mask.bool(), -1e9)
        
        selected = F.softmax(selected, dim=-1)  # [B, n, T]
        feat_selected = torch.einsum('...nt,...td->...nd', selected, input)  # [B, n, h]

        # --- 语言引导分支 ---
        if query_mask is not None:
            query = (query * query_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / \
                    query_mask.sum(dim=1, keepdim=True).clamp(min=1e-6).unsqueeze(-1)
        else:
            query = query.mean(dim=1, keepdim=True)  # [B, 1, h]
        attn = self.fc_feat(input) + self.fc_query(query)  # [B, T, h]
        attn = self.w(attn).transpose(1, 2)  # [B, n, T]
        
        # 应用掩码（同上）
        if input_mask is not None:
            attn = attn.masked_fill(~input_mask.bool(), -1e9)
        
        attn = F.softmax(attn, dim=-1)  # [B, n, T]
        feat_query = torch.einsum('...nt,...td->...nd', attn, input)  # [B, n, h]

        # --- 融合 ---
        feat = self.norm(
            self.video_gate(input.mean(1, keepdim=True)) * feat_selected + 
            self.query_gate(query) * feat_query
        )
        return feat

    def forward(self, input, query, mask=None, query_mask=None):
        return self.tokenlearner(input, query, mask, query_mask)

# 0. 基础的基于注意力机制的视频记忆压缩模块，支持融合外部权重
class CompressorWithExternalWeights(nn.Module):
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

    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1, weight_alpha=0.5):
        """
        初始化压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 压缩后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
        """
        super(CompressorWithExternalWeights, self).__init__()

        self.dimension = dimension
        self.compress_len = compress_len
        self.num_heads = num_heads

        self.weight_alpha = weight_alpha
        
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

    def before_attn(self, Fv, mask=None):
        """
        在注意力机制之前进行操作
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
        return attn_weights, query, v
    
    def ex_weight_fusion(self, mask, batch_size, vid_len, attn_weights, extern_weight):
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
            combined_weights = self.weight_alpha * attn_weights + (1 - self.weight_alpha) * extern_weight_expanded
            
            # 确保权重总和为1
            # 由于已经对两个输入分别进行了softmax/归一化，并且是线性组合，
            # 所以combined_weights的每一行的和应该接近1，但为了确保精确性，我们再次归一化
            combined_weights_sum = combined_weights.sum(dim=-1, keepdim=True)
            combined_weights = combined_weights / (combined_weights_sum + 1e-9)
            
            # 使用融合后的权重计算加权和
            # (batch_size, num_heads, compress_len, vid_len) @ (batch_size, num_heads, vid_len, head_dim)
            # -> (batch_size, num_heads, compress_len, head_dim)
            # weighted_v = torch.matmul(combined_weights, v)
        else:
            # 如果没有外部权重，就使用原始注意力权重
            # weighted_v = torch.matmul(attn_weights, v)
            combined_weights = attn_weights
        return combined_weights

    def after_attn(self, weighted_v, query, batch_size, device):
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

        # 创建压缩后的掩码，形状为 (batch_size, compress_len)，全为True
        compress_mask = torch.ones((batch_size, self.compress_len), dtype=torch.bool, device=device)
        
        return Fv_compress, compress_mask
        

    def forward(self, Fv, extern_weight=None, mask=None):
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

        attn_weights, query, v = self.before_attn(Fv, mask)
        
        combined_weights = self.ex_weight_fusion(mask, batch_size, vid_len, attn_weights, extern_weight)

        weighted_v = torch.matmul(combined_weights, v)

        Fv_compress, compress_mask = self.after_attn(weighted_v, query, batch_size, Fv.device)
        
        return Fv_compress, compress_mask

# 0+. 加入位置权重的视频压缩模块，保留时序信息
class CompressorWithPositionWeights(CompressorWithExternalWeights):
    """
    基于注意力机制的视频记忆压缩模块，融合外部权重和位置权重
    
    位置权重的目的是在压缩时保留视频特征之间的时序关系，
    方法是为每个压缩后的特征位置设定"中心点"，
    基于原始特征与这些中心点的距离计算position_weight
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        mask: 指示有效位置的掩码，形状为 (batch_size, vid_len)
        extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
        
    输出:
        Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
        compress_mask: 压缩后的有效掩码，形状为 (batch_size, compress_len)
    """

    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1, 
                 attn_weight_alpha=0.33, extern_weight_alpha=0.33, position_weight_alpha=0.34):
        """
        初始化压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 压缩后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
            attn_weight_alpha: 注意力权重的融合系数
            extern_weight_alpha: 外部权重的融合系数
            position_weight_alpha: 位置权重的融合系数
        """
        # 调用父类初始化，但不使用父类的weight_alpha
        super(CompressorWithPositionWeights, self).__init__(
            dimension, compress_len, num_heads, dropout, weight_alpha=None)
        
        # 存储三种权重的融合系数
        self.attn_weight_alpha = attn_weight_alpha
        self.extern_weight_alpha = extern_weight_alpha
        self.position_weight_alpha = position_weight_alpha
        
        # 检查权重和为1
        total_alpha = attn_weight_alpha + extern_weight_alpha + position_weight_alpha
        assert abs(total_alpha - 1.0) < 1e-5, f"权重和应为1，但得到了{total_alpha}"

    def _get_valid_lengths(self, mask):
        """
        计算每个样本的有效长度
        
        参数:
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            
        返回:
            valid_lengths: 每个样本的有效长度，形状为 (batch_size,)
        """
        if mask is None:
            # 如果没有提供掩码，假设所有位置都有效
            return torch.full((mask.size(0),), mask.size(1), device=mask.device)
        else:
            # 计算每个样本的有效位置数量
            return mask.sum(dim=1).int()
    
    def _compute_position_weights(self, mask, batch_size, vid_len):
        """
        计算基于位置的权重
        
        参数:
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            batch_size: 批次大小
            vid_len: 视频长度
            
        返回:
            position_weights: 位置权重，形状为 (batch_size, compress_len, vid_len)
            need_compress: 指示每个样本是否需要压缩的掩码，形状为 (batch_size,)
            original_indices: 对于不需要压缩的样本，原始特征的索引，形状为 (batch_size, compress_len)
        """
        device = mask.device if mask is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 计算每个样本的有效长度
        valid_lengths = self._get_valid_lengths(mask)
        
        # 确定哪些样本需要压缩（有效长度大于compress_len）
        need_compress = valid_lengths > self.compress_len
        
        # 初始化位置权重和原始索引
        position_weights = torch.zeros(batch_size, self.compress_len, vid_len, device=device)
        original_indices = torch.zeros(batch_size, self.compress_len, dtype=torch.long, device=device)
        
        # 对每个样本计算位置权重
        for i in range(batch_size):
            if need_compress[i]:
                # 样本需要压缩
                valid_len = valid_lengths[i].item()
                
                # 在有效长度范围内找到有效位置的索引
                if mask is not None:
                    valid_indices = torch.nonzero(mask[i], as_tuple=True)[0]
                else:
                    valid_indices = torch.arange(valid_len, device=device)
                
                # 计算中心点（在有效特征中均匀分布）
                # 例如，如果valid_len=10, compress_len=4，中心点将在索引 1.25, 3.75, 6.25, 8.75
                # 取整后为 1, 4, 6, 9
                centers = torch.linspace(0, valid_len - 1, self.compress_len, device=device)
                centers = centers.round().long()
                
                # 从valid_indices中获取实际的中心点索引
                center_indices = valid_indices[centers]
                
                # 为每个中心点计算到每个有效位置的距离权重
                for j, center_idx in enumerate(center_indices):
                    # 计算中心点到每个位置的归一化距离
                    # 距离被归一化到[-1, 1]区间，中心点处为0
                    if mask is not None:
                        distances = torch.arange(vid_len, device=device)
                        distances = (distances - center_idx) / valid_len
                    else:
                        distances = torch.arange(vid_len, device=device)
                        distances = (distances - center_idx) / valid_len
                    
                    # 转换为权重 (1 - |distance|)，距离越近权重越大
                    weights = 1.0 - torch.abs(distances)
                    
                    # 确保只有有效位置有权重
                    if mask is not None:
                        weights = weights * mask[i].float()
                    
                    # 存储权重
                    position_weights[i, j] = weights
            else:
                # 样本不需要压缩，直接选择原始特征的前compress_len个
                valid_len = valid_lengths[i].item()
                
                # 找到有效位置的索引
                if mask is not None:
                    valid_indices = torch.nonzero(mask[i], as_tuple=True)[0]
                else:
                    valid_indices = torch.arange(vid_len, device=device)
                
                # 选择前compress_len个有效位置（或者所有有效位置）
                num_indices = min(valid_len, self.compress_len)
                selected_indices = valid_indices[:num_indices]
                
                # 将选定的索引存储起来
                original_indices[i, :num_indices] = selected_indices
                
                # 对选中的位置设置权重为1，其他位置为0
                for j, idx in enumerate(selected_indices):
                    position_weights[i, j, idx] = 1.0
        
        # 确保每行的权重和为1（对于有效权重）
        # 避免除以0，添加一个小的epsilon
        row_sums = position_weights.sum(dim=-1, keepdim=True)
        position_weights = position_weights / (row_sums + 1e-9)
        
        return position_weights, need_compress, original_indices

    def _weight_fusion(self, attn_weights, extern_weight, position_weights, mask, batch_size, vid_len):
        """
        融合注意力权重、外部权重和位置权重
        
        参数:
            attn_weights: 注意力权重，形状为 (batch_size, num_heads, compress_len, vid_len)
            extern_weight: 外部权重，形状为 (batch_size, vid_len)
            position_weights: 位置权重，形状为 (batch_size, compress_len, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            batch_size: 批次大小
            vid_len: 视频长度
            
        返回:
            final_weights: 融合后的权重，形状为 (batch_size, num_heads, compress_len, vid_len)
        """
        # 1. 处理外部权重
        if extern_weight is not None:
            # 确保extern_weight的形状正确
            assert extern_weight.shape == (batch_size, vid_len), \
                f"外部权重形状应为 {(batch_size, vid_len)}, 但得到了 {extern_weight.shape}"
            
            # 扩展外部权重以匹配压缩长度
            # (batch_size, vid_len) -> (batch_size, compress_len, vid_len)
            extern_weight_expanded = extern_weight.unsqueeze(1).expand(-1, self.compress_len, -1)
            
            # 应用掩码
            if mask is not None:
                mask_expanded = mask.unsqueeze(1)  # (batch_size, 1, vid_len)
                extern_weight_expanded = extern_weight_expanded.masked_fill(~mask_expanded.bool(), 0)
                
                # 重新归一化外部权重
                row_sums = extern_weight_expanded.sum(dim=-1, keepdim=True)
                extern_weight_expanded = extern_weight_expanded / (row_sums + 1e-9)
            
            # 再次扩展以匹配多头
            # (batch_size, compress_len, vid_len) -> (batch_size, 1, compress_len, vid_len)
            extern_weight_expanded = extern_weight_expanded.unsqueeze(1)
        else:
            # 如果没有外部权重，设为0，不参与融合
            extern_weight_expanded = torch.zeros_like(attn_weights)
        
        # 2. 处理位置权重
        # (batch_size, compress_len, vid_len) -> (batch_size, 1, compress_len, vid_len)
        position_weights_expanded = position_weights.unsqueeze(1)
        
        # 3. 融合三种权重
        final_weights = (
            self.attn_weight_alpha * attn_weights + 
            self.extern_weight_alpha * extern_weight_expanded + 
            self.position_weight_alpha * position_weights_expanded
        )
        
        # 确保权重总和为1
        final_weights_sum = final_weights.sum(dim=-1, keepdim=True)
        final_weights = final_weights / (final_weights_sum + 1e-9)
        
        return final_weights

    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            compress_mask: 压缩后的有效掩码，形状为 (batch_size, compress_len)
        """
        batch_size, vid_len, _ = Fv.size()
        device = Fv.device
        
        # 1. 计算位置权重
        position_weights, need_compress, original_indices = self._compute_position_weights(
            mask, batch_size, vid_len)
        
        # 2. 对于需要压缩的样本，使用注意力机制
        if need_compress.any():
            # 获取注意力权重和中间结果
            attn_weights, query, v = self.before_attn(Fv, mask)
            
            # 融合三种权重
            combined_weights = self._weight_fusion(
                attn_weights, extern_weight, position_weights, mask, batch_size, vid_len)
            
            # 使用融合后的权重计算加权和
            weighted_v = torch.matmul(combined_weights, v)
            
            # 后续处理得到压缩后的特征
            Fv_compress, compress_mask = self.after_attn(weighted_v, query, batch_size, device)
        # 特殊情况，全部样本的有效长度都小于compress_len，这时就没有压缩的必要了
        else:
            # 所有样本都不需要压缩，只复制前压缩长度个特征
            Fv_compress = torch.zeros(batch_size, self.compress_len, Fv.size(2), device=device)
            compress_mask = torch.zeros(batch_size, self.compress_len, dtype=torch.bool, device=device)
            
            # 针对每个不需要压缩的样本
            for i in range(batch_size):
                valid_len = min(self._get_valid_lengths(mask)[i].item(), self.compress_len)
                
                # 从原始特征中复制有效部分
                indices = original_indices[i, :valid_len]
                Fv_compress[i, :valid_len] = Fv[i, indices]
                
                # 设置有效掩码
                compress_mask[i, :valid_len] = True
        
        return Fv_compress, compress_mask

# 1. 计算attention前，先把ex_weight乘到KV上
class CompressorWithWeightedKV(CompressorWithExternalWeights):
    """
    在计算注意力之前，将外部权重乘到Key和Value上的视频压缩模块。
    
    方案1: 在计算attention前，先把extern_weight乘到KV上，然后再进行self-attention计算
    """
    
    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            weight_alpha: 注意力权重的权重，1-weight_alpha为外部权重的权重
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            combined_weights: 注意力权重，形状为 (batch_size, compress_len, vid_len)
        """
        batch_size, vid_len, _ = Fv.size()
        device = Fv.device
        
        # 如果提供了外部权重，直接将其应用到Fv上
        if extern_weight is not None:
            # 扩展形状以便乘法操作: (batch_size, vid_len) -> (batch_size, vid_len, 1)
            weighted_mask = extern_weight.unsqueeze(-1)
            
            # 将权重直接应用到Fv
            weighted_Fv = Fv * weighted_mask
        else:
            # 如果没有外部权重，使用原始的Fv
            weighted_Fv = Fv
        
        # 使用加权后的特征进行注意力计算
        attn_weights, query, v = self.before_attn(weighted_Fv, mask)
        
        # 计算加权和
        weighted_v = torch.matmul(attn_weights, v)
        
        # 后续处理
        Fv_compress, compress_mask = self.after_attn(weighted_v, query, batch_size, device)
        
        return Fv_compress, compress_mask

# 2. attention之后在输出前乘上ex_weight
class CompressorWithPostWeighting(CompressorWithExternalWeights):
    """
    在注意力计算之后，输出前将外部权重应用于结果的视频压缩模块。
    
    方案2: attention之后在输出前乘上ex_weight
    """
    
    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            weight_alpha: 注意力权重的权重，1-weight_alpha为外部权重的权重
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            attn_weights: 注意力权重，形状为 (batch_size, compress_len, vid_len)
        """
        batch_size, vid_len, _ = Fv.size()
        device = Fv.device
        
        # 常规的注意力计算
        attn_weights, query, v = self.before_attn(Fv, mask)
        
        # 如果提供了外部权重，修改注意力权重
        if extern_weight is not None:
            # 扩展外部权重以匹配多头注意力的形状
            # (batch_size, vid_len) -> (batch_size, 1, 1, vid_len)
            ext_weight_expanded = extern_weight.unsqueeze(1).unsqueeze(2)
            
            # 将扩展后的权重应用到注意力权重
            # (batch_size, num_heads, compress_len, vid_len) * (batch_size, 1, 1, vid_len)
            weighted_attn = attn_weights * ext_weight_expanded
            
            # 重新归一化
            sum_weights = weighted_attn.sum(dim=-1, keepdim=True)
            weighted_attn = weighted_attn / (sum_weights + 1e-9)
            
            # 使用加权后的注意力计算输出
            weighted_v = torch.matmul(weighted_attn, v)
        else:
            # 如果没有外部权重，使用原始的注意力权重
            weighted_v = torch.matmul(attn_weights, v)
        
        # 后续处理
        Fv_compress, compress_mask = self.after_attn(weighted_v, query, batch_size, device)
        
        return Fv_compress, compress_mask

# 3. 平滑机制，在fi直接乘权重的基础上，再加上一个favg*（1-si）
class CompressorWithSmoothMechanism(CompressorWithExternalWeights):
    """
    具有平滑机制的视频压缩模块：fi = fi * si + favg * (1-si)
    
    方案3: 平滑机制，在fi直接乘权重的基础上，再加上一个favg*（1-si）
    即fi = fi * si + favg*（1-si）
    si即每个视频特征上的外部权重，favg是一个视频在有效位的特征平均值
    """
    
    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重（si），形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            weight_alpha: 注意力权重的权重，1-weight_alpha为外部权重的权重
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            combined_weights: 注意力权重，形状为 (batch_size, compress_len, vid_len)
        """
        batch_size, vid_len, dimension = Fv.size()
        device = Fv.device
        
        # 计算特征平均值 (favg)
        if mask is not None:
            # 创建掩码来排除无效位置: (batch_size, vid_len, 1)
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, dimension)
            
            # 计算有效位置的平均值
            # 首先，将无效位置置为0
            masked_Fv = Fv * mask_expanded.float()
            
            # 然后，计算每个batch样本中有效特征的平均值
            # 对于每个batch样本，计算有效位置的数量
            valid_counts = mask.float().sum(dim=1, keepdim=True)  # (batch_size, 1)
            
            # 计算有效位置的特征总和
            sum_features = masked_Fv.sum(dim=1)  # (batch_size, dimension)
            
            # 计算平均值，避免除以0
            favg = sum_features / (valid_counts + 1e-9)  # (batch_size, dimension)
        else:
            # 如果没有掩码，计算所有位置的平均值
            favg = Fv.mean(dim=1)  # (batch_size, dimension)
        
        # 如果提供了外部权重 (si)
        if extern_weight is not None:
            # 确保extern_weight形状正确
            assert extern_weight.shape == (batch_size, vid_len), \
                f"外部权重形状应为 {(batch_size, vid_len)}, 但得到了 {extern_weight.shape}"
            
            # 应用掩码（如果提供）
            if mask is not None:
                si = extern_weight.clone()
                si = si.masked_fill(~mask.bool(), 0)
            else:
                si = extern_weight
            
            # 实现平滑机制: fi = fi * si + favg * (1-si)
            # 扩展si和favg以便于乘法操作
            si_expanded = si.unsqueeze(-1)  # (batch_size, vid_len, 1)
            favg_expanded = favg.unsqueeze(1)  # (batch_size, 1, dimension)
            
            # 计算平滑后的特征
            smoothed_Fv = Fv * si_expanded + favg_expanded * (1 - si_expanded)
        else:
            # 如果没有外部权重，使用原始的Fv
            smoothed_Fv = Fv
        
        # 使用平滑后的特征进行注意力计算
        attn_weights, query, v = self.before_attn(smoothed_Fv, mask)
        
        # 计算加权和
        weighted_v = torch.matmul(attn_weights, v)
        
        # 后续处理
        Fv_compress, compress_mask = self.after_attn(weighted_v, query, batch_size, device)
        
        return Fv_compress, compress_mask

# 4. 残差，将压缩结果Fs和最近compress_len个long memory相加
class ResidualCompressor(CompressorWithExternalWeights):
    def __init__(self, compress_len, dimension, num_heads=8, dropout=0.1, weight_alpha=0.5, residual_weight=0.5):
        super().__init__(compress_len=compress_len, dimension=dimension, weight_alpha=weight_alpha)
        self.residual_weight = residual_weight
        
    def forward(self, Fv, extern_weight=None, mask=None):
        batch_size, vid_len, _ = Fv.size()
        device = Fv.device
        
        # 压缩
        Fv_compress, compress_mask = super().forward(Fv, extern_weight, mask)

        # # 根据mask取后compress_len个有效的long memory
        # if mask is not None:
        #     long_memory = Fv[mask.bool()]

        #     print(long_memory.shape)
        #     print(compress_mask.shape)
        #     input("Press Enter to continue...")

        #     Fo = long_memory[:, -self.compress_len:, :]
        # else:
        #     Fo = Fv[:, -self.compress_len:, :]

        # 提取每个 batch 的有效特征
        bool_mask = mask.bool()
        Fo = []
        for batch in range(Fv.shape[0]):
            Fo.append(Fv[batch][bool_mask[batch]])
            Fo[batch] = Fo[batch][-self.compress_len:]

        # 将Fo中长度第一维长度小于compress_len的填充到compress_len
        padded_Fo = torch.zeros(batch_size, self.compress_len, _)
        for i in range(batch_size):
            if Fo[i].shape[0] < self.compress_len:
                padded_Fo[i] = F.pad(Fo[i], (0, 0, 0, self.compress_len - Fo[i].shape[0]), "constant", 0)
            else:
                padded_Fo[i] = Fo[i]
        # 移动到相同的设备
        padded_Fo = padded_Fo.to(device)
        # 残差
        # Fv_residual = Fv_compress + self.residual_weight * padded_Fo
        Fv_residual = padded_Fo + self.residual_weight * Fv_compress

        # 对于每个batch的结果，如果Fo长度小于等于compress_len，则直接返回未压缩的Fv，
        # 并将mask根据Fo的长度重新设置：前len(Fo[i])个为有效，其余为无效
        for i in range(batch_size):
            if Fo[i].shape[0] <= self.compress_len:
                Fv_residual[i] = Fv[i][:self.compress_len]
                compress_mask[i] = torch.zeros(self.compress_len, dtype=torch.bool, device=device)
                compress_mask[i][:len(Fo[i])] = True

        return Fv_residual, compress_mask

# 5. 残差+attn
class CrossAttentionResidualCompressor(CompressorWithExternalWeights):
    """
    使用交叉注意力机制的残差压缩模块
    
    在压缩结果Fs的基础上，取最近的compress_len个记忆帧Fo，
    通过 Fo + δ * attn(Fo, Fs) 的方式进行融合
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
        mask: 有效位置的掩码，形状为 (batch_size, vid_len)
        
    输出:
        Fv_residual: 融合后的压缩视频特征，形状为 (batch_size, compress_len, dimension)
        compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
    """
    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1, weight_alpha=0.5, attn_weight=0.5):
        """
        初始化交叉注意力残差压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 压缩后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
            weight_alpha: 注意力权重和外部权重的融合比例
            attn_weight: 注意力结果的权重系数δ
        """
        super().__init__(dimension, compress_len, num_heads, dropout, weight_alpha)
        self.attn_weight = attn_weight  # δ参数，控制注意力结果的权重
        
        # 交叉注意力层的投影矩阵
        self.cross_q_proj = nn.Linear(dimension, dimension)
        self.cross_k_proj = nn.Linear(dimension, dimension)
        self.cross_v_proj = nn.Linear(dimension, dimension)
        
        # 输出投影
        self.cross_out_proj = nn.Linear(dimension, dimension)
        
        # Dropout
        self.cross_dropout = nn.Dropout(dropout)
        
    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            
        返回:
            Fv_residual: 融合后的压缩视频特征，形状为 (batch_size, compress_len, dimension)
            compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
        """
        batch_size, vid_len, dimension = Fv.size()
        device = Fv.device
        
        # 1. 首先获取压缩的视频特征 Fs (Fv_compress)
        Fs, compress_mask = super().forward(Fv, extern_weight, mask)
        
        # 2. 提取每个batch的最近compress_len个有效特征作为Fo
        bool_mask = mask.bool() if mask is not None else torch.ones(batch_size, vid_len, dtype=torch.bool, device=device)
        Fo_list = []
        for batch in range(batch_size):
            # 获取当前批次的有效特征
            valid_features = Fv[batch][bool_mask[batch]]
            # 取最后compress_len个
            Fo_list.append(valid_features[-self.compress_len:])
        
        # 3. 创建Fo的填充张量和Fo的掩码
        padded_Fo = torch.zeros(batch_size, self.compress_len, dimension, device=device)
        fo_mask = torch.zeros(batch_size, self.compress_len, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            fo_len = Fo_list[i].shape[0]
            # 如果Fo长度小于compress_len，进行填充
            if fo_len < self.compress_len:
                padded_Fo[i, :fo_len] = Fo_list[i]
                fo_mask[i, :fo_len] = True
            else:
                padded_Fo[i] = Fo_list[i]
                fo_mask[i] = torch.ones(self.compress_len, dtype=torch.bool, device=device)
        
        # 4. 对于batch中Fo长度小于等于compress_len的样本，直接使用原始特征
        # 创建结果变量，初始为Fs
        Fv_residual = Fs.clone()
        special_cases = []
        
        for i in range(batch_size):
            if Fo_list[i].shape[0] <= self.compress_len:
                # 如果Fo长度小于等于compress_len，直接使用原始特征
                Fv_residual[i, :Fo_list[i].shape[0]] = Fv[i][bool_mask[i]][-Fo_list[i].shape[0]:]
                compress_mask[i] = torch.zeros(self.compress_len, dtype=torch.bool, device=device)
                compress_mask[i][:Fo_list[i].shape[0]] = True
                special_cases.append(i)
        
        # 如果所有batch都是特殊情况，直接返回结果
        if len(special_cases) == batch_size:
            return Fv_residual, compress_mask
        
        # 创建一个掩码，标记哪些batch需要进行交叉注意力计算
        batch_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        for idx in special_cases:
            batch_mask[idx] = False
        
        # 5. 对其余样本执行交叉注意力: Fo作为query，Fs作为key和value
        if batch_mask.any():
            # 筛选需要进行注意力计算的batch
            filtered_Fo = padded_Fo[batch_mask]
            filtered_Fs = Fs[batch_mask]
            filtered_fo_mask = fo_mask[batch_mask]
            
            # 交叉注意力计算
            # 5.1 投影query, key, value
            q = self.cross_q_proj(filtered_Fo)  # (filtered_batch_size, compress_len, dimension)
            k = self.cross_k_proj(filtered_Fs)  # (filtered_batch_size, compress_len, dimension)
            v = self.cross_v_proj(filtered_Fs)  # (filtered_batch_size, compress_len, dimension)
            
            # 5.2 分割多头
            filtered_batch_size = filtered_Fo.size(0)
            head_dim = dimension // self.num_heads
            
            # 重塑为多头格式
            q = q.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            k = k.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            
            # 5.3 计算注意力分数
            # (filtered_batch_size, num_heads, compress_len, head_dim) @ (filtered_batch_size, num_heads, head_dim, compress_len)
            # -> (filtered_batch_size, num_heads, compress_len, compress_len)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            
            # 应用Fo的掩码（确保填充部分不参与注意力计算）
            if filtered_fo_mask is not None:
                # 扩展掩码以适应多头注意力的形状
                # (filtered_batch_size, compress_len) -> (filtered_batch_size, 1, compress_len, 1)
                q_mask = filtered_fo_mask.unsqueeze(1).unsqueeze(-1)
                # 在无效查询位置应用大的负值
                scores = scores.masked_fill(~q_mask.bool(), -1e9)
            
            # 应用softmax得到注意力权重
            attn_weights = F.softmax(scores, dim=-1)  # (filtered_batch_size, num_heads, compress_len, compress_len)
            attn_weights = self.cross_dropout(attn_weights)
            
            # 5.4 计算加权和
            # (filtered_batch_size, num_heads, compress_len, compress_len) @ (filtered_batch_size, num_heads, compress_len, head_dim)
            # -> (filtered_batch_size, num_heads, compress_len, head_dim)
            weighted_v = torch.matmul(attn_weights, v)
            
            # 5.5 合并多头结果
            weighted_v = weighted_v.transpose(1, 2).contiguous()  # (filtered_batch_size, compress_len, num_heads, head_dim)
            weighted_v = weighted_v.view(filtered_batch_size, self.compress_len, dimension)  # (filtered_batch_size, compress_len, dimension)
            
            # 5.6 应用输出投影
            attn_output = self.cross_out_proj(weighted_v)  # (filtered_batch_size, compress_len, dimension)
            
            # 6. 计算残差连接: Fo + δ * attn(Fo, Fs)
            # 对于Fo的无效(padding)部分，我们不应用残差
            filtered_residual = filtered_Fo + self.attn_weight * attn_output
            
            # 7. 将结果放回到原始batch中
            Fv_residual[batch_mask] = filtered_residual
        
        return Fv_residual, compress_mask

class CrossAttentionResidualCompressorWithQuery(TextGuidedCompressor):
    """
    使用交叉注意力机制的残差压缩模块
    
    在压缩结果Fs的基础上，取最近的compress_len个记忆帧Fo，
    通过 Fo + δ * attn(Fo, Fs) 的方式进行融合
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
        mask: 有效位置的掩码，形状为 (batch_size, vid_len)
        
    输出:
        Fv_residual: 融合后的压缩视频特征，形状为 (batch_size, compress_len, dimension)
        compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
    """
    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1, weight_alpha=0.5, attn_weight=0.5):
        """
        初始化交叉注意力残差压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 压缩后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
            weight_alpha: 注意力权重和外部权重的融合比例
            attn_weight: 注意力结果的权重系数δ
        """
        super().__init__(t_dim=dimension,v_dim = dimension,hidden_dim=dimension, compress_len=compress_len, num_heads=num_heads, dropout=dropout, weight_alpha=weight_alpha)
        self.attn_weight = attn_weight  # δ参数，控制注意力结果的权重
        
        # 交叉注意力层的投影矩阵
        self.cross_q_proj = nn.Linear(dimension, dimension)
        self.cross_k_proj = nn.Linear(dimension, dimension)
        self.cross_v_proj = nn.Linear(dimension, dimension)
        
        # 输出投影
        self.cross_out_proj = nn.Linear(dimension, dimension)
        
        # Dropout
        self.cross_dropout = nn.Dropout(dropout)
        
    def forward(self, Ft, text_mask, Fv, extern_weight=None, vid_mask=None):
        """
        前向传播
        
        参数:
            Ft: 文本特征，形状为 (batch_size, num_token, t_dim)
            text_mask: 文本掩码，形状为 (batch_size, num_token)
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            
        返回:
            Fv_residual: 融合后的压缩视频特征，形状为 (batch_size, compress_len, dimension)
            compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
        """
        batch_size, vid_len, dimension = Fv.size()
        device = Fv.device
        
        # 1. 首先获取压缩的视频特征 Fs (Fv_compress)
        Fs, compress_mask = super().forward(Ft, Fv, vid_mask = vid_mask, text_mask = text_mask, extern_weight = extern_weight)
        
        # 2. 提取每个batch的最近compress_len个有效特征作为Fo
        bool_mask = vid_mask.bool() if vid_mask is not None else torch.ones(batch_size, vid_len, dtype=torch.bool, device=device)
        Fo_list = []
        for batch in range(batch_size):
            # 获取当前批次的有效特征
            valid_features = Fv[batch][bool_mask[batch]]
            # 取最后compress_len个
            Fo_list.append(valid_features[-self.compress_len:])
        
        # 3. 创建Fo的填充张量和Fo的掩码
        padded_Fo = torch.zeros(batch_size, self.compress_len, dimension, device=device)
        fo_mask = torch.zeros(batch_size, self.compress_len, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            fo_len = Fo_list[i].shape[0]
            # 如果Fo长度小于compress_len，进行填充
            if fo_len < self.compress_len:
                padded_Fo[i, :fo_len] = Fo_list[i]
                fo_mask[i, :fo_len] = True
            else:
                padded_Fo[i] = Fo_list[i]
                fo_mask[i] = torch.ones(self.compress_len, dtype=torch.bool, device=device)
        
        # 4. 对于batch中Fo长度小于等于compress_len的样本，直接使用原始特征
        # 创建结果变量，初始为Fs
        Fv_residual = Fs.clone()
        special_cases = []
        
        for i in range(batch_size):
            if Fo_list[i].shape[0] <= self.compress_len:
                # 如果Fo长度小于等于compress_len，直接使用原始特征
                Fv_residual[i, :Fo_list[i].shape[0]] = Fv[i][bool_mask[i]][-Fo_list[i].shape[0]:]
                compress_mask[i] = torch.zeros(self.compress_len, dtype=torch.bool, device=device)
                compress_mask[i][:Fo_list[i].shape[0]] = True
                special_cases.append(i)
        
        # 如果所有batch都是特殊情况，直接返回结果
        if len(special_cases) == batch_size:
            return Fv_residual, compress_mask
        
        # 创建一个掩码，标记哪些batch需要进行交叉注意力计算
        batch_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        for idx in special_cases:
            batch_mask[idx] = False
        
        # 5. 对其余样本执行交叉注意力: Fo作为query，Fs作为key和value
        if batch_mask.any():
            # 筛选需要进行注意力计算的batch
            filtered_Fo = padded_Fo[batch_mask]
            filtered_Fs = Fs[batch_mask]
            filtered_fo_mask = fo_mask[batch_mask]
            
            # 交叉注意力计算
            # 5.1 投影query, key, value
            q = self.cross_q_proj(filtered_Fo)  # (filtered_batch_size, compress_len, dimension)
            k = self.cross_k_proj(filtered_Fs)  # (filtered_batch_size, compress_len, dimension)
            v = self.cross_v_proj(filtered_Fs)  # (filtered_batch_size, compress_len, dimension)
            
            # 5.2 分割多头
            filtered_batch_size = filtered_Fo.size(0)
            head_dim = dimension // self.num_heads
            
            # 重塑为多头格式
            q = q.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            k = k.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            
            # 5.3 计算注意力分数
            # (filtered_batch_size, num_heads, compress_len, head_dim) @ (filtered_batch_size, num_heads, head_dim, compress_len)
            # -> (filtered_batch_size, num_heads, compress_len, compress_len)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            
            # 应用Fo的掩码（确保填充部分不参与注意力计算）
            if filtered_fo_mask is not None:
                # 扩展掩码以适应多头注意力的形状
                # (filtered_batch_size, compress_len) -> (filtered_batch_size, 1, compress_len, 1)
                q_mask = filtered_fo_mask.unsqueeze(1).unsqueeze(-1)
                # 在无效查询位置应用大的负值
                scores = scores.masked_fill(~q_mask.bool(), -1e9)
            
            # 应用softmax得到注意力权重
            attn_weights = F.softmax(scores, dim=-1)  # (filtered_batch_size, num_heads, compress_len, compress_len)
            attn_weights = self.cross_dropout(attn_weights)
            
            # 5.4 计算加权和
            # (filtered_batch_size, num_heads, compress_len, compress_len) @ (filtered_batch_size, num_heads, compress_len, head_dim)
            # -> (filtered_batch_size, num_heads, compress_len, head_dim)
            weighted_v = torch.matmul(attn_weights, v)
            
            # 5.5 合并多头结果
            weighted_v = weighted_v.transpose(1, 2).contiguous()  # (filtered_batch_size, compress_len, num_heads, head_dim)
            weighted_v = weighted_v.view(filtered_batch_size, self.compress_len, dimension)  # (filtered_batch_size, compress_len, dimension)
            
            # 5.6 应用输出投影
            attn_output = self.cross_out_proj(weighted_v)  # (filtered_batch_size, compress_len, dimension)
            
            # 6. 计算残差连接: Fo + δ * attn(Fo, Fs)
            # 对于Fo的无效(padding)部分，我们不应用残差
            filtered_residual = filtered_Fo + self.attn_weight * attn_output
            
            # 7. 将结果放回到原始batch中
            Fv_residual[batch_mask] = filtered_residual
        
        return Fv_residual, compress_mask