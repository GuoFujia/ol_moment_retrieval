import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class InterMemory(nn.Module):
    """
    交互式记忆模块：用于管理历史信息并预测未来特征
    基于MGSL-Net的记忆机制，包含记忆槽的维护和基于记忆的特征预测
    """
    def __init__(self, 
                 d_model=256,           # 模型特征维度
                 future_length=32,      # 未来特征长度
                 memory_slots=64,       # 记忆槽数量
                 similarity_threshold=0.1, # 相似度阈值，控制记忆槽更新
                 num_heads=8):          # 注意力头数量
        super(InterMemory, self).__init__()
        
        self.d_model = d_model
        self.future_length = future_length
        self.memory_slots = memory_slots
        self.similarity_threshold = similarity_threshold
        self.num_heads = num_heads
        self.updateCnt = 0
        
        # 初始化记忆槽，形状为 [memory_slots, d_model]
        self.memory = nn.Parameter(torch.randn(memory_slots, d_model))
        # L2归一化
        # self.memory.data = F.normalize(self.memory.data, p=2, dim=-1)
        
        # 特征映射层，用于生成记忆更新的写入键和写入值
        self.write_key_proj = nn.Linear(d_model, d_model)
        self.write_val_proj = nn.Linear(d_model, d_model)
        self.erase_val_proj = nn.Linear(d_model, d_model)
        
        # 未来特征初始化嵌入，指定设备
        self.future_embedding = nn.Parameter(torch.randn(future_length, d_model))
        
        # 用于从记忆中读取信息的注意力机制
        self.memory_read_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # 用于对记忆读取结果进行自注意力处理的模块
        self.memory_self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # 跟踪哪些视频帧已经被更新到记忆中
        self.updated_frames = defaultdict(set)

        # 记忆槽使用频率跟踪
        self.register_buffer('slot_usage_counter', torch.zeros(memory_slots))
        
    def reset(self):
        self.memory = nn.Parameter(torch.randn(self.memory_slots, self.d_model))
        self.updateCnt = 0
        self.slot_usage_counter.zero_()
        self.updated_frames = defaultdict(set)
        # L2归一化
        # self.memory.data = F.normalize(self.memory.data, p=2, dim=-1)

    def getUpdateCnt(self):
        return self.updateCnt
    
    def setInterMemory(self, memory, updateCnt):
        self.memory = memory
        self.updateCnt = updateCnt

    # 后续可以加一个set或者表，记录下哪些帧已经更新过了（qid, idx）作为键，只更新没有更新过的帧
    def update_memory(self, features, short_start, vid, mask=None):
        """
        基于输入特征更新记忆槽
        实现基于相似度的选择性写入和擦除机制
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            short_start: 每个批次样本在原视频中的起始帧索引，形状为 [batch_size]
            vid: 每个批次样本所属的视频ID，形状为 [batch_size]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
        """

        self.updateCnt += 1

        batch_size, seq_length, _ = features.shape

        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
        else:
            # 确保mask是布尔类型
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 逐帧/逐特征处理输入
        for b in range(batch_size):
            # 计算当前批次的有效帧数量
            valid_feat = mask[b].sum().item()
            
            if valid_feat == 0:
                continue
            
            # 获取当前批次有效帧的索引（按照从左到右的顺序）
            valid_indices = torch.where(mask[b])[0]

            # for t in range(seq_length):
            for i in range(valid_feat-1, -1, -1):
                t = valid_indices[i]  # 当前处理的帧在seq_length中的位置
                
                # 计算原视频中的帧索引
                # 从右往左第i+1个有效帧对应视频中的(short_start-i-1)帧
                frame_idx = short_start[b].item() - (valid_feat - 1 - i)
                video_id = vid[b]
                
                # 检查该帧是否已经被更新到记忆中
                frame_key = (video_id, frame_idx)
                if frame_key in self.updated_frames:
                    # 如果已经更新过，则跳过
                    continue
                
                # 将该帧标记为已更新
                self.updated_frames[video_id].add(frame_idx)

                # 获取当前帧的特征
                feature = features[b, t].unsqueeze(0)  # [1, d_model]
                
                # 生成写入键
                write_key = self.write_key_proj(feature)  # [1, d_model]
                
                # 计算写入键与所有记忆槽的余弦相似度
                # 对记忆槽和写入键进行L2归一化
                memory_norm = F.normalize(self.memory, p=2, dim=1)
                key_norm = F.normalize(write_key, p=2, dim=1)
                
                # 计算余弦相似度 [memory_slots]
                similarity = torch.matmul(memory_norm, key_norm.t()).squeeze()
                
                # 使用softmax获取写入权重
                write_weights = F.softmax(similarity, dim=0)
                
                # 生成写入值和擦除值
                write_val = self.write_val_proj(feature)  # [1, d_model]
                erase_val = torch.sigmoid(self.erase_val_proj(feature))  # [1, d_model]，值范围(0,1)
                
                # 动态调整相似度阈值
                sim_thd = self.slot_usage_counter*self.similarity_threshold
                sim_thd = sim_thd.clamp(max=self.similarity_threshold*10)
                update_mask = (similarity > sim_thd)

                # # 固定阈值
                # update_mask = (similarity > self.similarity_threshold)
                
                if update_mask.any():
                    # 对需要更新的记忆槽应用擦除和写入操作
                    for i in range(self.memory_slots):
                        if update_mask[i]:
                            self.slot_usage_counter[i] += 1
                            # 基于相似度的擦除与写入（类似那篇文章中的更新公式）
                            # m_i' = m_i * (1 - w_i * e_t) + w_i * u_t
                            # 但是说实话我不太理解这里et是什么作用，一个从输入特征线性映射+sigmoid得到的值，为什么能指导旧记忆的保留程度
                            erase_term = 1.0 - write_weights[i] * erase_val
                            # 不用erase_val，更新退化为特征和旧记忆的加权融合
                            # erase_term = 1.0 - write_weights[i]
                            write_term = write_weights[i] * write_val
                            
                            # 更新对应的记忆槽
                            self.memory.data[i] = self.memory.data[i] * erase_term + write_term
        # 对更新后的记忆进行L2归一化
        # self.memory.data = F.normalize(self.memory.data, p=2, dim=-1)
        if torch.isnan(self.memory).any():
            print("after update, memory contains NaN")
            input("Press Enter to continue...")

    def read_memory(self, query_features, mask=None):
        """
        从记忆中读取与查询特征相关的信息
        
        Args:
            query_features: 拼接特征，形状为 [batch_size, concat_seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, concat_seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
        Returns:
            memory_output: 从记忆中检索的输出，形状为 [batch_size, concat_seq_length, d_model]
        """
        batch_size = query_features.shape[0]
        
        # 扩展记忆以匹配批次大小
        # 形状: [batch_size, memory_slots, d_model]
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)

        if mask is None:
            attn_mask = None
        else:
            if mask.dtype == torch.bool:
                # 如果是布尔型，True=有效，False=填充
                mask_float = mask.float()  # True->1.0, False->0.0
            else:
                # 如果是数值型，假设 1=有效，0=填充
                mask_float = mask.clamp(0, 1)  # 确保是 0/1
            # 转换mask为float类型，并将填充位(False)变为-inf，有效位(True)变为0
            # [batch_size, concat_seq_length] -> [batch_size, concat_seq_length, 1]
            attn_mask = mask_float.unsqueeze(-1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
            # 加性，有效位加0不影响，填充位加-inf，softmax后会趋近于0
            # 将attn_mask扩展到[batch_size, concat_seq_length, memory_slots]
            attn_mask = attn_mask.expand(-1, -1, self.memory_slots)
            # 调整为多头注意力需要的形状 [B*num_heads, L, M]
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)  # 复制到每个注意力头


        
        # 使用注意力机制从记忆中读取信息
        # Q: 查询特征 [batch_size, concat_seq_length, d_model]
        # K, V: 记忆 [batch_size, memory_slots, d_model]
        memory_output, _ = self.memory_read_attention(
            query=query_features,
            key=memory_expanded,
            value=memory_expanded,
            attn_mask=attn_mask
        )

        if torch.isnan(memory_output).any():
            print("in interMemory, read_memory, memory_output contains NaN")
            torch.set_printoptions(profile="full")
            for i in range(batch_size):
                for j in range(self.memory_slots):
                    if torch.isnan(memory_output[i, j]).any():
                        print(f"Batch {i}, Memory Slot {j} contains NaN")
                        print(memory_output[i, j])
                        input("Press Enter to continue...")
            input("all nan batches is found, Press Enter to continue...")
        
        return memory_output
    
    def read_memory_avg(self, query_features, mask=None):
        """
        从记忆中读取与查询特征相关的信息
        新方法：将每个槽独立处理，然后对结果求平均
        
        Args:
            query_features: 拼接特征，形状为 [batch_size, concat_seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, concat_seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
        Returns:
            memory_output: 从记忆中检索的输出，形状为 [batch_size, concat_seq_length, d_model]
        """
        batch_size, seq_length, _ = query_features.shape
        device = query_features.device
        
        # 创建一个存储所有slot注意力结果的张量
        all_slot_outputs = torch.zeros(batch_size, seq_length, self.d_model, device=device)
        
        # 处理掩码
        if mask is None:
            attn_mask = None
        else:
            if mask.dtype == torch.bool:
                mask_float = mask.float()  # True->1.0, False->0.0
            else:
                mask_float = mask.clamp(0, 1)  # 确保是 0/1
            
            # 转换为注意力掩码格式
            attn_mask = mask_float.unsqueeze(-1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        
        # 遍历每个记忆槽
        for i in range(self.memory_slots):
            # 获取当前记忆槽
            # 形状：[d_model] -> [1, 1, d_model] -> [batch_size, 1, d_model]
            current_slot = self.memory[i].unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            
            # 如果有掩码，调整为当前记忆槽的格式
            if attn_mask is not None:
                # 对于单个槽，只需要 [batch_size, seq_length, 1]
                current_mask = attn_mask[:, :, 0:1]
                # 调整为多头注意力需要的形状
                current_mask = current_mask.repeat_interleave(self.num_heads, dim=0)
            else:
                current_mask = None
            
            # 对当前槽执行注意力
            slot_output, _ = self.memory_read_attention(
                query=query_features,
                key=current_slot,
                value=current_slot,
                attn_mask=current_mask
            )
            
            # 将结果添加到输出中
            all_slot_outputs += slot_output
        
        # 计算平均值
        memory_output = all_slot_outputs / self.memory_slots
        
        if torch.isnan(memory_output).any():
            print("in interMemory, read_memory, memory_output contains NaN")
            torch.set_printoptions(profile="full")
            for i in range(batch_size):
                if torch.isnan(memory_output[i]).any():
                    print(f"Batch {i} contains NaN")
                    print(memory_output[i])
                    input("Press Enter to continue...")
            input("all nan batches is found, Press Enter to continue...")
        
        return memory_output
    
    def predict_future(self, history_features, current_features, mask=None):
        """
        基于历史特征、当前特征和记忆预测未来特征
        
        Args:
            history_features: 历史特征，形状为 [batch_size, history_length, d_model]
            current_features: 当前特征，形状为 [batch_size, current_length, d_model]
            mask: 历史特征的有效性掩码，形状为 [batch_size, history_length]
            
        Returns:
            future_features: 预测的未来特征，形状为 [batch_size, future_length, d_model]
        """
        batch_size = history_features.shape[0]
        
        # 扩展未来嵌入以匹配批次大小
        # 形状: [batch_size, future_length, d_model]
        future_embed = self.future_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接历史特征、当前特征和未来嵌入
        # 形状: [batch_size, (history_length + current_length + future_length), d_model]
        concatenated_features = torch.cat([history_features, current_features, future_embed], dim=1)

        # 拼接特征的mask,形状为 [batch_size, (history_length + current_length + future_length)]
        if mask is not None:
            current_mask = torch.ones(batch_size, current_features.shape[1], 
                                device=history_features.device, dtype=mask.dtype)
            future_mask = torch.ones(batch_size, future_embed.shape[1], 
                               device=history_features.device, dtype=mask.dtype)
            mask = torch.cat([mask, current_mask, future_mask], dim=1)

        # 从记忆中读取相关信息
        # memory_output = self.read_memory(concatenated_features, mask)
        memory_output = self.read_memory_avg(concatenated_features, mask)

        # 用读取了记忆信息的结果做self-attention，使未来嵌入部分能够关注下历史帧和当前帧的信息
        memory_output = self.memory_self_attention(memory_output, memory_output, memory_output)[0]
        
        # 只提取增强特征中的未来部分（最后future_length个时间步）
        # 形状: [batch_size, future_length, d_model]
        future_part = memory_output[:, -self.future_length:, :]
        
        if torch.isnan(future_part).any():
            print("in interMemory, predict_future, future_part contains NaN")
            print(future_part)
            input("Press Enter to continue...")

        # 除了直接将这部分返回，是不是还可以在返回之前，对这部分做projection
        # 不对，预测结果输出后，就拿来和其他两部份特征拼接了，现在就是从拼接结果中截取的，应该这样就行了

        # 返回一个全为1，形状为 [batch_size, future_length]的mask
        future_mask = torch.ones(batch_size, self.future_length, 
                                device=mask.device, dtype=mask.dtype)
        
        return future_part, future_mask

# class InterMemorySeq(nn.Module):
#     """
#     序列化interMemory，每个记忆槽存储特征序列而不是一个特征向量
#     """
#     def __init__(self, 
#                  d_model=256,            # 模型特征维度
#                  future_length=32,       # 未来特征长度
#                  memory_slots=64,        # 记忆槽数量
#                  slot_len=64,            # 每个槽的序列长度
#                  similarity_threshold=0.1, # 相似度阈值，控制记忆槽更新
#                  num_heads=8,            # 注意力头数量
#                  pooling_type='mean'):   # 池化类型：'mean'或'max'，用于输入序列和记忆槽的相似度计算
#         super(InterMemorySeq, self).__init__()

#         self.d_model = d_model
#         self.future_length = future_length
#         self.memory_slots = memory_slots
#         self.slot_len = slot_len
#         self.similarity_threshold = similarity_threshold
#         self.num_heads = num_heads
#         self.pooling_type = pooling_type
#         self.updateCnt = 0

#         # 初始化序列化记忆槽，形状为 [memory_slots, slot_len, d_model]
#         self.memory = nn.Parameter(torch.randn(memory_slots, slot_len, d_model))
#         # # 归一化初始记忆
#         # self.memory.data = F.normalize(self.memory.data, p=2, dim=-1)
#         # 记忆槽使用频率跟踪
#         self.register_buffer('slot_usage_counter', torch.zeros(memory_slots))

#         # 特征映射层，用于相似度计算
#         self.query_proj = nn.Linear(d_model, d_model)
#         self.memory_proj = nn.Linear(d_model, d_model)
        
#         # # 特征映射层，用于生成记忆更新的写入值
#         # self.write_val_proj = nn.Linear(d_model, d_model)
        
#         # 交叉注意力层，用于记忆更新
#         self.memory_update_attention = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=num_heads,
#             batch_first=True
#         )
        
#         # 控制门机制，决定更新比例
#         self.update_gate = nn.Linear(d_model * 2, d_model)
        
#         # 未来特征初始化嵌入
#         self.future_embedding = nn.Parameter(torch.randn(future_length, d_model))
        
#         # 用于从记忆中读取信息的注意力机制
#         self.memory_read_attention = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=num_heads,
#             batch_first=True
#         )

#         # 用于对记忆读取结果进行自注意力处理的模块
#         self.memory_self_attention = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=num_heads,
#             batch_first=True
#         )

#     def reset(self):
#         self.memory.data = torch.randn(self.memory_slots, self.slot_len, self.d_model)
#         # self.memory.data = F.normalize(self.memory.data, p=2, dim=-1)
#         self.slot_usage_counter.zero_()
#         self.updateCnt = 0

#     def getUpdateCnt(self):
#         return self.updateCnt
    
#     def setInterMemory(self, memory,future_embedding, updateCnt):
#         self.memory = memory
#         self.future_embedding = future_embedding
#         self.updateCnt = updateCnt

#     def frobenius_similarity(self, seq1, seq2, mask1=None, mask2=None):
#         """
#         计算两个带掩码的序列的Frobenius相似度
#         Args:
#             seq1: [seq_len1, d_model]
#             seq2: [seq_len2, d_model]
#             mask1: [seq_len1], 1表示有效，0表示无效（可选）
#             mask2: [seq_len2], 1表示有效，0表示无效（可选）
#             eps: 防止除零的小常数
#         Returns:
#             similarity: 标量值
#         """
#         # 归一化序列
#         seq1_norm = F.normalize(seq1, p=2, dim=-1)
#         seq2_norm = F.normalize(seq2, p=2, dim=-1)

#         # 应用掩码（如果提供）
#         if mask1 is not None:
#             seq1_norm = seq1_norm * mask1.unsqueeze(-1).float()
#         if mask2 is not None:
#             seq2_norm = seq2_norm * mask2.unsqueeze(-1).float()
        
#         sim_matrix = torch.matmul(seq1_norm, seq2_norm.t())  # [seq_len1, seq_len2]

#         # 计算有效元素数量
#         if mask1 is not None and mask2 is not None:
#             valid_counts = mask1.float().sum() * mask2.float().sum()
#         elif mask1 is not None:
#             valid_counts = mask1.float().sum() * seq2.shape[0]
#         elif mask2 is not None:
#             valid_counts = seq1.shape[0] * mask2.float().sum()
#         else:
#             valid_counts = seq1.shape[0] * seq2.shape[0]

#         # 平均相似度
#         similarity = sim_matrix.sum() / (valid_counts + 1e-8)
#         return similarity

#     def compute_sequence_similarity_pool_cosSim(self, features, mask=None):
#         """
#         计算输入特征序列与所有记忆槽的相似度
        
#         Args:
#             features: 输入特征，形状为 [batch_size, seq_length, d_model]
#             mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
#                     如果为None，则视所有特征为有效
                    
#         Returns:
#             similarities: 相似度矩阵，形状为 [batch_size, memory_slots]
#         """
#         batch_size, seq_length, _ = features.shape
        
#         # 如果没有提供mask，则默认所有特征都有效
#         if mask is None:
#             mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
#         else:
#             # 确保mask是布尔类型
#             if mask.dtype != torch.bool:
#                 mask = mask.bool()
        
#         # 将特征和记忆投影到相似度空间
#         # 这里两个序列用两个proj，可以支持输入特征维度和interMemory内部维度不同的情况
#         projected_features = self.query_proj(features)  # [batch_size, seq_length, d_model]
#         projected_memory = self.memory_proj(self.memory)  # [memory_slots, slot_len, d_model]
        
#         # 根据指定的池化类型进行池化操作
#         # 对特征进行池化，考虑mask
#         if self.pooling_type == 'mean':
#             # 对masked位置设为0，然后求和除以有效元素数量
#             mask_expanded = mask.unsqueeze(-1).float()  # [batch_size, seq_length, 1]
#             pooled_features = (projected_features * mask_expanded).sum(dim=1)  # [batch_size, d_model]
#             # 计算每个batch中有效特征的数量（防止除零）
#             valid_counts = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [batch_size, 1]
#             pooled_features = pooled_features / valid_counts  # [batch_size, d_model]
#         elif self.pooling_type == 'max':
#             # 对masked位置设为一个很小的值（如-1e9）
#             mask_expanded = mask.unsqueeze(-1)  # [batch_size, seq_length, 1]
#             masked_features = projected_features.clone()
#             masked_features[~mask_expanded] = -1e9  # 将无效位置设为一个很小的值
#             pooled_features = torch.max(masked_features, dim=1)[0]  # [batch_size, d_model]
#         else:
#             raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
#         # 对记忆槽进行池化
#         if self.pooling_type == 'mean':
#             pooled_memory = projected_memory.mean(dim=1)  # [memory_slots, d_model]
#         elif self.pooling_type == 'max':
#             pooled_memory = torch.max(projected_memory, dim=1)[0]  # [memory_slots, d_model]
        
#         # # 归一化池化后的特征和记忆，用于计算余弦相似度
#         pooled_features_norm = F.normalize(pooled_features, p=2, dim=1)  # [batch_size, d_model]
#         pooled_memory_norm = F.normalize(pooled_memory, p=2, dim=1)  # [memory_slots, d_model]
        
#         # 计算余弦相似度
#         # [batch_size, memory_slots] = [batch_size, d_model] @ [d_model, memory_slots]
#         similarities = torch.matmul(pooled_features_norm, pooled_memory_norm.t())
        
#         return similarities

#     def compute_sequence_similarity_frobenius(self, features, mask=None):
#         """
#         计算输入特征序列与所有记忆槽的Frobenius相似度
        
#         Args:
#             features: 输入特征，形状为 [batch_size, seq_length, d_model]
#             mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
#                     如果为None，则视所有特征为有效
                    
#         Returns:
#             similarities: 相似度矩阵，形状为 [batch_size, memory_slots]
#         """
#         batch_size, seq_length, _ = features.shape
#         memory_slots, slot_len, _ = self.memory.shape
        
#         # 如果没有提供mask，则默认所有特征都有效
#         if mask is None:
#             mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
#         else:
#             # 确保mask是布尔类型
#             if mask.dtype != torch.bool:
#                 mask = mask.bool()
        
#         # 将特征和记忆投影到相似度空间
#         projected_features = self.query_proj(features)  # [batch_size, seq_length, d_model]
#         projected_memory = self.memory_proj(self.memory)  # [memory_slots, slot_len, d_model]
        
#         # 初始化相似度矩阵
#         similarities = torch.zeros(batch_size, memory_slots, device=features.device)
        
#         # 对每个batch和每个memory slot计算Frobenius相似度
#         for b in range(batch_size):
#             for m in range(memory_slots):
#                 # 获取当前batch的特征和对应的mask
#                 current_features = projected_features[b]  # [seq_length, d_model]
#                 current_mask = mask[b] if mask is not None else None  # [seq_length]
                
#                 # 获取当前memory slot的特征
#                 current_memory = projected_memory[m]  # [slot_len, d_model]
                
#                 # 计算Frobenius相似度
#                 similarities[b, m] = self.frobenius_similarity(
#                     current_features, 
#                     current_memory,
#                     current_mask,
#                     None  # 假设memory slot没有mask
#                 )
        
#         return similarities

#     def update_memory_normal(self, features, mask=None, short_start=None, vid=None):
#         """
#         基于输入特征序列更新记忆槽
#         使用注意力机制融合信息到相似度高的记忆槽
        
#         Args:
#             features: 输入特征，形状为 [batch_size, seq_length, d_model]
#             mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
#                     如果为None，则视所有特征为有效
#             short_start: 
#             vid: 这两个暂时用不到了，先留着
#         """
#         self.updateCnt += 1
#         batch_size, seq_length, _ = features.shape
#         device = features.device
        
#         # 如果没有提供mask，则默认所有特征都有效
#         if mask is None:
#             mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
#         else:
#             # 确保mask是布尔类型
#             if mask.dtype != torch.bool:
#                 mask = mask.bool()
        
#         # 计算输入特征与记忆槽的相似度
#         # similarities = self.compute_sequence_similarity(features, mask)  # [batch_size, memory_slots]
#         similarities = self.compute_sequence_similarity_frobenius(features, mask)  # [batch_size, memory_slots]
        
#         # 统计每个batch更新的记忆槽数量
#         update_counts = 0

#         # 逐批次处理更新
#         for b in range(batch_size):
#             # 检查当前批次是否有有效特征
#             if not mask[b].any():
#                 continue
            
#             # 获取当前批次的特征和掩码
#             batch_features = features[b].unsqueeze(0)  # [1, seq_length, d_model]
#             batch_mask = mask[b].unsqueeze(0)  # [1, seq_length]
            
#             # 找出相似度超过阈值的记忆槽
#             similarity = similarities[b]  # [memory_slots]
            
#             # 每个槽的相似度阈值都是动态的，使用计数越高，阈值越高，最大不超过self.similarity_threshold的10倍
#             sim_thd = self.slot_usage_counter*self.similarity_threshold
#             sim_thd = sim_thd.clamp(max=self.similarity_threshold*10)
#             update_mask = (similarity > sim_thd)

#             # # 固定的相似度阈值
#             # update_mask = (similarity > self.similarity_threshold)
            
#             if not update_mask.any():
#                 continue
            
#             # 获取需要更新的记忆槽索引
#             update_indices = torch.where(update_mask)[0]

#             # 记录当前batch更新的记忆槽数量
#             update_counts += len(update_indices)
            
#             # 更新所选记忆槽
#             for idx in update_indices:
#                 # 增加使用计数
#                 self.slot_usage_counter[idx] += 1
                
#                 # 获取当前记忆槽
#                 current_memory = self.memory[idx].unsqueeze(0)  # [1, slot_len, d_model]
                
#                 # 准备注意力掩码
#                 if not batch_mask.all():    # 有被masked掉的位置
#                     # 创建注意力掩码，将无效位置设置为很小的值
#                     attn_mask = batch_mask.float()
#                     attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
#                     attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
#                     # 将attn_mask扩展到 [1, 1, seq_length]-->[1, slot_len, seq_length]
#                     attn_mask = attn_mask.unsqueeze(1).expand(-1, self.slot_len, -1)
#                     # 调整为多头注意力需要的形状[num_heads, slot_len, seq_length]
#                     attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
#                 else:
#                     attn_mask = None
                
#                 # 使用交叉注意力机制更新记忆
#                 # 记忆槽作为query，输入特征作为key和value
#                 updated_memory, _ = self.memory_update_attention(
#                     query=current_memory,        # [1, slot_len, d_model]
#                     key=batch_features,          # [1, seq_length, d_model]
#                     value=batch_features,        # [1, seq_length, d_model]
#                     attn_mask=attn_mask          # [num_heads, slot_len, seq_length] or None
#                 )
                
#                 # 使用门控机制控制更新比例
#                 # 拼接原始记忆和更新后的记忆
#                 concat_memory = torch.cat([current_memory, updated_memory], dim=-1)  # [1, slot_len, 2*d_model]
#                 update_gate = torch.sigmoid(self.update_gate(concat_memory))  # [1, slot_len, d_model]
                
#                 # 应用门控更新
#                 new_memory = current_memory * (1 - update_gate) + updated_memory * update_gate
                
#                 # 更新记忆槽
#                 self.memory.data[idx] = new_memory.squeeze(0)
        
#         # 打印更新的记忆槽数量
#         print(f"在这批样本中，进行了{update_counts}次记忆槽更新.\n最大相似度为{similarities.max()},平均相似度为{similarities.mean()}")

#         # # 对更新后的记忆进行归一化
#         # self.memory.data = F.normalize(self.memory.data, p=2, dim=-1)
        
#         # 检查是否存在NaN值
#         if torch.isnan(self.memory).any():
#             print("after update, memory contains NaN")
#             input("Press Enter to continue...")

#     def update_memory_competitive(self, features, temperature, mask=None, short_start=None, vid=None):
#         """
#         实现竞争性记忆更新机制，使记忆槽更专注于不同的模式
        
#         Args:
#             features: 输入特征，形状为 [batch_size, seq_length, d_model]
#             mask: 特征有效性掩码，形状为 [batch_size, seq_length]
#             temperature: 温度参数，控制竞争程度，值越小竞争越激烈
#         """
#         self.updateCnt += 1
#         batch_size, seq_length, _ = features.shape
#         device = features.device
        
#         # 如果没有提供mask，则默认所有特征都有效
#         if mask is None:
#             mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
#         else:
#             # 确保mask是布尔类型
#             if mask.dtype != torch.bool:
#                 mask = mask.bool()
        
#         # 计算输入特征与记忆槽的相似度
#         similarities = self.compute_sequence_similarity_pool_cosSim(features, mask)  # [batch_size, memory_slots]
        
#         # 逐批次处理更新
#         for b in range(batch_size):
#             # 检查当前批次是否有有效特征
#             if not mask[b].any():
#                 continue
            
#             # 获取当前批次的特征和掩码
#             batch_features = features[b].unsqueeze(0)  # [1, seq_length, d_model]
#             batch_mask = mask[b].unsqueeze(0)  # [1, seq_length]
            
#             # 获取当前批次与所有记忆槽的相似度，并应用温度参数
#             similarity = similarities[b] / temperature  # [memory_slots]
            
#             # 使用softmax获取竞争性权重，温度越低，分布越尖锐，胜者通吃效应越明显
#             update_weights = F.softmax(similarity, dim=0)  # [memory_slots]
            
#             # 确定需要更新的记忆槽
#             min_weight_threshold = 1.0 / self.memory_slots  # 以平均权重为阈值
#             update_mask = (update_weights > min_weight_threshold)
            
#             if not update_mask.any():
#                 continue
            
#             # 获取需要更新的记忆槽索引
#             update_indices = torch.where(update_mask)[0]
            
#             # 准备注意力掩码
#             if not batch_mask.all():    # 有被masked掉的位置
#                 # 创建注意力掩码，将无效位置设置为很小的值
#                 attn_mask = batch_mask.float()
#                 attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
#                 attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
#                 # 将attn_mask扩展到 [1, 1, seq_length]-->[1, slot_len, seq_length]
#                 attn_mask = attn_mask.unsqueeze(1).expand(-1, self.slot_len, -1)
#                 # 调整为多头注意力需要的形状[num_heads, slot_len, seq_length]
#                 attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
#             else:
#                 attn_mask = None
            
#             # 更新所选记忆槽，更新强度与权重成正比
#             for idx in update_indices:
#                 # 增加使用计数，与权重成正比
#                 weight = update_weights[idx].item()
#                 self.slot_usage_counter[idx] += 1
                
#                 # 获取当前记忆槽
#                 current_memory = self.memory[idx].unsqueeze(0)  # [1, slot_len, d_model]
                
#                 # 使用交叉注意力机制更新记忆
#                 updated_memory, _ = self.memory_update_attention(
#                     query=current_memory,        # [1, slot_len, d_model]
#                     key=batch_features,          # [1, seq_length, d_model]
#                     value=batch_features,        # [1, seq_length, d_model]
#                     attn_mask=attn_mask          # [num_heads, slot_len, seq_length] or None
#                 )
                
#                 # 使用门控机制控制更新比例，受权重影响
#                 concat_memory = torch.cat([current_memory, updated_memory], dim=-1)  # [1, slot_len, 2*d_model]
#                 update_gate = torch.sigmoid(self.update_gate(concat_memory))  # [1, slot_len, d_model]
                
#                 # 根据权重调整更新门的强度，权重越高，更新越强
#                 adjusted_gate = update_gate * weight
                
#                 # 应用门控更新
#                 new_memory = current_memory * (1 - adjusted_gate) + updated_memory * adjusted_gate
                
#                 # 更新记忆槽
#                 self.memory.data[idx] = new_memory.squeeze(0)
        
#         # # 对更新后的记忆进行归一化
#         # self.memory.data = F.normalize(self.memory.data, p=2, dim=2)
        
#         # 检查是否存在NaN值
#         if torch.isnan(self.memory).any():
#             print("after update, memory contains NaN")
#             input("Press Enter to continue...")

#         # 方法2: Frobenius范数相似度

#     def reorganize_memory(self, reorganize_threshold=0.8, update_interval=50):
#         """
#         定期整理记忆槽，将高度相似的记忆槽合并，释放槽位存储新信息
        
#         Args:
#             reorganize_threshold: 触发合并的相似度阈值
#             update_interval: 执行整理的更新间隔
#         """
#         # 每隔一定次数的更新才执行整理
#         if self.updateCnt == 0 or self.updateCnt % update_interval!=0:
#             return
        
#         print(f"Reorganizing memory at update count {self.updateCnt}\n")
        
#         # 计算记忆槽之间的相似度矩阵
#         # # 先对每个记忆槽序列进行池化
#         # if self.pooling_type == 'mean':
#         #     pooled_memory = self.memory.mean(dim=1)  # [memory_slots, d_model]
#         # elif self.pooling_type == 'max':
#         #     pooled_memory = torch.max(self.memory, dim=1)[0]  # [memory_slots, d_model]
        
#         # # 归一化池化后的记忆
#         # memory_norm = F.normalize(pooled_memory, p=2, dim=1)  # [memory_slots, d_model]
        
#         # # 计算成对余弦相似度矩阵
#         # similarity_matrix = torch.matmul(memory_norm, memory_norm.t())  # [memory_slots, memory_slots]
#         # # similarity_matrix = torch.matmul(pooled_memory, pooled_memory.t())

#         device = self.memory.device
#         similarity_matrix = torch.zeros(self.memory_slots, self.memory_slots, device=device)

#         # 将self.memory投影
#         projected_memory = self.memory_proj(self.memory)

#         # 计算所有槽对之间的相似度
#         for i in range(self.memory_slots):
#             for j in range(i, self.memory_slots):  # 只计算上三角，利用对称性
#                 if i == j:
#                     similarity_matrix[i, j] = 1.0  # 自相似度为1
#                 else:
#                     sim = self.frobenius_similarity(projected_memory[i], projected_memory[j])
#                     similarity_matrix[i, j] = sim
#                     similarity_matrix[j, i] = sim  # 对称赋值
        
#         # 打印前5个记忆槽的相似度
#         print("Similarity matrix:")
#         print(similarity_matrix)
        
#         # 移除对角线上的自相似度（值为1）
#         eye_mask = torch.eye(self.memory_slots, device=self.memory.device)
#         similarity_matrix = similarity_matrix * (1 - eye_mask)
        
#         # 寻找高度相似的记忆槽对
#         high_similarity_pairs = torch.where(similarity_matrix > reorganize_threshold)
        
#         # 创建已处理记忆槽的集合
#         processed_slots = set()
        
#         # 处理每一对高度相似的记忆槽
#         for i, j in zip(high_similarity_pairs[0].tolist(), high_similarity_pairs[1].tolist()):
#             # 如果槽已经被处理过，则跳过
#             if i in processed_slots or j in processed_slots:
#                 continue
            
#             # 将两个相似槽加入处理集合
#             processed_slots.add(i)
#             processed_slots.add(j)
            
#             # 根据更新频率选择保留哪个槽
#             if self.slot_usage_counter[i] > self.slot_usage_counter[j]:
#                 keep_idx, reset_idx = i, j
#             else:
#                 keep_idx, reset_idx = j, i
            
#             # 合并两个记忆槽 (按更新频率加权平均)
#             total_usage = self.slot_usage_counter[i] + self.slot_usage_counter[j]
#             weight_i = self.slot_usage_counter[i] / total_usage
#             weight_j = self.slot_usage_counter[j] / total_usage
            
#             # 将合并结果保存到使用频率更高的槽中
#             self.memory.data[keep_idx] = (
#                 self.memory[i] * weight_i + self.memory[j] * weight_j
#             )
            
#             # 重置使用频率较低的槽
#             # 随机初始化，确保与其他槽保持差异
#             self.memory.data[reset_idx] = torch.randn_like(self.memory[reset_idx])
#             self.slot_usage_counter[reset_idx] = 0
        
#         # # 对所有记忆槽进行归一化
#         # self.memory.data = F.normalize(self.memory.data, p=2, dim=2)
        
#         # 如果处理了记忆槽，打印信息
#         if processed_slots:
#             print(f"Reorganized {len(processed_slots)//2} pairs of memory slots.\n")

#     def update_memory_reorganization(self, features, mask=None, 
#                                         reorganize_threshold=0.8, 
#                                         update_interval=50,
#                                         temperature=0.5):
#         """
#         更新记忆并定期整理，包装了更新和整理功能
        
#         Args:
#             features: 输入特征，形状为 [batch_size, seq_length, d_model]
#             mask: 特征有效性掩码，形状为 [batch_size, seq_length]
#             similarity_threshold: 触发记忆槽合并的相似度阈值
#             update_interval: 执行整理的更新间隔
#             temperature: 竞争性更新的温度参数
#         """
#         # 执行普通记忆更新
#         self.update_memory_normal(features, mask)
#         # 竞争性更新
#         # self.update_memory_competitive(features, temperature, mask)      
        
#         # 执行记忆槽整理
#         self.reorganize_memory(reorganize_threshold, update_interval)

#     def update_memory(self, features, type, mask=None, temperature=0.5, reorganize_threshold=0.8, update_interval=50, short_start=None, vid=None):
#         if type == "normal":
#             self.update_memory_normal(features, mask)
#         elif type == "competitive" and temperature is not None:
#             self.update_memory_competitive(features, temperature, mask)
#         elif type == "reorganization" and reorganize_threshold is not None and update_interval is not None and temperature is not None:
#             self.update_memory_reorganization(features, mask=mask, reorganize_threshold=reorganize_threshold, update_interval=update_interval, temperature=temperature)

#     def read_memory(self, query_features, mask=None):
#         """
#         从所有记忆槽中读取信息并平均结果
        
#         Args:
#             query_features: 查询特征，形状为 [batch_size, seq_length, d_model]
#             mask: 查询特征的有效性掩码，形状为 [batch_size, seq_length]
            
#         Returns:
#             memory_output: 从记忆中读取的输出，形状为 [batch_size, seq_length, d_model]
#         """
#         batch_size, seq_length, _ = query_features.shape
#         device = query_features.device
        
#         # 创建一个存储所有槽注意力结果的张量
#         all_slot_outputs = torch.zeros(batch_size, seq_length, self.d_model, device=device)
        
#         # 处理掩码
#         if mask is None:
#             attn_mask = None
#         else:
#             if mask.dtype == torch.bool:
#                 mask_float = mask.float()  # True->1.0, False->0.0
#             else:
#                 mask_float = mask.clamp(0, 1)  # 确保是 0/1
            
#             # 转换为加性注意力掩码格式[batch_size, seq_length, 1]
#             attn_mask = mask_float.unsqueeze(-1)
#             attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
#             attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        
#         # 遍历每个记忆槽
#         for i in range(self.memory_slots):
#             # 获取当前记忆槽
#             # 形状：[slot_len, d_model] -> [1, slot_len, d_model] -> [batch_size, slot_len, d_model]
#             current_slot = self.memory[i].unsqueeze(0).expand(batch_size, -1, -1)
            
#             # 如果有掩码，调整为当前设置需要的格式
#             if attn_mask is not None:
#                 # 创建交叉注意力的掩码
#                 # 查询是query_features，键和值是current_slot
#                 # 掩码形状应为 [batch_size, seq_length, slot_len]
#                 current_mask = attn_mask.expand(-1, -1, self.slot_len)
#                 # 调整为多头注意力需要的形状
#                 current_mask = current_mask.repeat_interleave(self.num_heads, dim=0)
#             else:
#                 current_mask = None
            
#             # 执行交叉注意力，query_features作为query，记忆槽作为key和value
#             slot_output, _ = self.memory_read_attention(
#                 query=query_features,           # [batch_size, seq_length, d_model]
#                 key=current_slot,               # [batch_size, slot_len, d_model]
#                 value=current_slot,             # [batch_size, slot_len, d_model]
#                 attn_mask=current_mask          # [B*num_heads, seq_length, slot_len] or None
#             )
            
#             # 将结果添加到输出中
#             all_slot_outputs += slot_output
        
#         # 计算平均值
#         memory_output = all_slot_outputs / self.memory_slots
        
#         # 检查是否存在NaN值
#         if torch.isnan(memory_output).any():
#             print("in interMemorySeq, read_memory, memory_output contains NaN")
#             torch.set_printoptions(profile="full")
#             for i in range(batch_size):
#                 if torch.isnan(memory_output[i]).any():
#                     print(f"Batch {i} contains NaN")
#                     print(memory_output[i])
#                     input("Press Enter to continue...")
#             input("all nan batches is found, Press Enter to continue...")
        
#         return memory_output

#     def predict_future(self, history_features, current_features, mask=None):
#         """
#         基于历史特征、当前特征和记忆预测未来特征
        
#         Args:
#             history_features: 历史特征，形状为 [batch_size, history_length, d_model]
#             current_features: 当前特征，形状为 [batch_size, current_length, d_model]
#             mask: 历史特征的有效性掩码，形状为 [batch_size, history_length]
            
#         Returns:
#             future_features: 预测的未来特征，形状为 [batch_size, future_length, d_model]
#             future_mask: 未来特征的掩码，形状为 [batch_size, future_length]
#         """
#         batch_size = history_features.shape[0]
#         device = history_features.device
        
#         # 扩展未来嵌入以匹配批次大小
#         # 形状: [batch_size, future_length, d_model]
#         future_embed = self.future_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
#         # 拼接历史特征、当前特征和未来嵌入
#         # 形状: [batch_size, (history_length + current_length + future_length), d_model]
#         concatenated_features = torch.cat([history_features, current_features, future_embed], dim=1)

#         # 拼接特征的mask
#         if mask is not None:
#             current_mask = torch.ones(batch_size, current_features.shape[1], 
#                                     device=device, dtype=mask.dtype)
#             future_mask = torch.ones(batch_size, future_embed.shape[1], 
#                                    device=device, dtype=mask.dtype)
#             concat_mask = torch.cat([mask, current_mask, future_mask], dim=1)
#         else:
#             concat_mask = None

#         # 从记忆中读取相关信息
#         memory_output = self.read_memory(concatenated_features, concat_mask)

#         # 用读取了记忆信息的结果做self-attention，使未来嵌入部分能够关注历史帧和当前帧的信息
#         memory_output = self.memory_self_attention(memory_output, memory_output, memory_output)[0]
        
#         # 只提取增强特征中的未来部分（最后future_length个时间步）
#         # 形状: [batch_size, future_length, d_model]
#         future_part = memory_output[:, -self.future_length:, :]
        
#         # 检查是否存在NaN值
#         if torch.isnan(future_part).any():
#             print("in interMemorySeq, predict_future, future_part contains NaN")
#             print(future_part)
#             input("Press Enter to continue...")

#         # 返回预测的未来特征和对应的掩码
#         future_mask = torch.ones(batch_size, self.future_length, 
#                                 device=device, dtype=torch.bool if mask is None 
#                                 else mask.dtype)
        
#         return future_part, future_mask

class InterMemorySeq(nn.Module):
    """
    序列化interMemory，每个记忆槽存储特征序列而不是一个特征向量
    支持自动优化记忆槽多样性的版本
    """
    def __init__(self, 
                 d_model=256,            # 模型特征维度
                 future_length=32,       # 未来特征长度
                 memory_slots=64,        # 记忆槽数量
                 slot_len=64,            # 每个槽的序列长度
                 similarity_threshold=0.1, # 相似度阈值，控制记忆槽更新
                 num_heads=8,            # 注意力头数量
                 pooling_type='mean',    # 池化类型：'mean'或'max'
                 # 新增参数用于自动优化
                 diversity_loss_weight=0.5,  # 多样性损失权重
                 optimization_interval=10,   # 优化间隔（多少次forward后进行一次优化）
                 optimizer_lr=1e-4,         # 内部优化器学习率
                 ):
        super(InterMemorySeq, self).__init__()

        # 原有参数
        self.d_model = d_model
        self.future_length = future_length
        self.memory_slots = memory_slots
        self.slot_len = slot_len
        self.similarity_threshold = similarity_threshold
        self.num_heads = num_heads
        self.pooling_type = pooling_type
        self.updateCnt = 0

        # 新增自动优化相关参数
        self.diversity_loss_weight = diversity_loss_weight
        self.optimization_interval = optimization_interval
        
        # 初始化序列化记忆槽，形状为 [memory_slots, slot_len, d_model]

        # 1.随机初始化
        self.memory = nn.Parameter(torch.randn(memory_slots, slot_len, d_model))
        # 2.正交初始化
        # memory = torch.empty(memory_slots, slot_len, d_model)
        # nn.init.orthogonal_(memory)
        # self.memory = nn.Parameter(memory)
        # 3.傅里叶基初始化
        # def fourier_basis_initialization(memory_slots, slot_len, d_model):
        #     """用不同频率的傅里叶基函数初始化记忆槽"""
        #     memory = torch.zeros(memory_slots, slot_len, d_model)
            
        #     # 生成不同的频率模式
        #     freqs = torch.linspace(0.1, 1.0, memory_slots)
            
        #     for i in range(memory_slots):
        #         for j in range(d_model):
        #             # 每个特征维度用不同相位的正弦波
        #             phase = j * 2 * torch.pi / d_model
        #             memory[i, :, j] = torch.sin(freqs[i] * torch.arange(slot_len) + phase)
            
        #     return memory

        # self.memory = nn.Parameter(fourier_basis_initialization(memory_slots, slot_len, d_model))

        # 记忆槽使用频率跟踪
        self.register_buffer('slot_usage_counter', torch.zeros(memory_slots))
        
        # 优化步数计数器
        self.register_buffer('forward_counter', torch.tensor(0))

        # 特征映射层，用于相似度计算
        self.query_proj = nn.Linear(d_model, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)
        
        # 交叉注意力层，用于记忆更新
        self.memory_update_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 控制门机制，决定更新比例
        self.update_gate = nn.Linear(d_model * 2, d_model)
        
        # 未来特征初始化嵌入
        self.future_embedding = nn.Parameter(torch.randn(future_length, d_model))
        
        # 用于从记忆中读取信息的注意力机制
        self.memory_read_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # 用于对记忆读取结果进行自注意力处理的模块
        self.memory_self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # 简单分类器 - MLP，用于判断输入特征是否更新
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )

        # 内部优化器，用于优化记忆多样性
        self.memory_optimizer = torch.optim.Adam([self.memory], lr=optimizer_lr)
            

    def reset(self):
        """重置记忆状态和计数器"""
        self.memory.data = torch.randn(self.memory_slots, self.slot_len, self.d_model)
        self.slot_usage_counter.zero_()
        self.forward_counter.zero_()
        self.updateCnt = 0

    def getUpdateCnt(self):
        return self.updateCnt
    
    def setInterMemory(self, memory=None,future_embedding=None, updateCnt=None):
        if memory is not None:
            self.memory = memory
        if future_embedding is not None:
            self.future_embedding = future_embedding
        if updateCnt is not None:
            self.updateCnt = updateCnt

    def classify(self, x):
        """
        分类方法，直接返回0/1结果
        推理的时候调用，可以在之前加个 with torch.no_grad():

        Args:
            x: [batch_size, seq_length, d_model]
        
        Returns:
            predictions: [batch_size] 0/1分类结果
        """
        pooled = torch.mean(x, dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return (torch.sigmoid(logits) > 0.5).long()

    def compute_diversity_loss_negCos(self):
        """
        计算记忆槽多样性损失
        使用负余弦相似度鼓励记忆槽之间的差异化
        
        Returns:
            diversity_loss: 多样性损失值
        """

        device = self.memory.device
        frob_similarity_matrix = torch.zeros(self.memory_slots, self.memory_slots, device=device)

        # 将self.memory投影
        projected_memory = self.memory_proj(self.memory)

        # 计算所有槽对之间的相似度
        for i in range(self.memory_slots):
            for j in range(i, self.memory_slots):  # 只计算上三角，利用对称性
                if i == j:
                    frob_similarity_matrix[i, j] = 1.0  # 自相似度为1
                else:
                    sim = self.frobenius_similarity(projected_memory[i], projected_memory[j])
                    # sim = self.hybrid_pool_similarity(projected_memory[i], projected_memory[j])
                    # sim = self.multi_scale_semantic_similarity(projected_memory[i], projected_memory[j])
                    frob_similarity_matrix[i, j] = sim
                    frob_similarity_matrix[j, i] = sim  # 对称赋值
        
        # 排除对角线元素（自己与自己的相似度）
        mask = torch.eye(self.memory_slots, device=self.memory.device, dtype=torch.bool)
        frob_similarity_matrix = frob_similarity_matrix.masked_fill(mask, 0.0)
        
        # 计算平均余弦相似度作为损失（希望这个值越小越好）
        # 分母是 N(N-1)，即除了对角线的所有元素数量
        diversity_loss = frob_similarity_matrix.sum() / (self.memory_slots * (self.memory_slots - 1))
        
        return diversity_loss

    def compute_diversity_loss_ortho(self):
        """
        计算基于正交化正则的记忆槽多样性损失
        使用Frobenius范数计算MM^T与单位矩阵的差异
        这个loss的范围应该在[0,2N(N−1)]，N是记忆槽的数量
        
        Returns:
            ortho_loss: 正交性损失值 (越小表示记忆槽越正交)
        """
        # 获取记忆槽的表示 (memory_slots, slot_len, d_model)
        memory = self.memory
        
        # 沿序列维度做平均池化，得到每个槽的聚合表示 (memory_slots, d_model)
        memory_repr = memory.mean(dim=1)
        
        # 对记忆槽表示进行L2归一化 
        memory_repr = F.normalize(memory_repr, p=2, dim=1)
        
        # 计算MM^T (memory_slots, memory_slots)
        mmt = torch.mm(memory_repr, memory_repr.t())
        
        # 创建单位矩阵
        identity = torch.eye(self.memory_slots, device=memory.device)
        
        # 计算Frobenius范数的平方差
        ortho_loss = torch.norm(mmt - identity, p='fro') ** 2
        
        return ortho_loss

    def optimize_memory_diversity(self):
        """
        执行记忆多样性优化
        """
        
        with torch.set_grad_enabled(True):  # 临时启用梯度
            # 计算多样性损失
            # diversity_loss = self.compute_diversity_loss_negCos()
            diversity_loss = self.compute_diversity_loss_ortho()
            
            # 加权损失
            memory_loss = self.diversity_loss_weight * diversity_loss
            
            # 执行优化步骤
            self.memory_optimizer.zero_grad()
            try:
                memory_loss.backward()
            except Exception as e:
                print(f"Memory diversity optimization failed: {e}")
                print("diversity loss: ", diversity_loss)
                print("memory loss: ", memory_loss)
                print("optimizer params: ", self.memory_optimizer.param_groups)
                input("Press Enter to continue...")
                return
            self.memory_optimizer.step()
        
        with open("interMemory_Optimization.log", 'a') as f:
            f.write(f"Memory diversity optimization: loss = {memory_loss.item():.6f}\n")
        print(f"Memory diversity optimization: loss = {memory_loss.item():.6f}")

    def frobenius_similarity(self, seq1, seq2, mask1=None, mask2=None):
        """
        计算两个带掩码的序列的Frobenius相似度
        Args:
            seq1: [seq_len1, d_model]
            seq2: [seq_len2, d_model]
            mask1: [seq_len1], 1表示有效，0表示无效（可选）
            mask2: [seq_len2], 1表示有效，0表示无效（可选）
        Returns:
            similarity: 标量值
        """
        # 归一化序列
        seq1_norm = F.normalize(seq1, p=2, dim=-1)
        seq2_norm = F.normalize(seq2, p=2, dim=-1)

        # 应用掩码（如果提供）
        if mask1 is not None:
            seq1_norm = seq1_norm * mask1.unsqueeze(-1).float()
        if mask2 is not None:
            seq2_norm = seq2_norm * mask2.unsqueeze(-1).float()
        
        sim_matrix = torch.matmul(seq1_norm, seq2_norm.t())  # [seq_len1, seq_len2]

        # 计算有效元素数量
        if mask1 is not None and mask2 is not None:
            valid_counts = mask1.float().sum() * mask2.float().sum()
        elif mask1 is not None:
            valid_counts = mask1.float().sum() * seq2.shape[0]
        elif mask2 is not None:
            valid_counts = seq1.shape[0] * mask2.float().sum()
        else:
            valid_counts = seq1.shape[0] * seq2.shape[0]

        # 平均相似度
        similarity = sim_matrix.sum() / (valid_counts + 1e-8)
        return similarity

    def temporal_pool(self, x, pool_layers=2, mask=None):
        """
        时序感知层次化池化
        Args:
            x: [seq_len, d_model] 输入序列，seq_len是序列长度，d_model是特征维度
            mask: [seq_len] (可选) 用于处理变长序列的掩码
        Returns:
            池化后的特征向量 [d_model]
        """
        # 如果提供了掩码，应用掩码
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
            
        # 多尺度时序池化
        # 调整维度为 [1, d_model, seq_len] 以适应1D卷积
        current_seq = x.permute(1, 0).unsqueeze(0)  
        
        # 进行多层池化
        for _ in range(pool_layers):
            # 混合池化层 - 结合平均池化和最大池化的优势
            avg_pool = F.avg_pool1d(current_seq, kernel_size=3, stride=2, padding=1)
            max_pool = F.max_pool1d(current_seq, kernel_size=3, stride=2, padding=1)
            current_seq = 0.5 * (avg_pool + max_pool)  # 平均池化和最大池化的加权平均
            
        # 最终聚合 - 结合全局平均和最大池化
        return 0.5 * (current_seq.mean(dim=-1) + current_seq.max(dim=-1)[0])

    # 计算混合池化相似度
    def hybrid_pool_similarity(self, seq1, seq2, mask1=None, mask2=None):
        """
        计算两个视频序列的相似度
        Args:
            seq1: [seq_len1, d_model] 第一个视频序列
            seq2: [seq_len2, d_model] 第二个视频序列
            mask1: [seq_len1] (可选) 第一个序列的掩码
            mask2: [seq_len2] (可选) 第二个序列的掩码
        Returns:
            相似度标量 (范围[-1,1])
        """
        # 归一化处理 - 使特征向量位于单位球面上
        seq1 = F.normalize(seq1, p=2, dim=-1)
        seq2 = F.normalize(seq2, p=2, dim=-1)
        
        # 时序特征提取
        feat1 = self.temporal_pool(seq1, pool_layers=2, mask=mask1).squeeze()
        feat2 = self.temporal_pool(seq2, pool_layers=2, mask=mask2).squeeze()
        
        # 相似度计算 - 使用余弦相似度
        return F.cosine_similarity(feat1, feat2, dim=0)

    # 计算多尺度相似度
    def multi_scale_semantic_similarity(self, seq1, seq2, mask1=None, mask2=None, 
                                   n_segments=3, top_k_frames=5, 
                                   weights=(0.4, 0.3, 0.3)):
        """
        多尺度语义相似度计算 - 结合全局、分段、关键帧三个层次
        
        Args:
            seq1: [seq_len1, d_model] - 第一个视频序列的特征
            seq2: [seq_len2, d_model] - 第二个视频序列的特征  
            mask1: [seq_len1] - 第一个序列的掩码，1表示有效，0表示无效（可选）
            mask2: [seq_len2] - 第二个序列的掩码，1表示有效，0表示无效（可选）
            n_segments: int - 时间分段数量，默认3段（开始、中间、结束）
            top_k_frames: int - 提取的关键帧数量，默认5帧
            weights: tuple - 三个层次的权重 (全局, 分段, 关键帧)，默认(0.4, 0.3, 0.3)
            
        Returns:
            similarity: torch.Tensor - 标量，语义相似度分数 [0, 1]
        """
        
        # ==================== 数据预处理 ====================
        
        # 如果其中一个有效位长度为0，直接返回0
        if mask1 is not None and mask1.sum().item() == 0:
            return 0.0
        if mask2 is not None and mask2.sum().item() == 0:
            return 0.0

        # L2归一化，便于余弦相似度计算
        seq1_norm = F.normalize(seq1, p=2, dim=-1, eps=1e-8)
        seq2_norm = F.normalize(seq2, p=2, dim=-1, eps=1e-8)
        
        # 应用掩码，将无效位置的特征置零
        if mask1 is not None:
            seq1_norm = seq1_norm * mask1.unsqueeze(-1).float()
            valid_len1 = mask1.sum().item()  # 有效帧数
        else:
            valid_len1 = seq1.shape[0]
            
        if mask2 is not None:
            seq2_norm = seq2_norm * mask2.unsqueeze(-1).float()
            valid_len2 = mask2.sum().item()  # 有效帧数
        else:
            valid_len2 = seq2.shape[0]
        
        # 提取有效序列（去除padding部分）
        seq1_valid = seq1_norm[:valid_len1]
        seq2_valid = seq2_norm[:valid_len2]
        
        
        # ==================== 层次1: 全局语义相似度 ====================
        
        # 计算整个视频的全局语义表示（平均池化）
        global_feat1 = seq1_valid.mean(dim=0)  # [d_model]
        global_feat2 = seq2_valid.mean(dim=0)  # [d_model]
        
        # 计算全局语义相似度
        global_similarity = F.cosine_similarity(global_feat1, global_feat2, dim=0)
        
        # ==================== 层次2: 时间分段语义相似度 ====================
        
        def extract_temporal_segments(seq, n_segments):
            """
            将视频序列按时间均匀分段，提取每段的语义特征
            捕捉视频不同阶段的语义信息（如动作的开始、发展、结束）
            
            Args:
                seq: [seq_len, d_model] - 输入序列
                n_segments: int - 分段数量
                
            Returns:
                segments: [n_segments, d_model] - 每段的语义特征
            """
            seq_len = len(seq)
            seg_len = seq_len // n_segments  # 每段的长度
            segments = []
            
            for i in range(n_segments):
                # 计算当前段的起始和结束位置
                start_idx = i * seg_len
                end_idx = start_idx + seg_len if i < n_segments - 1 else seq_len
                
                # 提取当前段的特征（确保不超出序列长度）
                if start_idx < seq_len:
                    # 段内平均
                    segment = seq[start_idx:end_idx]
                    segment_feat = segment.mean(dim=0) if len(segment) > 0 else torch.zeros_like(seq[0])
                    segments.append(segment_feat)
            
            # 如果某些段为空，用零向量填充
            while len(segments) < n_segments:
                segments.append(torch.zeros_like(seq[0]))
                
            return torch.stack(segments)
        
        # 提取两个视频的时间分段特征
        segments1 = extract_temporal_segments(seq1_valid, n_segments)  # [n_segments, d_model]
        segments2 = extract_temporal_segments(seq2_valid, n_segments)  # [n_segments, d_model]
        
        # 计算分段间的相似度矩阵
        # segment_sim_matrix[i,j] 表示视频1的第i段与视频2的第j段的相似度
        segment_sim_matrix = torch.matmul(segments1, segments2.t())  # [n_segments, n_segments]
        
        # 对于每个段，找到其在另一个视频中的最佳匹配段
        # 这允许不同视频中相同语义内容出现在不同时间位置
        best_matches = segment_sim_matrix.max(dim=1)[0]  # [n_segments]
        segment_similarity = best_matches.max()  # 所有段的最佳匹配相似度，不用mean是因为只要有分成匹配的两个段，后续update会通过attn自动对应上
        
        # ==================== 层次3: 关键帧语义相似度 ====================
        
        def extract_key_frames(seq, top_k):
            """
            提取视频中最重要的关键帧
            使用特征范数作为重要性指标，范数大的帧通常包含更丰富的语义信息
            
            Args:
                seq: [seq_len, d_model] - 输入序列
                top_k: int - 提取的关键帧数量
                
            Returns:
                key_features: [min(top_k, seq_len), d_model] - 关键帧特征
            """
            # 计算每帧特征的L2范数作为重要性分数
            # 范数越大，表示该帧的特征越"突出"，语义信息越丰富
            importance_scores = seq.norm(dim=1)  # [seq_len]
            
            # 选择top-k个最重要的帧
            actual_k = min(top_k, len(seq))
            top_indices = torch.topk(importance_scores, k=actual_k, dim=0)[1]
            
            # 提取关键帧特征
            key_features = seq[top_indices]  # [actual_k, d_model]
            
            return key_features
        
        # 提取两个视频的关键帧
        key_frames1 = extract_key_frames(seq1_valid, top_k_frames)
        key_frames2 = extract_key_frames(seq2_valid, top_k_frames)
        
        # 计算关键帧的整体语义表示（平均池化）
        key_feat1 = key_frames1.mean(dim=0)  # [d_model]
        key_feat2 = key_frames2.mean(dim=0)  # [d_model]
        
        # 计算关键帧语义相似度
        key_similarity = F.cosine_similarity(key_feat1, key_feat2, dim=0)
        
        # ==================== 多尺度融合 ====================
        
        # 将三个层次的相似度按权重融合
        # global: 捕捉整体语义主题
        # segment: 捕捉时间演变模式  
        # key: 捕捉关键语义事件
        final_similarity = (weights[0] * global_similarity + 
                        weights[1] * segment_similarity + 
                        weights[2] * key_similarity)
        
        # 确保结果在合理范围内
        final_similarity = torch.clamp(final_similarity, -1.0, 1.0)
        
        return final_similarity

    def compute_sequence_similarity_pool_cosSim(self, features, mask=None):
        """
        计算输入特征序列与所有记忆槽的相似度
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
                    
        Returns:
            similarities: 相似度矩阵，形状为 [batch_size, memory_slots]
        """
        batch_size, seq_length, _ = features.shape
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
        else:
            # 确保mask是布尔类型
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 将特征和记忆投影到相似度空间
        projected_features = self.query_proj(features)  # [batch_size, seq_length, d_model]
        projected_memory = self.memory_proj(self.memory)  # [memory_slots, slot_len, d_model]
        
        # 根据指定的池化类型进行池化操作
        if self.pooling_type == 'mean':
            # 对特征进行池化，考虑mask
            mask_expanded = mask.unsqueeze(-1).float()  # [batch_size, seq_length, 1]
            pooled_features = (projected_features * mask_expanded).sum(dim=1)  # [batch_size, d_model]
            valid_counts = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [batch_size, 1]
            pooled_features = pooled_features / valid_counts  # [batch_size, d_model]
        elif self.pooling_type == 'max':
            mask_expanded = mask.unsqueeze(-1)  # [batch_size, seq_length, 1]
            masked_features = projected_features.clone()
            masked_features[~mask_expanded] = -1e9
            pooled_features = torch.max(masked_features, dim=1)[0]  # [batch_size, d_model]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        # 对记忆槽进行池化
        if self.pooling_type == 'mean':
            pooled_memory = projected_memory.mean(dim=1)  # [memory_slots, d_model]
        elif self.pooling_type == 'max':
            pooled_memory = torch.max(projected_memory, dim=1)[0]  # [memory_slots, d_model]
        
        # 归一化池化后的特征和记忆，用于计算余弦相似度
        pooled_features_norm = F.normalize(pooled_features, p=2, dim=1)  # [batch_size, d_model]
        pooled_memory_norm = F.normalize(pooled_memory, p=2, dim=1)  # [memory_slots, d_model]
        
        # 计算余弦相似度
        similarities = torch.matmul(pooled_features_norm, pooled_memory_norm.t())
        
        return similarities

    def compute_sequence_similarity_frobenius(self, features, mask=None):
        """
        计算输入特征序列与所有记忆槽的Frobenius相似度
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
                    
        Returns:
            similarities: 相似度矩阵，形状为 [batch_size, memory_slots]
        """
        batch_size, seq_length, _ = features.shape
        memory_slots, slot_len, _ = self.memory.shape
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 将特征和记忆投影到相似度空间
        projected_features = self.query_proj(features)  # [batch_size, seq_length, d_model]
        projected_memory = self.memory_proj(self.memory)  # [memory_slots, slot_len, d_model]
        
        # 初始化相似度矩阵
        similarities = torch.zeros(batch_size, memory_slots, device=features.device)
        
        # 对每个batch和每个memory slot计算Frobenius相似度
        for b in range(batch_size):
            for m in range(memory_slots):
                current_features = projected_features[b]  # [seq_length, d_model]
                current_mask = mask[b] if mask is not None else None  # [seq_length]
                current_memory = projected_memory[m]  # [slot_len, d_model]
                
                similarities[b, m] = self.frobenius_similarity(
                    current_features, 
                    current_memory,
                    current_mask,
                    None
                )
        
        return similarities

    def compute_sequence_similarity_hybrid_pool(self, features, mask=None):
        """
        计算输入特征序列与所有记忆槽的混合池化相似度
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
                    
        Returns:
            similarities: 相似度矩阵，形状为 [batch_size, memory_slots]
        """
        batch_size, seq_length, _ = features.shape
        memory_slots, slot_len, _ = self.memory.shape
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 将特征和记忆投影到相似度空间
        projected_features = self.query_proj(features)  # [batch_size, seq_length, d_model]
        projected_memory = self.memory_proj(self.memory)  # [memory_slots, slot_len, d_model]
        
        # 初始化相似度矩阵
        similarities = torch.zeros(batch_size, memory_slots, device=features.device)
        
        # 对每个batch和每个memory slot计算Frobenius相似度
        for b in range(batch_size):
            for m in range(memory_slots):
                current_features = projected_features[b]  # [seq_length, d_model]
                current_mask = mask[b] if mask is not None else None  # [seq_length]
                current_memory = projected_memory[m]  # [slot_len, d_model]
                
                similarities[b, m] = self.hybrid_pool_similarity(
                    current_features, 
                    current_memory,
                    current_mask,
                    None
                )
        
        return similarities

    def compute_sequence_similarity_multi_scale(self, features, mask=None):
        """
        计算输入特征序列与所有记忆槽的多尺度相似度
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
                    
        Returns:
            similarities: 相似度矩阵，形状为 [batch_size, memory_slots]
        """
        batch_size, seq_length, _ = features.shape
        memory_slots, slot_len, _ = self.memory.shape
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 将特征和记忆投影到相似度空间
        projected_features = self.query_proj(features)  # [batch_size, seq_length, d_model]
        projected_memory = self.memory_proj(self.memory)  # [memory_slots, slot_len, d_model]
        
        # 初始化相似度矩阵
        similarities = torch.zeros(batch_size, memory_slots, device=features.device)
        
        # 对每个batch和每个memory slot计算相似度
        for b in range(batch_size):
            for m in range(memory_slots):
                current_features = projected_features[b]  # [seq_length, d_model]
                current_mask = mask[b] if mask is not None else None  # [seq_length]
                current_memory = projected_memory[m]  # [slot_len, d_model]
                
                similarities[b, m] = self.multi_scale_semantic_similarity(
                    current_features, 
                    current_memory,
                    current_mask,
                    None,
                    n_segments=3,
                    top_k_frames=5,
                    weights=(0.4, 0.3, 0.3)
                )
        
        return similarities
        

    def update_memory_normal(self, features, mask=None, short_start=None, vid=None):
        """
        基于输入特征序列更新记忆槽
        使用注意力机制融合信息到相似度高的记忆槽
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
            short_start: 
            vid: 这两个暂时用不到了，先留着
        """
        self.updateCnt += 1
        batch_size, seq_length, _ = features.shape
        device = features.device
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 计算输入特征与记忆槽的相似度
        similarities = self.compute_sequence_similarity_frobenius(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_hybrid_pool(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_multi_scale(features, mask)  # [batch_size, memory_slots]

        # 统计每个样本输入，多少个记忆槽进行了更新
        update_per_sample = torch.zeros(batch_size, device=device, dtype=torch.int32)

        # 逐批次处理更新
        for b in range(batch_size):
            # 检查当前批次是否有有效特征
            if not mask[b].any():
                continue
            
            # 获取当前批次的特征和掩码
            batch_features = features[b].unsqueeze(0)  # [1, seq_length, d_model]
            batch_mask = mask[b].unsqueeze(0)  # [1, seq_length]
            
            # 找出相似度超过阈值的记忆槽
            similarity = similarities[b]  # [memory_slots]
            
            # # 动态相似度阈值
            sim_thd = self.slot_usage_counter*self.similarity_threshold
            sim_thd = sim_thd.clamp(max=self.similarity_threshold*10)
            update_mask = (similarity > sim_thd)

            # 固定相似度阈值
            # update_mask = (similarity > self.similarity_threshold)
            
            if not update_mask.any():
                continue
            
            # 获取需要更新的记忆槽索引
            update_indices = torch.where(update_mask)[0]
            update_per_sample[b] = len(update_indices)
            
            # 更新所选记忆槽
            for idx in update_indices:
                # 增加使用计数
                self.slot_usage_counter[idx] += 1
                
                # 获取当前记忆槽
                current_memory = self.memory[idx].unsqueeze(0)  # [1, slot_len, d_model]
                
                # 准备注意力掩码
                if not batch_mask.all():
                    attn_mask = batch_mask.float()
                    attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
                    attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, self.slot_len, -1)
                    attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
                else:
                    attn_mask = None
                
                # 使用交叉注意力机制更新记忆
                updated_memory, _ = self.memory_update_attention(
                    query=current_memory,        # [1, slot_len, d_model]
                    key=batch_features,          # [1, seq_length, d_model]
                    value=batch_features,        # [1, seq_length, d_model]
                    attn_mask=attn_mask          # [num_heads, slot_len, seq_length] or None
                )
                
                # 使用门控机制控制更新比例
                concat_memory = torch.cat([current_memory, updated_memory], dim=-1)  # [1, slot_len, 2*d_model]
                update_gate = torch.sigmoid(self.update_gate(concat_memory))  # [1, slot_len, d_model]
                
                # 应用门控更新
                new_memory = current_memory * (1 - update_gate) + updated_memory * update_gate
                
                # 更新记忆槽
                self.memory.data[idx] = new_memory.squeeze(0)
        
        # 如果不是全0，打印每个样本更新的记忆槽数量
        if update_per_sample.sum() > 0:
            print("Update per sample:", update_per_sample)

        # 打印相似度最大值和平均值
        print("Similarity max:", similarities.max())
        print("Similarity mean:", similarities.mean())

        # 检查是否存在NaN值
        if torch.isnan(self.memory).any():
            print("Warning: Memory contains NaN after update")

    def update_memory_normal_topk(self, features, mask=None,topk=4, short_start=None, vid=None):
        """
        基于输入特征序列更新记忆槽, 选择相似度最高的k个进行更新
        使用注意力机制融合信息到相似度高的记忆槽
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
            short_start: 
            vid: 这两个暂时用不到了，先留着
        """
        self.updateCnt += 1
        batch_size, seq_length, _ = features.shape
        device = features.device
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 计算输入特征与记忆槽的相似度
        similarities = self.compute_sequence_similarity_frobenius(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_hybrid_pool(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_multi_scale(features, mask)  # [batch_size, memory_slots]

        # 统计每个样本输入，多少个记忆槽进行了更新
        update_per_sample = torch.zeros(batch_size, device=device, dtype=torch.int32)

        # 逐批次处理更新
        for b in range(batch_size):
            # 检查当前批次是否有有效特征
            if not mask[b].any():
                continue
            
            # 获取当前批次的特征和掩码
            batch_features = features[b].unsqueeze(0)  # [1, seq_length, d_model]
            batch_mask = mask[b].unsqueeze(0)  # [1, seq_length]
            
            # 找出相似度最高的k个记忆槽
            similarity = similarities[b]  # [memory_slots]
            # 找出相似度最高的k个记忆槽
            topk_indices = torch.topk(similarity, k=topk, largest=True).indices
            
            # 更新所选记忆槽
            for idx in topk_indices:
                # 增加使用计数
                self.slot_usage_counter[idx] += 1
                
                # 获取当前记忆槽
                current_memory = self.memory[idx].unsqueeze(0)  # [1, slot_len, d_model]
                
                # 准备注意力掩码
                if not batch_mask.all():
                    attn_mask = batch_mask.float()
                    attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
                    attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, self.slot_len, -1)
                    attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
                else:
                    attn_mask = None
                
                # 使用交叉注意力机制更新记忆
                updated_memory, _ = self.memory_update_attention(
                    query=current_memory,        # [1, slot_len, d_model]
                    key=batch_features,          # [1, seq_length, d_model]
                    value=batch_features,        # [1, seq_length, d_model]
                    attn_mask=attn_mask          # [num_heads, slot_len, seq_length] or None
                )
                
                # 使用门控机制控制更新比例
                concat_memory = torch.cat([current_memory, updated_memory], dim=-1)  # [1, slot_len, 2*d_model]
                update_gate = torch.sigmoid(self.update_gate(concat_memory))  # [1, slot_len, d_model]
                
                # 应用门控更新
                new_memory = current_memory * (1 - update_gate) + updated_memory * update_gate
                
                # 更新记忆槽
                self.memory.data[idx] = new_memory.squeeze(0)

        # 打印相似度最大值和平均值
        print("Similarity max:", similarities.max())
        print("Similarity mean:", similarities.mean())

        # 检查是否存在NaN值
        if torch.isnan(self.memory).any():
            print("Warning: Memory contains NaN after update")

    def update_memory_normal_guiyi(self, features, mask=None, short_start=None, vid=None):
        """
        基于输入特征序列更新记忆槽,将相似度归一化后，根据相似度阈值更新
        使用注意力机制融合信息到相似度高的记忆槽
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
            short_start: 
            vid: 这两个暂时用不到了，先留着
        """
        self.updateCnt += 1
        batch_size, seq_length, _ = features.shape
        device = features.device
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 计算输入特征与记忆槽的相似度
        similarities = self.compute_sequence_similarity_frobenius(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_hybrid_pool(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_multi_scale(features, mask)  # [batch_size, memory_slots]

        # 统计每个样本输入，多少个记忆槽进行了更新
        update_per_sample = torch.zeros(batch_size, device=device, dtype=torch.int32)

        # 逐批次处理更新
        for b in range(batch_size):
            # 检查当前批次是否有有效特征
            if not mask[b].any():
                continue
            
            # 获取当前批次的特征和掩码
            batch_features = features[b].unsqueeze(0)  # [1, seq_length, d_model]
            batch_mask = mask[b].unsqueeze(0)  # [1, seq_length]
            
            # 找出相似度超过阈值的记忆槽
            similarity = similarities[b]  # [memory_slots]
            # 相似度归一化到0-1区间
            similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
            
            # # 动态相似度阈值
            # sim_thd = self.slot_usage_counter*self.similarity_threshold
            # sim_thd = sim_thd.clamp(max=self.similarity_threshold*10)
            # update_mask = (similarity > sim_thd)

            # 固定相似度阈值
            update_mask = (similarity > self.similarity_threshold)
            
            if not update_mask.any():
                continue
            
            # 获取需要更新的记忆槽索引
            update_indices = torch.where(update_mask)[0]
            update_per_sample[b] = len(update_indices)
            
            # 更新所选记忆槽
            for idx in update_indices:
                # 增加使用计数
                self.slot_usage_counter[idx] += 1
                
                # 获取当前记忆槽
                current_memory = self.memory[idx].unsqueeze(0)  # [1, slot_len, d_model]
                
                # 准备注意力掩码
                if not batch_mask.all():
                    attn_mask = batch_mask.float()
                    attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
                    attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, self.slot_len, -1)
                    attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
                else:
                    attn_mask = None
                
                # 使用交叉注意力机制更新记忆
                updated_memory, _ = self.memory_update_attention(
                    query=current_memory,        # [1, slot_len, d_model]
                    key=batch_features,          # [1, seq_length, d_model]
                    value=batch_features,        # [1, seq_length, d_model]
                    attn_mask=attn_mask          # [num_heads, slot_len, seq_length] or None
                )
                
                # 使用门控机制控制更新比例
                concat_memory = torch.cat([current_memory, updated_memory], dim=-1)  # [1, slot_len, 2*d_model]
                update_gate = torch.sigmoid(self.update_gate(concat_memory))  # [1, slot_len, d_model]
                
                # 应用门控更新
                new_memory = current_memory * (1 - update_gate) + updated_memory * update_gate
                
                # 更新记忆槽
                self.memory.data[idx] = new_memory.squeeze(0)
        
        # 如果不是全0，打印更新的最大记忆槽数量，最小记忆槽数量，平均记忆槽数量
        if update_per_sample.sum() > 0:
            print("batch Update max:", update_per_sample.max())
            print("batch Update min:", update_per_sample.min())
            print("batch Update mean:", update_per_sample.mean())

        # 打印相似度最大值和平均值
        print("Similarity max:", similarities.max())
        print("Similarity mean:", similarities.mean())

        # 检查是否存在NaN值
        if torch.isnan(self.memory).any():
            print("Warning: Memory contains NaN after update")

    def update_memory_normal_gtframes(self, features, gt_frames_in_long_memory, mask=None, short_start=None, vid=None):
        """
        根据历史特征中和gt相交的帧数量/分类器输出决定是否更新
        更新的记忆槽仍基于相似度（topk）
        使用注意力机制融合信息到相似度高的记忆槽
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            gt_frames_in_long_memory: 历史帧与GT窗口相交的帧数量，形状为[batch_size]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
            short_start: 
            vid: 这两个暂时用不到了，先留着
        """
        self.updateCnt += 1
        batch_size, seq_length, _ = features.shape
        device = features.device
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 计算输入特征与记忆槽的相似度
        similarities = self.compute_sequence_similarity_frobenius(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_hybrid_pool(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_multi_scale(features, mask)  # [batch_size, memory_slots]

        # 统计每个样本输入，多少个记忆槽进行了更新
        update_per_sample = torch.zeros(batch_size, device=device, dtype=torch.int32)

        # 逐批次处理更新
        for b in range(batch_size):
            # 检查当前样本是否有有效特征
            if not mask[b].any():
                continue
                
            # 判断当前样本是否需要更新,如果当前样本的相交帧数量小于7（中位数），则不更新
            if gt_frames_in_long_memory[b] < 7:
                continue
            
            # 获取当前样本的特征和掩码
            batch_features = features[b].unsqueeze(0)  # [1, seq_length, d_model]
            batch_mask = mask[b].unsqueeze(0)  # [1, seq_length]
            
            # 找出相似度最高的k个记忆槽
            similarity = similarities[b]  # [memory_slots]
            # 找出相似度最高的k个记忆槽
            topk_indices = torch.topk(similarity, k=topk, largest=True).indices
            
            # 更新所选记忆槽
            for idx in topk_indices:
                # 增加使用计数
                self.slot_usage_counter[idx] += 1
                
                # 获取当前记忆槽
                current_memory = self.memory[idx].unsqueeze(0)  # [1, slot_len, d_model]
                
                # 准备注意力掩码
                if not batch_mask.all():
                    attn_mask = batch_mask.float()
                    attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
                    attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, self.slot_len, -1)
                    attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
                else:
                    attn_mask = None
                
                # 使用交叉注意力机制更新记忆
                updated_memory, _ = self.memory_update_attention(
                    query=current_memory,        # [1, slot_len, d_model]
                    key=batch_features,          # [1, seq_length, d_model]
                    value=batch_features,        # [1, seq_length, d_model]
                    attn_mask=attn_mask          # [num_heads, slot_len, seq_length] or None
                )
                
                # 使用门控机制控制更新比例
                concat_memory = torch.cat([current_memory, updated_memory], dim=-1)  # [1, slot_len, 2*d_model]
                update_gate = torch.sigmoid(self.update_gate(concat_memory))  # [1, slot_len, d_model]
                
                # 应用门控更新
                new_memory = current_memory * (1 - update_gate) + updated_memory * update_gate
                
                # 更新记忆槽
                self.memory.data[idx] = new_memory.squeeze(0)

        # 打印相似度最大值和平均值
        print("Similarity max:", similarities.max())
        print("Similarity mean:", similarities.mean())

        # 检查是否存在NaN值
        if torch.isnan(self.memory).any():
            print("Warning: Memory contains NaN after update")


    def read_memory(self, query_features, mask=None):
        """
        从所有记忆槽中读取信息并平均结果
        
        Args:
            query_features: 查询特征，形状为 [batch_size, seq_length, d_model]
            mask: 查询特征的有效性掩码，形状为 [batch_size, seq_length]
            
        Returns:
            memory_output: 从记忆中读取的输出，形状为 [batch_size, seq_length, d_model]
        """
        batch_size, seq_length, _ = query_features.shape
        device = query_features.device
        
        # 创建一个存储所有槽注意力结果的张量
        all_slot_outputs = torch.zeros(batch_size, seq_length, self.d_model, device=device)
        
        # 处理掩码
        if mask is None:
            attn_mask = None
        else:
            if mask.dtype == torch.bool:
                mask_float = mask.float()
            else:
                mask_float = mask.clamp(0, 1)
            
            attn_mask = mask_float.unsqueeze(-1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        
        # 遍历每个记忆槽
        for i in range(self.memory_slots):
            current_slot = self.memory[i].unsqueeze(0).expand(batch_size, -1, -1)
            
            if attn_mask is not None:
                current_mask = attn_mask.expand(-1, -1, self.slot_len)
                current_mask = current_mask.repeat_interleave(self.num_heads, dim=0)
            else:
                current_mask = None
            
            # 执行交叉注意力
            slot_output, _ = self.memory_read_attention(
                query=query_features,           # [batch_size, seq_length, d_model]
                key=current_slot,               # [batch_size, slot_len, d_model]
                value=current_slot,             # [batch_size, slot_len, d_model]
                attn_mask=current_mask          # [B*num_heads, seq_length, slot_len] or None
            )
            
            all_slot_outputs += slot_output
        
        # 计算平均值
        memory_output = all_slot_outputs / self.memory_slots
        
        return memory_output

    def predict_future(self, history_features, current_features, mask=None):
        """
        基于历史特征、当前特征和记忆预测未来特征
        
        Args:
            history_features: 历史特征，形状为 [batch_size, history_length, d_model]
            current_features: 当前特征，形状为 [batch_size, current_length, d_model]
            mask: 历史特征的有效性掩码，形状为 [batch_size, history_length]
            
        Returns:
            future_features: 预测的未来特征，形状为 [batch_size, future_length, d_model]
            future_mask: 未来特征的掩码，形状为 [batch_size, future_length]
        """
        batch_size = history_features.shape[0]
        device = history_features.device
        
        # 扩展未来嵌入以匹配批次大小
        future_embed = self.future_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接历史特征、当前特征和未来嵌入
        concatenated_features = torch.cat([history_features, current_features, future_embed], dim=1)

        # 拼接特征的mask
        if mask is not None:
            current_mask = torch.ones(batch_size, current_features.shape[1], 
                                    device=device, dtype=mask.dtype)
            future_mask = torch.ones(batch_size, future_embed.shape[1], 
                                   device=device, dtype=mask.dtype)
            concat_mask = torch.cat([mask, current_mask, future_mask], dim=1)
        else:
            concat_mask = None

        # 从记忆中读取相关信息
        memory_output = self.read_memory(concatenated_features, concat_mask)

        # 用读取了记忆信息的结果做self-attention
        memory_output = self.memory_self_attention(memory_output, memory_output, memory_output)[0]
        
        # 只提取增强特征中的未来部分
        future_part = memory_output[:, -self.future_length:, :]
        
        # 返回预测的未来特征和对应的掩码
        future_mask = torch.ones(batch_size, self.future_length, 
                                device=device, dtype=torch.bool if mask is None 
                                else mask.dtype)
        
        return future_part, future_mask

    def forward(self, history_features, current_features, mask=None, short_start=None, vid=None):
        """
        前向传播：更新记忆槽并预测未来特征
        每经过optimization_interval次调用后，自动执行记忆多样性优化
        
        Args:
            history_features: 历史特征，形状为 [batch_size, history_length, d_model]
            current_features: 当前特征，形状为 [batch_size, current_length, d_model]
            mask: 历史特征的有效性掩码，形状为 [batch_size, history_length]
            short_start: 
            vid: 保留参数
            
        Returns:
            future_features: 预测的未来特征，形状为 [batch_size, future_length, d_model]
            future_mask: 未来特征的掩码，形状为 [batch_size, future_length]
        """
        # 1. 更新记忆槽
        # self.update_memory_normal(features=history_features, mask=mask)
        # topk 更新
        self.update_memory_normal_topk(features=history_features, mask=mask, topk=4)
        # 相似度归一化后根据阈值更新
        # self.update_memory_normal_guiyi(features=history_features, mask=mask)
        
        # 2. 预测未来特征
        future_features, future_mask = self.predict_future(history_features, current_features, mask)
        
        # # 3. 增加forward计数器并检查是否需要执行多样性优化
        # self.forward_counter += 1
        # if (self.forward_counter % self.optimization_interval == 0):
        #     with open("interMemory_Optimization.log", 'a') as f:
        #         f.write(f"\n--- Performing memory diversity optimization (step {self.forward_counter}) ---\n")
        #     print(f"\n--- Performing memory diversity optimization (step {self.forward_counter}) ---")
            
        #     self.optimize_memory_diversity()
        
        return future_features, future_mask

    def set_auto_optimization(self, diversity_loss_weight=None, 
                            optimization_interval=None, optimizer_lr=None):
        """
        动态设置自动优化参数
        
        Args:
            diversity_loss_weight: 多样性损失权重
            optimization_interval: 优化间隔
            optimizer_lr: 优化器学习率
        """
        
        if diversity_loss_weight is not None:
            self.diversity_loss_weight = diversity_loss_weight
            
        if optimization_interval is not None:
            self.optimization_interval = optimization_interval
            
        if optimizer_lr is not None:
            # 重新创建优化器
            self.memory_optimizer = torch.optim.Adam([self.memory], lr=optimizer_lr)
            
        print(f"Auto optimization settings updated: "
              f"loss_weight={self.diversity_loss_weight}, "
              f"interval={self.optimization_interval}")

    # 使用示例和说明
    """
    使用方法：

    1. 基本初始化：
    memory_module = InterMemorySeq(
        d_model=256,
        future_length=32,
        memory_slots=64,
        slot_len=64,
        diversity_loss_weight=0.5,    # 多样性损失权重
        optimization_interval=10,     # 每10次forward后优化一次
    )

    2. 在训练循环中自动优化：
    for batch in dataloader:
        history_features, current_features, mask = batch
        
        # forward会自动更新记忆并进行周期性优化
        future_pred, future_mask = memory_module(history_features, current_features, mask)
        
        # 使用预测结果计算主任务损失
        main_loss = criterion(future_pred, future_target)
        main_loss.backward()
        optimizer.step()

    3. 动态调整优化参数：
    memory_module.set_auto_optimization(
        diversity_loss_weight=0.2,   # 降低多样性损失权重
        optimization_interval=20,     # 增加优化间隔
        optimizer_lr=5e-5            # 调整优化器学习率
    )
    """



# 使用示例
def demo1():
    # 设置参数
    batch_size = 4
    d_model = 256
    future_length = 32
    
    # 创建模型实例
    memory = InterMemory(
        d_model=d_model,
        future_length=future_length
    )
    
    history_len = 64
    current_len = 8
    # 创建模拟输入
    features = torch.randn(batch_size, history_len, d_model)
    current_features = torch.randn(batch_size, current_len, d_model)
    # 为历史特征模拟一个mask，随机值
    mask = torch.randint(0, 2, (batch_size, history_len)).bool()
    
    # 使用模式1：更新记忆
    memory.update_memory(features, mask)
    print("记忆已更新")
    
    # 使用模式2：预测未来特征
    future_features = memory.predict_future(features, current_features, mask)
    
    # 打印输出形状
    print(f"输入历史特征形状: {features.shape}")
    print(f"输入当前特征形状: {current_features.shape}")
    print(f"预测未来特征形状: {future_features[0].shape}")

def demo2():
    # 设置参数
    batch_size = 4
    d_model = 256
    future_length = 32
    slot_len = 64
    
    # 创建模型实例
    memory = InterMemorySeq(
        d_model=d_model,
        future_length=future_length,
        slot_len=slot_len,
        pooling_type='mean'  # 使用平均池化
    )
    
    history_len = 64
    current_len = 8
    # 创建模拟输入
    features = torch.randn(batch_size, history_len, d_model)
    current_features = torch.randn(batch_size, current_len, d_model)
    # 为历史特征模拟一个mask
    mask = torch.randint(0, 2, (batch_size, history_len)).bool()
    
    # 使用模式1：更新记忆
    memory.update_memory(features, mask)
    print("记忆已更新")
    
    # 使用模式2：预测未来特征
    future_features, future_mask = memory.predict_future(features, current_features, mask)
    
    # 打印输出形状
    print(f"输入历史特征形状: {features.shape}")
    print(f"输入当前特征形状: {current_features.shape}")
    print(f"预测未来特征形状: {future_features.shape}")
    print(f"预测未来掩码形状: {future_mask.shape}")
    print(f"记忆槽使用计数: {memory.slot_usage_counter}")

if __name__ == "__main__":
    # demo1()
    demo2()