from torch.utils.data import DataLoader
import torch
from onlineDataset import StartEndDataset,start_end_collate_ol, prepare_batch_inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def check_collate_fn(dataset):
    print("\nCollate函数检查:")
    # 创建测试批次
    test_batch = [dataset[i] for i in range(4)]
    
    # 应用collate函数
    metas, batched_inputs = start_end_collate_ol(test_batch)
    
    print("批次元数据数量:", len(metas))
    print("批次输入键:", batched_inputs.keys())
    
    # 检查填充情况
    print("视频特征形状:", batched_inputs['video_feat_short'].shape)
    print("查询特征形状:", batched_inputs['query_feat'].shape)
    print("标签形状:", batched_inputs['start_label'].shape)
    
    # 检查prepare_batch_inputs
    device = torch.device("cpu")
    model_inputs, targets = prepare_batch_inputs(metas, batched_inputs, device)
    
    print("\n准备批次输入后:")
    print("模型输入键:", model_inputs.keys())
    print("目标键:", targets.keys())

def check_feature_alignment(dataset):
    print("\n特征对齐检查:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        meta = sample['meta']
        inputs = sample['model_inputs']
        
        print(f"\n样本 {i}: {meta['vid']}")
        
        # 检查视频特征
        video_feat = dataset._get_video_feat_by_vid(meta['vid'])
        print(f"原始视频特征长度: {len(video_feat)}")
        print(f"短记忆特征长度: {inputs['video_feat_short'].shape[0]}")
        
        # 检查标签与特征对齐
        assert inputs['start_label'].shape[0] == inputs['video_feat_short'].shape[0], "标签与特征长度不匹配"
        
        # 检查GT窗口是否在特征范围内
        for gt_start, gt_end in meta['gt_windows']:
            assert gt_start < len(video_feat) and gt_end <= len(video_feat), "GT窗口超出视频范围"

def check_dataloader(dataset, batch_size=4):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: start_end_collate_ol(batch, long_memory_sample_length=dataset.long_memory_sample_length),
        shuffle=False,
        num_workers=0
    )
    
    print("\n数据加载器检查:")
    for batch_idx, (metas, batch_inputs) in enumerate(dataloader):
        print(f"\n批次 {batch_idx}:")
        print("元数据数量:", len(metas))
        print("输入键:", batch_inputs.keys())
        
        # 检查特征形状
        print("视频特征形状:", batch_inputs['video_feat_short'].shape)
        print("查询特征形状:", batch_inputs['query_feat'].shape)
        print("开始标签形状:", batch_inputs['start_label'].shape)
        
        # 只检查第一个批次
        if batch_idx == 0:
            break

def check_label_generation(dataset):
    print("\n标签生成检查:")
    flag = 1
    i = 0 
    # 统计每个样本的gt_frames_in_long_memory属性值
    num_gt_frames_in_long_memory = []
    for i in range(len(dataset.chunk_infos)):
        chunk_info = dataset.chunk_infos[i]
        num_gt_frames_in_long_memory.append(chunk_info['gt_frames_in_long_memory'])
    print(f"平均: {np.mean(num_gt_frames_in_long_memory):.1f}")
    print(f"最小: {min(num_gt_frames_in_long_memory)}")
    print(f"最大: {max(num_gt_frames_in_long_memory)}")
    # 计算中位数，上三分位数和上四分位数
    print(f"中位数: {np.median(num_gt_frames_in_long_memory)}")
    print(f"67%位数: {np.percentile(num_gt_frames_in_long_memory, 67)}")
    print(f"80%位数: {np.percentile(num_gt_frames_in_long_memory, 80)}")
    # 再打印分别等于0-max的样本数量
    for i in range(0, max(num_gt_frames_in_long_memory) + 1):
        print(f"等于 {i} 的样本数量: {sum(1 for x in num_gt_frames_in_long_memory if x == i)}")
    
    # while flag:
    #     chunk_info = dataset.chunk_infos[i]
    #     print(f"\n样本 {i}:")
    #     print(f"视频ID: {chunk_info['vid']}")
    #     print(f"查询: {chunk_info['query']}")
    #     print(f"GT窗口: {chunk_info['gt_windows']}")
    #     print(f"短记忆起始: {chunk_info['short_memory_start']}")
    #     print(f"开始标签形状: {chunk_info['start_label']}")
    #     print(f"中间标签形状: {chunk_info['middle_label']}")
    #     print(f"结束标签形状: {chunk_info['end_label']}")
    #     print(f"显著性分数正样本: {chunk_info['saliency_pos_labels']}")
    #     print(f"显著性分数负样本: {chunk_info['saliency_neg_labels']}")
    #     print(f"显著性分数数值: {chunk_info['saliency_all_labels']}")

    #     flag = input("是否继续检查下一个样本？(1/0): ")
    #     if flag == '1':
    #         i += 1
    #     else:
    #         break

def check_data_distribution(dataset):
    # 检查视频长度分布
    video_lengths = []
    for item in dataset.data:
        video_feat = dataset._get_video_feat_by_vid(item["vid"])
        video_lengths.append(len(video_feat))
    
    print("\n视频长度统计:")
    print(f"平均长度: {np.mean(video_lengths):.1f}")
    print(f"最小长度: {min(video_lengths)}")
    print(f"最大长度: {max(video_lengths)}")
    
    # 检查查询长度分布
    if not dataset.use_glove:
        query_lengths = []
        for item in dataset.data:
            query_feat = dataset._get_query_feat_by_qid(item["qid"])
            query_lengths.append(len(query_feat))
        
        print("\n查询长度统计:")
        print(f"平均长度: {np.mean(query_lengths):.1f}")
        print(f"最小长度: {min(query_lengths)}")
        print(f"最大长度: {max(query_lengths)}")

if __name__ == "__main__":

    dataset = 'qvhighlight'
    feature = 'clip_slowfast'

    # feature directory
    v_feat_dirs = None
    t_feat_dir = None
    t_feat_dir_pretrain_eval = None

    if feature == 'clip_slowfast':
        v_feat_dirs = [f'features/{dataset}/clip', f'features/{dataset}/slowfast']
        t_feat_dir = f'features/{dataset}/clip_text'

    elif feature == 'clip':
        v_feat_dirs = [f'features/{dataset}/clip']
        t_feat_dir = f'features/{dataset}/clip_text'

    elif feature == 'resnet_glove':
        v_feat_dirs = [f'features/{dataset}/resnet']
        t_feat_dir = f'features/{dataset}/glove'

    elif feature == 'i3d_clip':
        v_feat_dirs = [f'features/{dataset}/i3d']
        t_feat_dir = f'features/{dataset}/clip_text'

    # 将特征路径修改
    base_path = '/media/sda/szr/lighthouse'
    v_feat_dirs = [os.path.join(base_path, path) for path in v_feat_dirs]
    t_feat_dir = os.path.join(base_path, t_feat_dir)
  
    dataset = StartEndDataset(
        dset_name="qvhighlight",
        data_path="/home/gfj/lighthouse-main/data/qvhighlight/highlight_train_release.jsonl",
        # data_path="/home/gfj/lighthouse-main/data/qvhighlight/highlight_val_release.jsonl",
        # data_path="/home/gfj/lighthouse-main/data/tacos/tacos_val_release.jsonl",
        # data_path="/home/gfj/lighthouse-main/data/activitynet/activitynet_train_release.jsonl",
        v_feat_dirs=v_feat_dirs,
        q_feat_dir=t_feat_dir,
        short_memory_sample_length=8,
        long_memory_sample_length=64,
        future_memory_sample_length=0,
        test_mode=True,
        domain=None, 
        a_feat_dirs=None,
    )
    
    # 执行检查
    check_data_distribution(dataset)
    check_label_generation(dataset)
    check_feature_alignment(dataset)
    check_dataloader(dataset)
    check_collate_fn(dataset)
    
    print("\n所有检查完成!")