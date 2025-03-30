import json
from collections import defaultdict
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="Analyze dataset statistics.")
parser.add_argument('--dataset', type=str, default='tacos', choices=['tacos', 'activitynet', 'qvhighlight'],
                    help='Dataset name (tacos, activitynet, qvhighlight). Default is tacos.')
parser.add_argument('--split', type=str, default='val', choices=['val', 'train', 'test', 'new_val', 'new_train', 'new_test'],
                    help='Dataset split (val, train, test). Default is val.')
args = parser.parse_args()

# 根据输入参数构建文件路径
if args.dataset == 'qvhighlight':
    file_path = f'/home/gfj/lighthouse-main/data/{args.dataset}/highlight_{args.split}_release.jsonl'
else:
    file_path = f'/home/gfj/lighthouse-main/data/{args.dataset}/{args.dataset}_{args.split}_release.jsonl'

# 初始化统计变量
total_queries = 0
video_durations = {}  # 存储每个视频的时长（去重）
video_query_count = defaultdict(int)  # 存储每个视频的查询数量
total_rel_win_length = 0  # 所有相关窗口的总长度

# 读取文件
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        vid = data['vid']
        duration = data['duration']
        relevant_windows = data['relevant_windows']

        # 记录视频时长（去重）
        if vid not in video_durations:
            video_durations[vid] = duration

        # 更新视频的查询数量
        video_query_count[vid] += 1
        total_queries += 1

        # 计算所有相关窗口的总长度
        for window in relevant_windows:
            start, end = window
            total_rel_win_length += (end - start)

# 计算统计信息
total_videos = len(video_durations)
total_duration = sum(video_durations.values())
average_query_per_video = total_queries / total_videos
average_video_duration = total_duration / total_videos
average_rel_win_length = total_rel_win_length / total_queries

# 输出结果
print(f"Dataset: {args.dataset}, Split: {args.split}")
print(f"Total number of videos: {total_videos}")
print(f"Total number of queries: {total_queries}")
print(f"Average number of queries per video: {average_query_per_video:.2f}")
print(f"Average video duration: {average_video_duration:.2f} seconds")
print(f"Average relevant window length: {average_rel_win_length:.2f} seconds")