import json
import random
from collections import defaultdict

# 定义文件路径
file_paths = {
    'test': '/home/gfj/lighthouse-main/data/tacos/tacos_test_release.jsonl',
    'train': '/home/gfj/lighthouse-main/data/tacos/tacos_train_release.jsonl',
    'val': '/home/gfj/lighthouse-main/data/tacos/tacos_val_release.jsonl'
}

# 读取文件并记录每个 vid 对应的 lines
vid_to_lines = defaultdict(list)
split_to_vids = defaultdict(set)

# 读取原始文件，记录每个 vid 的 lines 以及每个 split 的 vid 集合
for split, path in file_paths.items():
    with open(path, 'r') as file:
        for line in file:
            data = json.loads(line)
            vid = data['vid']
            vid_to_lines[vid].append(line)
            split_to_vids[split].add(vid)
    print(f"{split} has {len(split_to_vids[split])} unique vids and {sum(len(vid_to_lines[vid]) for vid in split_to_vids[split])} lines")

# 获取所有 vid 并打乱顺序
all_vids = list(vid_to_lines.keys())
random.shuffle(all_vids)

# 计算每个新 split 的目标 vid 数量
total_vids = len(all_vids)
target_test_vids = len(split_to_vids['test'])
target_train_vids = len(split_to_vids['train'])
target_val_vids = len(split_to_vids['val'])

# 分配 vid 到新的 split
new_split_to_vids = {
    'new_test': set(all_vids[:target_test_vids]),
    'new_train': set(all_vids[target_test_vids:target_test_vids + target_train_vids]),
    'new_val': set(all_vids[target_test_vids + target_train_vids:target_test_vids + target_train_vids + target_val_vids])
}

# 写入新的文件
new_file_paths = {
    'new_test': '/home/gfj/lighthouse-main/data/tacos/tacos_new_test_release.jsonl',
    'new_train': '/home/gfj/lighthouse-main/data/tacos/tacos_new_train_release.jsonl',
    'new_val': '/home/gfj/lighthouse-main/data/tacos/tacos_new_val_release.jsonl'
}

for new_split, new_path in new_file_paths.items():
    with open(new_path, 'w') as file:
        for vid in new_split_to_vids[new_split]:
            file.writelines(vid_to_lines[vid])
    num_vids = len(new_split_to_vids[new_split])
    num_lines = sum(len(vid_to_lines[vid]) for vid in new_split_to_vids[new_split])
    print(f"{new_split} has {num_vids} unique vids and {num_lines} lines")

print("Dataset has been successfully reshuffled by vid and saved to new files.")