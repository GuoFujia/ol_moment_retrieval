import torch

# 加载 .ckpt 文件
checkpoint = torch.load("results/ol_moment_detr/qvhighlight/clip_slowfast/best.ckpt", map_location="cpu")

# 打印文件内容
print(checkpoint.keys())
print(checkpoint["model"])