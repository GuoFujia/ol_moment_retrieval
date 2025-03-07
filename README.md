1. 1.将预测部分的四分类改为四个二分类

2. 处理数据集部分存在的错误：修改了部分逻辑，样本的semantic标签和st，ed帧互斥

3. 修改评估部分逻辑，主要是eval_submission_ol_2函数，原来在单个样本内生成候选片段；修改后先根据“qid-vid”将样本分组，再在组内跨样本生成候选
       修改后性能也没有提高，但是在train的过程中，前几个epoch的R@5数值比较高，随后又降低

   ![image-20250303172735480](C:/Users/Guo jia/AppData/Roaming/Typora/typora-user-images/image-20250303172735480.png)

4. 另外，调试发现模型同时进行short_memory_sample_length帧的类别预测，这若干帧计算出的类别预测结果相互之间很接近，明显的没有在特殊帧出现类别logits的倾斜。如下图，第五帧为ed帧，那么理论上第五帧的ed logits值应该明显相比其他临近帧高，但是事实上并没有出现这种现象。

   而且但就logits的数值而言，也有点太平均了（接近0.25）……

   ![image-20250303224310188](C:/Users/Guo jia/AppData/Roaming/Typora/typora-user-images/image-20250303224310188.png)

   模型在encoder中通过self-attention处理了包含long_memory和short_memory的所有输入信息，然后在decoder中，通过设置num_queries等于short_memory长度，**希望使**decoder专注于生成short_memory部分的表示。但是事实上，只是令num_queries等于short_memory长度，得到的表示只是长度与short_memory保持一致，事实上还是包含了long_memory和short_memory两部分的信息。就导致在后面的映射部分利用这些“污染”了的信息（包含了多余的long_memory的信息）映射short_memory的类别，这样可能是导致错误的原因。

   因此修改预测部分的逻辑，预测short_memory和long_memory，但是在输出时只输出short_memory部分的结果

   修改后没有发现性能明显变化……

5. 在frame预测部分引入高斯软标签。同时修改了数据加载，以支持样本中包含多个gt_window的情况

   修改后多个性能指标明显提高（见下表第一行）

6. 纠正了上述修改导致的loss标签不一致问题（会导致跳过saliency loss），并对比了loss权重（见下表后两行）

| R@1,  IoU=0.5 | [R@1,   IoU=0.7 ](mailto:R@1, IoU=0.7) | [R@5,   IoU=0.5 ](mailto:R@5, IoU=0.5) | [R@5,   IoU=0.7](mailto:R@5, IoU=0.7) | saliency_mse | saliency_mae |                          备注                          |
| ------------- | -------------------------------------- | -------------------------------------- | ------------------------------------- | ------------ | ------------ | :----------------------------------------------------: |
| 2.4           | 1.44                                   | 27.4                                   | 8.65                                  | 2.64         | 1.1          |                    帧预测添加软标签                    |
| 5.29          | 1.44                                   | 25.96                                  | 8.65                                  | 2.4          | 1.13         | 修改了saliency部分部分权重标签命名不一致导致的梯度消失 |
| 8.17          | 1.44                                   | 23.56                                  | 7.21                                  | 2.16         | 1.22         |          修改了loss权重label：10，saliency：5          |
