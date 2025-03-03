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

