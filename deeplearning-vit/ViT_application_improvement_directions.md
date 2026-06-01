# ViT 应用或改进方向评估

论文：*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*

## 任务要求评估

汇报需要包括：

- 原论文的关键创新点介绍
- 对原论文进行应用或者改进的原理介绍
- 对原论文进行应用或者改进的实验结果
- 应用或者改进的总结

交付清单：

- 可运行的代码库
- 可复现实验结果的数据集
- README 说明
- 使用 Latex AAAI 模板撰写的英文大作业报告
- 报告章节参考 AAAI 历年已录用论文

综合这些要求，选题最好满足以下条件：

- 数据集公开、下载方便、训练成本可控
- 代码实现难度适中，能在短时间内跑出结果
- 实验结果可以用表格、曲线或可视化清楚展示
- 能自然对应 ViT 原论文的核心创新点
- 五分钟 PPT 中有清晰叙事：原论文方法、问题或应用场景、我们的做法、实验结果、总结

## 方向一：ViT 在小规模数据集上的迁移学习应用

推荐指数：高。  
特点：最稳，最容易复现。

### 题目示例

**Fine-Tuning Vision Transformer for Small-Scale Image Classification**

### 核心思路

使用原论文 ViT 的核心结构：

- Patch embedding
- Transformer encoder
- Class token
- Position embedding

加载 ImageNet 预训练 ViT 模型，在小规模图像分类数据集上进行微调，例如：

- CIFAR-10
- CIFAR-100
- Flowers102
- Food101 子集

这个方向主要体现 ViT 的应用价值：虽然原论文强调大规模预训练，但通过迁移学习，ViT 也可以应用到较小规模的下游图像分类任务中。

### 可做实验

| 实验类型 | 内容 |
|---|---|
| Baseline | ResNet18 或简单 CNN |
| ViT 方法 | ViT-B/16、ViT-Tiny 或 ViT-Small 微调 |
| 冻结策略 | 冻结 backbone vs 全量微调 |
| 数据增强 | 无增强 vs RandAugment / Mixup / CutMix |

### 可展示结果

| 指标 | 说明 |
|---|---|
| Accuracy | 分类准确率 |
| Training curve | 训练和验证曲线 |
| Confusion matrix | 类别混淆情况 |
| Error cases | 错误分类样本分析 |

### 优点

- 实现难度低
- 训练流程清楚
- 可复现性强
- 适合快速完成代码库和 README
- 报告结构容易展开

### 风险

- 创新性偏弱，更像应用实验
- 如果只做微调，容易显得工作量不足

### 适合的报告结构

1. Introduction
2. Related Work
3. Method: ViT Fine-Tuning
4. Experiments
5. Results and Analysis
6. Conclusion

## 方向二：改进 ViT 的 Patch Embedding，加入 CNN Stem

推荐指数：最高。  
特点：兼顾改进性、可实现性和汇报完整度。

### 题目示例

**Improving Vision Transformers on Small Datasets with Convolutional Patch Embedding**

### 核心思路

原始 ViT 直接将图像划分为固定大小的 patch，例如 16x16 patch，然后通过线性投影得到 token：

```text
Image -> Patchify -> Linear Projection -> Transformer Encoder
```

这种方式简洁，但缺少 CNN 的局部归纳偏置。在小规模数据集上，直接 patchify 可能不能充分利用图像中的局部纹理、边缘和空间邻近关系。

改进方法是将原来的线性 patch embedding 替换为轻量 CNN stem：

```text
Image -> Conv 3x3 -> BN/ReLU -> Conv 3x3 stride -> Patch Tokens -> Transformer Encoder
```

CNN stem 先提取局部视觉特征，再将特征转换成 token 送入 Transformer。这样可以结合：

- CNN 的局部建模能力
- Transformer 的全局关系建模能力

### 可做实验

| 模型 | Patch Embedding | 数据集 |
|---|---|---|
| Vanilla ViT | Linear patch embedding | CIFAR-10 |
| ConvStem-ViT | CNN stem patch embedding | CIFAR-10 |
| ResNet18 | CNN baseline | CIFAR-10 |

### 可展示结果

| 指标 | 说明 |
|---|---|
| Accuracy | 分类准确率 |
| Training curve | 收敛速度和稳定性 |
| Params | 参数量对比 |
| FLOPs | 计算量对比，可选 |
| Attention map | 注意力区域可视化，可选 |

### 优点

- 有明确的“原论文不足 -> 改进方法 -> 实验验证”逻辑
- 工作量适中
- 改动点清晰，适合讲原理
- 实验结果可以通过准确率、训练曲线、参数量等方式展示
- 比单纯迁移学习更有改进性质

### 风险

- 需要修改 ViT 模型结构
- 如果训练轮数太少，提升可能不明显
- 需要控制变量，保证 Vanilla ViT 和 ConvStem-ViT 的训练设置一致

### 推荐原因

这是最推荐的方向。它最符合课程汇报要求，既能介绍原论文 ViT 的关键创新，也能围绕 patch embedding 做一个清楚、可实现、可验证的小改进。

## 方向三：ViT 注意力可视化与可解释性应用

推荐指数：中高。  
特点：展示效果最好，适合 PPT。

### 题目示例

**Interpreting Vision Transformer Decisions via Attention Rollout**

### 核心思路

ViT 的核心机制是 self-attention。可以利用 attention map 或 attention rollout 方法，将模型在分类时关注的图像区域可视化出来，从而分析 ViT 的决策依据。

基本流程：

```text
Input Image -> ViT -> Attention Matrices -> Attention Rollout -> Heatmap Visualization
```

可以进一步结合遮挡实验进行验证：

- 遮挡高注意力区域，观察分类置信度是否明显下降
- 遮挡随机区域，观察分类置信度变化
- 比较两者差异，验证注意力区域是否具有解释意义

### 可做实验

| 实验类型 | 内容 |
|---|---|
| 分类实验 | 微调或直接使用预训练 ViT |
| 可视化实验 | 展示 attention map / attention rollout |
| 错例分析 | 对错误分类样本进行注意力分析 |
| 遮挡实验 | 高注意力区域遮挡 vs 随机遮挡 |

### 可展示结果

| 指标或图像 | 说明 |
|---|---|
| Accuracy | 分类准确率 |
| Attention heatmap | 模型关注区域 |
| Confidence drop | 遮挡前后的置信度变化 |
| Case study | 正确样本和错误样本对比 |

### 优点

- PPT 展示效果强
- 有大量图片和热力图可展示
- 可以体现 ViT self-attention 的机制特点
- 实验代码量适中

### 风险

- 更偏应用分析，改进性较弱
- 可解释性结论需要谨慎，不能简单等同于因果解释
- 如果只做可视化，实验深度可能不足

## 三个方向对比

| 排名 | 方向 | 推荐理由 | 风险 |
|---|---|---|---|
| 1 | ConvStem-ViT 改进 Patch Embedding | 有明确改进点，实验好做，报告好写 | 需要修改模型结构 |
| 2 | ViT 迁移学习应用 | 最稳，代码和结果最容易复现 | 创新性偏弱 |
| 3 | ViT 注意力可解释性应用 | 展示效果强，PPT 好看 | 实验深度需要设计好 |

## 最终建议

如果目标是稳妥完成课程大作业并且体现一定改进性，建议选择：

**方向二：改进 ViT 的 Patch Embedding，加入 CNN Stem。**

推荐理由：

- 与 ViT 原论文的 patch embedding 机制直接相关
- 改进动机清楚：补充局部归纳偏置
- 实现难度适中
- 可以跑出定量实验结果
- 适合写成 AAAI 风格英文报告
- 五分钟 PPT 中叙事完整

一个可行的最终项目标题是：

**ConvStem-ViT: Improving Vision Transformer Patch Embedding for Small-Scale Image Classification**

