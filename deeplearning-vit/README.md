# ConvStem-ViT: 在 CIFAR-10 上改进 Vision Transformer 的 Patch Embedding

本仓库包含了一个基于论文 **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** 的课程项目实现。

该项目选择了对原始 ViT patch embedding 进行改进的方向：在将 tokens 送入 Transformer 编码器之前，通过一个轻量级的卷积干网络（convolutional stem）替换掉直接的线性 patch 投影。

## 简介：这个仓库是做什么的？

这是一个用于深度学习课程实验的完整代码库，主要用于在 CIFAR-10 数据集上对比三种不同的图像分类模型：
1. **ResNet18**（经典的 CNN 基线模型）
2. **Vanilla ViT**（使用原始线性投影的标准化 Vision Transformer）
3. **ConvStem-ViT**（本项目提出的改进版本，引入小型的 CNN Stem 作为 tokenizer）

这个仓库提供了从**数据加载 -> 模型构建 -> 训练脚本 -> 评估与可视化 -> 甚至 LaTeX 报告模板**的端到端完整流程。你只需要安装依赖并运行脚本，即可复现实验并得到用于报告的数据和图表。

## 为什么要做这个项目？（动机）

原版的 ViT 将图像分割成不重叠的 patches，并将每个展平的 patch 投影成一个 token。这种设计简单且易于扩展，但是它比卷积网络（CNN）拥有更弱的局部归纳偏置（local inductive bias），尤其是在训练数据集规模较小（如 CIFAR-10）时。

本项目旨在验证：在进行全局的自注意力机制（self-attention）计算之前，先引入一个小型的 CNN stem 来提取局部的边缘和纹理特征，是否能提升 ViT 在 CIFAR-10 上的表现。

## 核心实现方法

| 方法 | 描述 |
|---|---|
| ResNet18 | 专为 32x32 CIFAR 图像适配的 CNN 基线模型 |
| Vanilla ViT | 采用线性 Patch 投影（Patch Embedding）的小型 ViT |
| ConvStem-ViT | Backbone 与 Vanilla ViT 相同，但将 Patch Embedding 替换成了 CNN Stem |

对于 CIFAR-10，输入图像分辨率为 32x32，patch 大小设置为 4x4。这保证了图像和 patch 之间的比例与 ViT-B/16 在 224x224 图像上的比例相近，同时使得计算量适合作为一门课程的项目。

## 仓库结构

```text
.
|-- configs/                   # 模型及训练配置文件夹
|   |-- convstem_vit.yaml      # ConvStem-ViT 的超参数配置
|   |-- resnet18.yaml          # ResNet18 的超参数配置
|   `-- vanilla_vit.yaml       # Vanilla ViT 的超参数配置
|-- report/                    # 报告文件夹 (LaTeX稿)
|   |-- references.bib         # 参考文献库
|   `-- main.tex               # 报告的主体内容
|-- scripts/                   # 一键运行脚本
|   |-- dry_run.ps1            # 快跑测试（测试代码是否报错）
|   `-- run_all.ps1            # 依次运行三种模型的完整训练
|-- src/                       # 源代码目录
|   |-- data.py                # 数据集加载和预处理
|   |-- engine.py              # 模型训练和评估的核心逻辑
|   |-- models.py              # 三种网络模型的定义代码
|   `-- utils.py               # 日志、随机种子设置等辅助工具
|-- evaluate.py                # 加载权重进行单次性能评估的脚本
|-- plot_results.py            # 根据实验结果绘制 training curves 的脚本
|-- train.py                   # 启动单次训练的主函数
|-- requirements.txt           # Python 依赖库
`-- ViT_application_improvement_directions.md # 应用和改进方向的文档
```

## 我该怎么做？（分步指南）

### 1. 配置开发环境

首先，在你的终端（如 PowerShell 或 Cmd）里配置虚拟环境并安装所需的包：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

本项目依赖的库包括 PyTorch, torchvision, PyYAML, tqdm, matplotlib。

### 2. 测试代码是否可以成功跑通（Dry Run）

为了确保当前代码和依赖没有问题，可以运行下面这个脚本。它只会使用非常少量的数据快速跑一个 epoch（不具有实际训练效果，只是检查代码逻辑是否报错）：

```powershell
.\scripts\dry_run.ps1
```

### 3. 开始完整的实验训练

如果上一步顺利运行没有报错，你就可以开始真正的训练了（强烈建议在有 GPU 的机器上运行）：

你可以选择一键自动跑完所有的实验（推荐）：
```powershell
.\scripts\run_all.ps1
```

或者你想单独一个个运行，可以手动执行：
```powershell
python train.py --config configs/resnet18.yaml
python train.py --config configs/vanilla_vit.yaml
python train.py --config configs/convstem_vit.yaml
```
*(注：默认会自动下载 CIFAR-10 数据集到 `data/` 目录下。)*

### 4. 检查预期的输出

在上面的训练运行完毕后，项目会在 `outputs/` 目录下为每个模型生成以下文件：

```text
outputs/convstem_vit_cifar10/
|-- best.pt         # 验证集准确率最高的模型权重（Checkpoint）
|-- config.yaml     # 备份的配置文件，方便记录实验设置
|-- last.pt         # 最后一个 epoch 的模型权重
|-- metrics.csv     # 训练损失、测试准确率等指标的变化过程（供绘图使用）
`-- summary.json    # 最终结果概览（包含最佳准确率等信息）
```

### 5. 绘制结果对比图并评估性能

当这三个模型的训练都结束后，通过下方命令生成训练损失和精度变化的对比图：

```powershell
python plot_results.py --runs outputs/resnet18_cifar10 outputs/vanilla_vit_cifar10 outputs/convstem_vit_cifar10 --output results/training_curves.png
```
你可以在 `results/training_curves.png` 中查看并导出这张图，它可以直接放进你的实验报告里。

如果你想单独测试某个保存好权重的模型，可以用如下命令：
```powershell
python evaluate.py --checkpoint outputs/convstem_vit_cifar10/best.pt
```

## 实验设置简介

所有的模型默认都采用完全相同的训练协议以保持公平，你可以在 `configs/` 中的配置文件查看到：

- 数据集：CIFAR-10 (训练集 50,000 / 测试集 10,000)
- Epochs训练轮数：50
- Batch size (批大小)：128
- 优化器：AdamW 
- 学习率：3e-4 (配合 Cosine annealing 退火衰减调度)
- 数据增强：随机裁剪 + 随机水平翻转

**核心对比：`Vanilla ViT` 与 `ConvStem-ViT`**。这两个模型在 Transformer 编码器的深度、词嵌入维度、注意力头数等方面一致，唯一的区别就是“把图片切分成 token”这第一步所使用的嵌入模块设计（即线性投影对比卷积干网络）。

你可以根据跑完代码后生成的 `summary.json` 文件或者日志打印内容来填写下表，并用于你的期末报告中：

| 方法 | Patch Embedding | Best Test Accuracy (最高测试集准确率) | Params (参数量) | 备注 |
|---|---|---:|---:|---|
| ResNet18 | CNN | 待填 | 待填 | CNN baseline |
| Vanilla ViT | 线性投影 (Linear projection) | 待填 | 待填 | 原始 ViT 风格嵌入 |
| ConvStem-ViT | 卷积干网络 (ConvStem) | 待填 | 待填 | 提出的改进版本 |

## 完成你的报告

本仓库为你提供了一个标准的 AAAI 会议格式的 LaTeX 报告草稿模板，存放在 `report/main.tex` 中。

写作步骤建议：
1. 下载 AAAI 2024 的官方作者工具包 ([前往官网链接](https://aaai.org/authorkit24-2))。
2. 将工具包中的 `aaai24.sty` 和相关的 LaTeX 模板文件拷贝进 `report/` 目录。
3. 把你实验跑出的图表(`results/training_curves.png`) 导入到报告中。
4. 将参数列表、测得的具体准确率填入模板里的表格中。
5. 补充你对实验现象的分析后，编译导出最终的 PDF 即可。

---
最后提示：不要忘了在带 GPU 的环境中运行 `run_all.ps1`！等待跑完这 50 个 epochs 后，你就会得到用于书写报告所需要的所有实验数据。祝你取得优秀的成绩！
