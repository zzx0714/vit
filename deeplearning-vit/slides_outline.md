# Five-Minute PPT Outline

Topic: **ConvStem-ViT: Improving Vision Transformer Patch Embedding for Small-Scale Image Classification**

## Slide 1: Background and Original ViT

- Paper: *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
- Key idea: split image into patches and treat them as visual tokens
- Main components:
  - Patch embedding
  - Position embedding
  - Class token
  - Transformer encoder
- Message: ViT shows that image recognition can be done with a mostly pure Transformer architecture.

## Slide 2: Motivation for Improvement

- Original ViT uses direct linear patch projection.
- This design has limited local inductive bias.
- On small datasets such as CIFAR-10, learning local patterns from scratch can be harder.
- Improvement idea: add a lightweight CNN stem before Transformer encoding.

## Slide 3: Proposed Method

Vanilla ViT:

```text
Image -> Patchify / Linear Projection -> Transformer Encoder -> Classifier
```

ConvStem-ViT:

```text
Image -> Conv 3x3 -> BN/GELU -> Conv 3x3 -> BN/GELU -> Tokens -> Transformer Encoder -> Classifier
```

Controlled comparison:

- Same image size
- Same patch size
- Same Transformer depth
- Same optimizer and training schedule
- Only patch embedding is changed

## Slide 4: Experimental Setup and Results

Dataset:

- CIFAR-10
- 50,000 train images
- 10,000 test images
- 10 classes

Models:

| Method | Patch Embedding | Accuracy |
|---|---|---:|
| ResNet18 | CNN | TBD |
| Vanilla ViT | Linear | TBD |
| ConvStem-ViT | Conv stem | TBD |

Add a training curve from:

```text
results/training_curves.png
```

## Slide 5: Summary

- Original ViT innovation: image patches as tokens for Transformer encoders.
- Proposed improvement: replace linear patch embedding with a CNN stem.
- Expected benefit: stronger local feature extraction and better small-data behavior.
- Deliverables:
  - Runnable PyTorch code
  - CIFAR-10 reproducible setup
  - AAAI-style English report draft
  - Experiment scripts and result templates
