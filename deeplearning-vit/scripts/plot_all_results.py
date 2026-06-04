"""
一键生成全部实验可视化结果：
  图1  accuracy_curves.png       — 4 组实验 val accuracy 对比
  图2  loss_curves.png           — 4 组实验 loss 对比
  图3  convergence_speed.png     — 收敛速度对比（前 20 epoch 放大）
  图4  params_vs_accuracy.png    — 参数量 & 准确率双轴柱状图
  图5  mae_reconstruction_linear.png    — Linear MAE 重建可视化
  图6  mae_reconstruction_convstem.png  — ConvStem MAE 重建可视化
  图7  mae_error_heatmap.png     — 重建误差热力图
  表1  summary_table.csv         — 结果汇总
  表2  ablation_table.csv        — 2×2 消融表
"""
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 确保从项目根目录运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────
#  实验配置
# ──────────────────────────────────────────────

EXPERIMENTS = [
    {
        "name": "Vanilla ViT (scratch)",
        "dir": "outputs/vanilla_vit_cifar10",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    {
        "name": "ConvStem-ViT (scratch)",
        "dir": "outputs/convstem_vit_cifar10",
        "color": "#ff7f0e",
        "linestyle": "-",
    },
    {
        "name": "Vanilla ViT (MAE pretrained)",
        "dir": "outputs/finetune_vanilla_vit_cifar10",
        "color": "#1f77b4",
        "linestyle": "--",
    },
    {
        "name": "ConvStem-ViT (MAE pretrained)",
        "dir": "outputs/finetune_convstem_vit_cifar10",
        "color": "#ff7f0e",
        "linestyle": "--",
    },
]

MAE_EXPERIMENTS = [
    {
        "name": "Linear MAE",
        "config": "configs/mae_pretrain_tiny_imagenet.yaml",
        "checkpoint": "outputs/mae_pretrain_tiny_imagenet/best.pt",
        "output": "results/mae_reconstruction_linear.png",
    },
    {
        "name": "ConvStem MAE",
        "config": "configs/mae_pretrain_convstem_tiny_imagenet.yaml",
        "checkpoint": "outputs/mae_pretrain_convstem_tiny_imagenet/best.pt",
        "output": "results/mae_reconstruction_convstem.png",
    },
]

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(exp_dir):
    """加载 metrics.csv 为 dict 列表。"""
    csv_path = Path(exp_dir) / "metrics.csv"
    if not csv_path.exists():
        print(f"  ⚠ {csv_path} 不存在，跳过")
        return []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_summary(exp_dir):
    """加载 summary.json。"""
    json_path = Path(exp_dir) / "summary.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────
#  图 1: Accuracy 对比曲线
# ──────────────────────────────────────────────

def plot_accuracy_curves():
    print("生成 图1: accuracy_curves.png ...")
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp in EXPERIMENTS:
        metrics = load_metrics(exp["dir"])
        if not metrics:
            continue
        epochs = [int(m["epoch"]) for m in metrics]
        test_acc = [float(m["test_acc"]) for m in metrics]
        ax.plot(epochs, test_acc, label=exp["name"],
                color=exp["color"], linestyle=exp["linestyle"], linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("CIFAR-10 Test Accuracy Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_curves.png", dpi=150)
    plt.close()
    print("  ✓ 保存到 results/accuracy_curves.png")


# ──────────────────────────────────────────────
#  图 2: Loss 对比曲线
# ──────────────────────────────────────────────

def plot_loss_curves():
    print("生成 图2: loss_curves.png ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for exp in EXPERIMENTS:
        metrics = load_metrics(exp["dir"])
        if not metrics:
            continue
        epochs = [int(m["epoch"]) for m in metrics]
        train_loss = [float(m["train_loss"]) for m in metrics]
        test_loss = [float(m["test_loss"]) for m in metrics]
        ax1.plot(epochs, train_loss, label=exp["name"],
                 color=exp["color"], linestyle=exp["linestyle"], linewidth=1.5)
        ax2.plot(epochs, test_loss, label=exp["name"],
                 color=exp["color"], linestyle=exp["linestyle"], linewidth=1.5)

    ax1.set_title("Train Loss", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Test Loss", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
    plt.close()
    print("  ✓ 保存到 results/loss_curves.png")


# ──────────────────────────────────────────────
#  图 3: 收敛速度对比（前 20 epoch 放大）
# ──────────────────────────────────────────────

def plot_convergence_speed():
    print("生成 图3: convergence_speed.png ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    max_epochs = 20

    for exp in EXPERIMENTS:
        metrics = load_metrics(exp["dir"])
        if not metrics:
            continue
        epochs = [int(m["epoch"]) for m in metrics[:max_epochs]]
        test_acc = [float(m["test_acc"]) for m in metrics[:max_epochs]]
        ax.plot(epochs, test_acc, label=exp["name"],
                color=exp["color"], linestyle=exp["linestyle"], linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Convergence Speed (First 20 Epochs)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "convergence_speed.png", dpi=150)
    plt.close()
    print("  ✓ 保存到 results/convergence_speed.png")


# ──────────────────────────────────────────────
#  图 4: 参数量 & 准确率柱状图
# ──────────────────────────────────────────────

def plot_params_vs_accuracy():
    print("生成 图4: params_vs_accuracy.png ...")
    import torch
    from src.models import create_model
    import yaml

    names = []
    accuracies = []
    param_counts = []

    configs_for_params = [
        ("Vanilla ViT\n(scratch)", "configs/vanilla_vit.yaml", "outputs/vanilla_vit_cifar10"),
        ("ConvStem-ViT\n(scratch)", "configs/convstem_vit.yaml", "outputs/convstem_vit_cifar10"),
        ("Vanilla ViT\n(MAE pretrained)", "configs/finetune_vanilla_vit_cifar10.yaml", "outputs/finetune_vanilla_vit_cifar10"),
        ("ConvStem-ViT\n(MAE pretrained)", "configs/finetune_convstem_vit_cifar10.yaml", "outputs/finetune_convstem_vit_cifar10"),
    ]

    for label, config_path, output_dir in configs_for_params:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            model = create_model(cfg)
            n_params = sum(p.numel() for p in model.parameters())
        else:
            n_params = 0

        summary = load_summary(output_dir)
        acc = summary.get("best_test_acc", 0)

        names.append(label)
        accuracies.append(acc)
        param_counts.append(n_params / 1e6)  # 百万

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, accuracies, width, label="Test Accuracy", color="#2196F3", alpha=0.8)
    ax1.set_ylabel("Test Accuracy", fontsize=12, color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, param_counts, width, label="Parameters (M)", color="#FF9800", alpha=0.8)
    ax2.set_ylabel("Parameters (M)", fontsize=12, color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_title("Model Parameters vs Test Accuracy", fontsize=14)

    # 在柱子上标注数值
    for bar, val in zip(bars1, accuracies):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#2196F3")
    for bar, val in zip(bars2, param_counts):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.1f}M", ha="center", va="bottom", fontsize=9, color="#FF9800")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "params_vs_accuracy.png", dpi=150)
    plt.close()
    print("  ✓ 保存到 results/params_vs_accuracy.png")


# ──────────────────────────────────────────────
#  图 5 & 6: MAE 重建可视化
# ──────────────────────────────────────────────

def generate_mae_reconstructions():
    """为每个 MAE 实验生成重建可视化图。"""
    import torch
    import yaml
    from src.data import build_loaders, get_normalize_params
    from src.models_mae import MaskedAutoencoderViT

    for mae_exp in MAE_EXPERIMENTS:
        config_path = mae_exp["config"]
        ckpt_path = mae_exp["output_path"] = mae_exp["checkpoint"]
        output_path = RESULTS_DIR / Path(mae_exp["output"]).name

        if not Path(config_path).exists() or not Path(ckpt_path).exists():
            print(f"  ⚠ {mae_exp['name']}: config 或 checkpoint 不存在，跳过")
            continue

        print(f"生成 {mae_exp['name']} 重建可视化 ...")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg = config["model"]

        model = MaskedAutoencoderViT(
            img_size=config["data"]["image_size"],
            patch_size=model_cfg["patch_size"],
            in_chans=3,
            embed_dim=model_cfg["embed_dim"],
            depth=model_cfg["depth"],
            num_heads=model_cfg["num_heads"],
            decoder_embed_dim=model_cfg["decoder_embed_dim"],
            decoder_depth=model_cfg["decoder_depth"],
            decoder_num_heads=model_cfg["decoder_num_heads"],
            mlp_ratio=model_cfg["mlp_ratio"],
            patch_embed_type=model_cfg.get("patch_embed_type", "linear"),
        )
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model.to(device)
        model.eval()

        # 获取数据 — 多取一些图片用于丰富可视化
        config["data"]["train_limit"] = 64
        config["data"]["test_limit"] = 64
        _, test_loader = build_loaders(config)
        images, _ = next(iter(test_loader))
        images = images[:20].to(device)  # 取 20 张，分 5 页每页 4 张

        dataset_name = config["data"].get("dataset", "cifar10").lower()
        norm_mean, norm_std = get_normalize_params(dataset_name)

        mask_ratio = model_cfg.get("mask_ratio", 0.6)
        patch_size = model_cfg["patch_size"]
        image_size = config["data"]["image_size"]

        with torch.no_grad():
            loss, pred, mask = model(images, mask_ratio=mask_ratio)

            target = model.patchify(images)
            mean_tar = target.mean(dim=-1, keepdim=True)
            var_tar = target.var(dim=-1, keepdim=True)
            pred_unnorm = pred * (var_tar + 1e-6)**.5 + mean_tar

            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, target.shape[2])
            im_paste = target * (1 - mask_expanded) + pred_unnorm * mask_expanded
            im_masked = target * (1 - mask_expanded)

        # Unpatchify
        def unpatchify(x, p, img_size):
            h = w = img_size // p
            x = x.reshape(x.shape[0], h, w, p, p, 3)
            x = torch.einsum('nhwpqc->nchpwq', x)
            return x.reshape(x.shape[0], 3, h * p, h * p)

        def denormalize(img, mean, std):
            mean_t = torch.tensor(mean).view(3, 1, 1).to(img.device)
            std_t = torch.tensor(std).view(3, 1, 1).to(img.device)
            img = img * std_t + mean_t
            return torch.clamp(img, 0, 1)

        imgs_orig = denormalize(images, norm_mean, norm_std).cpu()
        imgs_recon = denormalize(unpatchify(im_paste, patch_size, image_size), norm_mean, norm_std).cpu()
        imgs_masked = denormalize(unpatchify(im_masked, patch_size, image_size), norm_mean, norm_std).cpu()

        # 绘图 — 每页 4 张 (3行×4列)，图片更大更清晰
        n_samples = imgs_orig.shape[0]
        n_per_page = 4
        n_pages = (n_samples + n_per_page - 1) // n_per_page

        for page in range(n_pages):
            start = page * n_per_page
            end = min(start + n_per_page, n_samples)
            n_col = end - start
            if n_col <= 0:
                break

            fig, axes = plt.subplots(3, n_col, figsize=(5 * n_col, 12))
            fig.suptitle(
                f'{mae_exp["name"]} Reconstruction — Page {page+1}/{n_pages}'
                f'\n(Row 1: Original | Row 2: {int(mask_ratio*100)}% Masked | Row 3: Reconstructed)',
                fontsize=14,
            )

            for i in range(n_col):
                idx = start + i
                axes[0, i].imshow(np.transpose(imgs_orig[idx].numpy(), (1, 2, 0)))
                axes[0, i].axis("off")
                axes[1, i].imshow(np.transpose(imgs_masked[idx].numpy(), (1, 2, 0)))
                axes[1, i].axis("off")
                axes[2, i].imshow(np.transpose(imgs_recon[idx].numpy(), (1, 2, 0)))
                axes[2, i].axis("off")

            plt.tight_layout()
            if page == 0:
                page_path = output_path
            else:
                page_path = RESULTS_DIR / (
                    Path(mae_exp["output"]).stem + f"_page{page+1}.png"
                )
            plt.savefig(page_path, dpi=150)
            plt.close()
            print(f"  ✓ 保存到 {page_path}")

        # ── 图 7: 误差热力图 ──
        # 保存当前实验的数据供热力图使用
        mae_exp["_orig"] = imgs_orig
        mae_exp["_recon"] = imgs_recon
        mae_exp["_loaded"] = True

    # 生成合并的误差热力图
    generate_error_heatmap(MAE_EXPERIMENTS)


def generate_error_heatmap(mae_experiments):
    print("生成 图7: mae_error_heatmap.png ...")

    loaded_exps = [e for e in mae_experiments if e.get("_loaded")]

    if len(loaded_exps) == 0:
        print("  ⚠ 没有可用的 MAE 重建数据，跳过热力图")
        return

    n_cols = 4  # 每页 4 张，图片更大更清晰
    n_rows = len(loaded_exps)
    n_pages = 3  # 共 3 页，每页 4 张 = 12 个样本

    for page in range(n_pages):
        start = page * n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols + 1.5, 5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row, exp in enumerate(loaded_exps):
            orig = exp["_orig"]
            recon = exp["_recon"]
            for i in range(n_cols):
                idx = start + i
                if idx >= orig.shape[0]:
                    axes[row, i].axis("off")
                    continue
                error = (orig[idx] - recon[idx]) ** 2
                error_map = error.mean(dim=0).numpy()
                im = axes[row, i].imshow(error_map, cmap="hot", vmin=0, vmax=0.05)
                axes[row, i].axis("off")
                # 显示平均 MSE 数值
                axes[row, i].set_title(f"MSE={error_map.mean():.4f}", fontsize=10)
                if i == 0:
                    axes[row, i].set_ylabel(exp["name"], fontsize=12, rotation=0, labelpad=120, va="center")

        # 添加颜色条
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("MSE per pixel", fontsize=11)

        fig.suptitle(
            f"Reconstruction Error Heatmap — Page {page+1}/{n_pages}\n"
            f"(Darker = Better Reconstruction | Brighter = Larger Error)",
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 0.92, 0.92])

        if page == 0:
            page_path = RESULTS_DIR / "mae_error_heatmap.png"
        else:
            page_path = RESULTS_DIR / f"mae_error_heatmap_page{page+1}.png"
        plt.savefig(page_path, dpi=150)
        plt.close()
        print(f"  ✓ 保存到 {page_path}")


# ──────────────────────────────────────────────
#  表 1: 结果汇总
# ──────────────────────────────────────────────

def generate_summary_table():
    print("生成 表1: summary_table.csv ...")

    fieldnames = ["method", "best_test_acc", "best_epoch", "params_M", "elapsed_s"]
    rows = []

    config_map = {
        "outputs/vanilla_vit_cifar10": ("configs/vanilla_vit.yaml", "Vanilla ViT (scratch)"),
        "outputs/convstem_vit_cifar10": ("configs/convstem_vit.yaml", "ConvStem-ViT (scratch)"),
        "outputs/finetune_vanilla_vit_cifar10": ("configs/finetune_vanilla_vit_cifar10.yaml", "Vanilla ViT (MAE pretrained)"),
        "outputs/finetune_convstem_vit_cifar10": ("configs/finetune_convstem_vit_cifar10.yaml", "ConvStem-ViT (MAE pretrained)"),
    }

    import torch
    from src.models import create_model
    import yaml

    for exp_dir, (config_path, method_name) in config_map.items():
        summary = load_summary(exp_dir)
        if not summary:
            continue

        # 计算参数量
        if Path(config_path).exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            model = create_model(cfg)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
        else:
            n_params = 0

        rows.append({
            "method": method_name,
            "best_test_acc": summary.get("best_test_acc", 0),
            "best_epoch": summary.get("best_epoch", 0),
            "params_M": round(n_params, 2),
            "elapsed_s": summary.get("elapsed_seconds", 0),
        })

    output_path = RESULTS_DIR / "summary_table.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ 保存到 {output_path}")
    # 打印表格
    print("\n  Results Summary:")
    print("  " + "-" * 80)
    print(f"  {'Method':<35} {'Acc':>8} {'Epoch':>6} {'Params':>8} {'Time(s)':>8}")
    print("  " + "-" * 80)
    for row in rows:
        print(f"  {row['method']:<35} {row['best_test_acc']:>8.4f} {row['best_epoch']:>6d} {row['params_M']:>7.2f}M {row['elapsed_s']:>8.1f}")
    print("  " + "-" * 80)


# ──────────────────────────────────────────────
#  表 2: 消融实验
# ──────────────────────────────────────────────

def generate_ablation_table():
    print("生成 表2: ablation_table.csv ...")

    # 从 summary 中读取结果
    scratch_vanilla = load_summary("outputs/vanilla_vit_cifar10").get("best_test_acc", "N/A")
    scratch_convstem = load_summary("outputs/convstem_vit_cifar10").get("best_test_acc", "N/A")
    pretrained_vanilla = load_summary("outputs/finetune_vanilla_vit_cifar10").get("best_test_acc", "N/A")
    pretrained_convstem = load_summary("outputs/finetune_convstem_vit_cifar10").get("best_test_acc", "N/A")

    rows = [
        {"patch_embed": "Linear", "pretraining": "No (scratch)", "test_accuracy": scratch_vanilla},
        {"patch_embed": "ConvStem", "pretraining": "No (scratch)", "test_accuracy": scratch_convstem},
        {"patch_embed": "Linear", "pretraining": "Yes (MAE)", "test_accuracy": pretrained_vanilla},
        {"patch_embed": "ConvStem", "pretraining": "Yes (MAE)", "test_accuracy": pretrained_convstem},
    ]

    output_path = RESULTS_DIR / "ablation_table.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch_embed", "pretraining", "test_accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ 保存到 {output_path}")
    print("\n  Ablation Study (2×2):")
    print("  " + "-" * 55)
    print(f"  {'Patch Embed':<15} {'Pretraining':<18} {'Test Accuracy':>15}")
    print("  " + "-" * 55)
    for row in rows:
        acc = row["test_accuracy"]
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        print(f"  {row['patch_embed']:<15} {row['pretraining']:<18} {acc_str:>15}")
    print("  " + "-" * 55)


# ──────────────────────────────────────────────
#  主函数
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generating all visualizations and tables")
    print("=" * 60)

    # 曲线图
    plot_accuracy_curves()
    plot_loss_curves()
    plot_convergence_speed()
    plot_params_vs_accuracy()

    # MAE 重建 + 误差热力图
    generate_mae_reconstructions()

    # 表格
    generate_summary_table()
    generate_ablation_table()

    print("\n" + "=" * 60)
    print("All done! Results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
