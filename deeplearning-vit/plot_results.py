import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from metrics.csv files.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories under outputs/.")
    parser.add_argument("--output", default="results/training_curves.png")
    args = parser.parse_args()

    plt.figure(figsize=(7, 4))
    for run in args.runs:
        run_path = Path(run)
        data = load_csv(run_path / "metrics.csv")
        plt.plot(data["epoch"], data["test_acc"], label=run_path.name)

    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
