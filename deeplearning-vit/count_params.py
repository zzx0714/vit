import argparse

from src.models import create_model
from src.utils import load_config


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Count trainable model parameters.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model = create_model(config)
    count = count_trainable_parameters(model)
    print(f"{config['experiment_name']}: {count:,} trainable parameters")


if __name__ == "__main__":
    main()
