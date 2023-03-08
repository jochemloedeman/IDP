import argparse

import pandas as pd


def extract_val_accuracy(path: str):
    metrics_df = pd.read_csv(path)
    accuracies = metrics_df['val_top1_accuracy']
    return print(accuracies.max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    extract_val_accuracy(args.path)
