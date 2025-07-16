from pathlib import Path
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_split_counts(data_dir,
                     splits=('train', 'validation', 'test'),
                     labels=('healthy', 'diseased')):
    """
    Get image counts for each class per split.
    """
    data = {'Set': [], 'Label': [], 'Frequency': []}
    data_dir = Path(data_dir)

    for split in splits:
        for label in labels:
            path = data_dir / split / label
            count = len(os.listdir(path)) if path.exists() else 0
            data['Set'].append(split)
            data['Label'].append(label)
            data['Frequency'].append(count)
            print(f"* {split} - {label}: {count} images")

    return pd.DataFrame(data)


def plot_split_distribution(df_freq, save_path=None):
    """
    Visualize class distribution across dataset splits.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_freq, x='Set', y='Frequency', hue='Label')
    plt.title("Dataset Class Distribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
