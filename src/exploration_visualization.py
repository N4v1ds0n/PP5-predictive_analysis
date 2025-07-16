import os
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import load_img, img_to_array   # type: ignore
import numpy as np
import seaborn as sns
from pathlib import Path
import itertools
from imageio import imread


def check_structure(data_dir, sets, labels):
    """
    Check integrity of the directory structure for the datasets.
    """
    for set_name in sets:
        for label in labels:
            path = os.path.join(data_dir, set_name, label)

            if not os.path.exists(path):
                print(f"Warning: '{label}' in '{set_name}' is missing.")
            elif not os.listdir(path):
                print(
                    f"Notice: '{label}' in '{set_name}' exists but is empty."
                    )
            else:
                image_count = len(os.listdir(path))
                print(
                    f"'{label}' in '{set_name}' is valid with "
                    f"{image_count} images."
                )


def count_images_for_class(data_dir, class_name, splits):
    return sum(
        len(os.listdir(os.path.join(data_dir, split, class_name)))
        for split in splits
    )


def preview_class_samples(dataset_path,
                          class_names,
                          n_images=5,
                          img_size=(128, 128)):
    """
    Display a grid of sample images per class for visual inspection.

    Args:
        dataset_path (str): Path to a dataset split (e.g., 'test' folder).
        class_names (list): List of class labels (['healthy', 'diseased']).
        n_images (int): Number of images to show per class.
        img_size (tuple): Target image size for display.
    """
    fig, axes = plt.subplots(len(class_names), n_images,
                             figsize=(n_images * 2.5, len(class_names) * 3))

    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(image_files) < n_images:
            raise ValueError(
                f"Not enough images in '{class_dir}' "
                f"to sample {n_images} images."
                )

        sampled_files = random.sample(image_files, n_images)

        for j, file_name in enumerate(sampled_files):
            img_path = os.path.join(class_dir, file_name)
            img = load_img(img_path, target_size=img_size)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            axes[i, j].set_title(class_name.capitalize())

    plt.tight_layout()
    plt.show()


def load_and_preprocess_images(data_dir,
                               target_size=(128, 128),
                               max_images_per_class=20):
    """
    Load a limited number of images per class from the dataset directory,
    resize them to target_size, normalize pixel values, and return as arrays.

    Args:
        data_dir (str): Base path to dataset folder with class subfolders.
        target_size (tuple): Desired image size as (height, width).
        max_images_per_class (int): Max number of images to load per class.

    Returns:
        X (np.ndarray): Image data array of shape (n_samples, height, width,
        channels).
        y (np.ndarray): Corresponding labels.
    """
    X_list = []
    y_list = []

    for label in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue  # skip files

        image_files = os.listdir(label_path)[:max_images_per_class]
        for img_name in image_files:
            img_path = os.path.join(label_path, img_name)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            X_list.append(img_array)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def plot_class_image_statistics(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str = None,
    figsize: tuple = (12, 5),
    save_image: bool = False,
) -> None:
    """
    Plot average and variability (standard deviation) images for each label.

    Args:
        X (np.ndarray): Array of image data.
        y (np.ndarray): Corresponding labels.
        output_dir (str): Directory to save images if save_image is True.
        figsize (tuple): Size of each figure.
        save_image (bool): Whether to save or just show the plots.
    """
    labels = np.unique(y)
    if save_image and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for label in labels:
        sns.set_style("white")
        mask = y == label
        images = X[mask]

        avg_img = np.mean(images, axis=0)
        std_img = np.std(images, axis=0)

        print(f"==== Label: {label} ====")
        print(f"Image shape: {avg_img.shape}")

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"Label: {label}", fontsize=14)

        axes[0].imshow(
            avg_img if avg_img.shape[-1] == 3 else avg_img.squeeze(),
            cmap=None if avg_img.shape[-1] == 3 else 'gray'
        )
        axes[0].set_title("Average Image")
        axes[0].axis("off")

        axes[1].imshow(
            std_img if std_img.shape[-1] == 3 else std_img.squeeze(),
            cmap=None if std_img.shape[-1] == 3 else 'gray'
        )
        axes[1].set_title("Variability (Std Dev)")
        axes[1].axis("off")

        plt.tight_layout()
        if save_image and output_dir:
            file_name = output_dir / f"avg_std_{label}.png"
            plt.savefig(file_name, bbox_inches="tight", dpi=150)
            print(f"Saved: {file_name}")
            plt.close()
        else:
            plt.show()
        print()


def filter_images_by_label(X, y, label):
    """
    Return a subset of images corresponding to a given label.

    Args:
        X (ndarray): Image data.
        y (ndarray): Label array.
        label (str): The label to filter by.

    Returns:
        ndarray: Filtered image array for the specified label.
    """
    return X[y == label]


def compare_average_images(X, y, label_a, label_b,
                           figsize=(12, 10), save_path=None):
    """
    Plot and compare the average images of two labels
    and their pixel-wise difference.

    Args:
        X (ndarray): Image data array of shape (N, H, W, C).
        y (ndarray): Corresponding labels.
        label_a (str): First label to compare.
        label_b (str): Second label to compare.
        figsize (tuple): Size of the matplotlib figure.
        save_path (str): Path to save the figure. If None,
        the plot is shown instead.
    """
    sns.set_style("white")

    avg_img_a = np.mean(filter_images_by_label(X, y, label_a), axis=0)
    avg_img_b = np.mean(filter_images_by_label(X, y, label_b), axis=0)

    diff_img = np.clip(avg_img_a - avg_img_b, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    cmap = None if avg_img_a.shape[-1] == 3 else 'gray'

    axes[0].imshow(avg_img_a, cmap=cmap)
    axes[0].set_title(f"Average: {label_a.capitalize()}")
    axes[0].axis("off")

    axes[1].imshow(avg_img_b, cmap=cmap)
    axes[1].set_title(f"Average: {label_b.capitalize()}")
    axes[1].axis("off")

    axes[2].imshow(diff_img, cmap=cmap)
    axes[2].set_title(f"Difference: {label_a} - {label_b}")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()


def image_montage(dir_path, label, nrows=3, ncols=3, figsize=(10, 12)):
    """
    Displays a grid of randomly selected images from a specific label folder.

    Args:
        dir_path (str): Path to dataset directory containing label folders.
        label (str): Label subdirectory to display images from.
        nrows (int): Number of rows in the montage.
        ncols (int): Number of columns in the montage.
        figsize (tuple): Size of the matplotlib figure.
    """
    sns.set_style("white")

    available_labels = os.listdir(dir_path)
    label_path = os.path.join(dir_path, label)

    if label not in available_labels:
        print(f"Label '{label}' not found. "
              f"Available labels: {available_labels}")
        return

    image_files = os.listdir(label_path)
    total_required = nrows * ncols

    if len(image_files) < total_required:
        print(
            f"Not enough images in '{label}' to fill a {nrows}x{ncols} grid.\n"
            f"Available: {len(image_files)} — Requested: {total_required}."
        )
        return

    selected_images = random.sample(image_files, total_required)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for idx, (row, col) in enumerate(itertools.product(range(nrows),
                                                       range(ncols))):
        img_path = os.path.join(label_path, selected_images[idx])
        img = imread(img_path)
        img_shape = img.shape

        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{img_shape[1]}×{img_shape[0]} px")
        axes[row, col].axis("off")

    fig.suptitle(f"Image Montage: {label.capitalize()}", fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
