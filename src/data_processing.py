import os
from PIL import Image
import shutil
import random


def remove_non_image_files(data_dir):
    image_extensions = ('.png', '.jpg', '.jpeg')

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip files in the root folder

        image_count = 0
        non_image_count = 0

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                if not filename.lower().endswith(image_extensions):
                    os.remove(file_path)
                    non_image_count += 1
                else:
                    image_count += 1

        print(
            f"Folder '{folder_name}': Image files = {image_count}, "
            f"Non-image files removed = {non_image_count}"
        )


def remove_corrupt_images(data_dir):
    """
    Detects and removes corrupt image files from a dataset.

    Args:
        dataset_dir (str): Path to the root dataset directory.

    Returns:
        list: A list of removed corrupt image file paths.
    """
    corrupt_images = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify image integrity without loading it
            except (IOError, SyntaxError, ValueError) as e:
                print(f"Removing corrupt image: {file_path} (Reason: {e})")
                corrupt_images.append(file_path)
                try:
                    os.remove(file_path)
                except Exception as remove_error:
                    print(f"Failed to remove {file_path}: {remove_error}")

    print(f"✅ Total corrupt images removed: {len(corrupt_images)}")
    return corrupt_images


def split_dataset(data_dir, train_ratio, validation_ratio, test_ratio):
    """
    Split the dataset into training, validation, and test sets.
    The dataset directory must contain one folder per class label.

    Args:
        my_data_dir (str): Path to the dataset directory.
        train_ratio (float): Proportion of images for training.
        validation_ratio (float): Proportion for validation.
        test_ratio (float): Proportion for testing.

    Returns:
        None
    """
    # Validate that the sum of train, validation, and test ratios equals 1.0
    if train_ratio + validation_ratio + test_ratio != 1.0:
        raise ValueError(
            "Error: train, validation and test ratios must sum to 1.0"
        )

    # Get the class labels in the dataset directory
    labels = [
        label for label in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, label))
        and label not in ['train', 'validation', 'test']
    ]

    # Create subdirectories for each split and class
    for split in ['train', 'validation', 'test']:
        for label in labels:
            os.makedirs(os.path.join(data_dir, split, label), exist_ok=True)

    for label in labels:
        class_path = os.path.join(data_dir, label)
        files = os.listdir(class_path)
        random.shuffle(files)

        train_end = int(len(files) * train_ratio)
        val_end = train_end + int(len(files) * validation_ratio)

        for i, file_name in enumerate(files):
            src = os.path.join(class_path, file_name)

            if i < train_end:
                dst = os.path.join(data_dir, 'train', label, file_name)
            elif i < val_end:
                dst = os.path.join(data_dir, 'validation', label, file_name)
            else:
                dst = os.path.join(data_dir, 'test', label, file_name)

            shutil.copy2(src, dst)

        # os.rmdir(class_path)


def clear_splits(data_dir):
    """
    Clear already existing train, validation,
    and test splits by deleting their directories.
    """
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            shutil.rmtree(split_path)  # Deletes the folder and contents


def count_dataset_images(base_path, sets, labels):
    """
    Count and report image files in a structured dataset directory.

    Args:
        base_path (str): Base directory where the dataset is stored.
        sets (list): Subfolders like ['train', 'validation', 'test'].
        labels (list): Class names like ['Healthy', 'Infected'].

    Returns:
        int: Total number of images across all sets and labels.
    """
    total_images = 0

    for set_name in sets:
        for label in labels:
            path = os.path.join(base_path, set_name, label)
            try:
                files = [
                    f for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))
                ]
                file_count = len(files)
                total_images += file_count
                print(f"There are {file_count} images in {set_name}/{label}")
            except FileNotFoundError:
                print(f"Warning: Directory '{path}' not found.")

    print(f"\nTotal number of images: {total_images}")
    return total_images
