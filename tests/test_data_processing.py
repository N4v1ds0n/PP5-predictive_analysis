import os
import tempfile
import shutil
from pathlib import Path
import pytest
from PIL import Image

from src.data_processing import (
    remove_non_image_files,
    remove_corrupt_images,
    split_dataset,
    clear_splits,
    count_dataset_images,
)


# Helper: create a dummy valid image file
def create_valid_image(path):
    img = Image.new('RGB', (10, 10), color='red')
    img.save(path)


# Helper: create a dummy corrupt image file (empty or random data)
def create_corrupt_image(path):
    with open(path, 'wb') as f:
        f.write(b"not an image content")


def test_remove_non_image_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        class_dir = Path(tmpdir, "classA")
        class_dir.mkdir()
        # Create images and non-images
        (class_dir / "img1.png").write_text("fake image content")
        (class_dir / "doc.txt").write_text("text file")
        (class_dir / "img2.jpg").write_text("another fake image")
        (class_dir / "notes.md").write_text("not an image")

        remove_non_image_files(tmpdir)

        files_left = {f.name for f in class_dir.iterdir()}
        assert "img1.png" in files_left
        assert "img2.jpg" in files_left
        assert "doc.txt" not in files_left
        assert "notes.md" not in files_left


def test_remove_corrupt_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_path = Path(tmpdir, "valid.png")
        corrupt_path = Path(tmpdir, "corrupt.png")

        create_valid_image(valid_path)
        create_corrupt_image(corrupt_path)

        removed = remove_corrupt_images(tmpdir)

        removed_paths = [os.path.normpath(p) for p in removed]
        assert os.path.normpath(str(corrupt_path)) in removed_paths
        assert corrupt_path.exists() is False
        assert valid_path.exists() is True


def test_split_dataset_and_clear_splits():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup dummy classes with files
        class_names = ["class1", "class2"]
        num_files = 10

        for cls in class_names:
            cls_dir = Path(tmpdir, cls)
            cls_dir.mkdir()
            for i in range(num_files):
                create_valid_image(cls_dir / f"img_{i}.png")

        # Clear any previous splits if any (should be none)
        clear_splits(tmpdir)

        # Check split_dataset with correct ratios
        split_dataset(tmpdir, train_ratio=0.6,
                      validation_ratio=0.2, test_ratio=0.2)

        # Check folders created and files split
        for split in ["train", "validation", "test"]:
            for cls in class_names:
                split_dir = Path(tmpdir, split, cls)
                assert split_dir.exists()
                files = list(split_dir.iterdir())
                # Files per split: 6, 2, 2 roughly for 10 files
                # Allow some margin due to int rounding
                assert len(files) >= 1

        # Now clear splits and check folders removed
        clear_splits(tmpdir)
        for split in ["train", "validation", "test"]:
            assert not Path(tmpdir, split).exists()


def test_split_dataset_invalid_ratios():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError):
            split_dataset(tmpdir, 0.5, 0.3, 0.3)  # sums > 1


def test_count_dataset_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        class_names = ["healthy", "infected"]
        sets = ["train", "validation", "test"]

        # Create folders and add images
        for s in sets:
            for c in class_names:
                dir_path = Path(tmpdir, s, c)
                dir_path.mkdir(parents=True)
                for i in range(3):
                    create_valid_image(dir_path / f"img_{i}.png")

        total = count_dataset_images(tmpdir, sets, class_names)
        assert total == 3 * len(class_names) * len(sets)

        # Test missing folder handling (remove one)
        shutil.rmtree(Path(tmpdir, "validation"))
