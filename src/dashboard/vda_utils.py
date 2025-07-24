import streamlit as st
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import itertools
import random


def show_average_images(output_dir: Path):
    """Displays average & variability images if available."""

    if st.checkbox("Average & Variability Images"):
        avg_healthy = output_dir / "avg_std_healthy.png"
        avg_infected = output_dir / "avg_std_diseased.png"

        if avg_healthy.exists() and avg_infected.exists():
            st.warning(
                "⚠️ Infected leaves show more **textural noise** and "
                "**white streaks at the center**, suggesting possible "
                "textural clues."
            )

            st.image(
                [str(avg_healthy), str(avg_infected)],
                caption=[
                    "Healthy Leaf - Average & Variability",
                    "Mildew-Infected Leaf - Average & Variability",
                ],
                use_container_width=True,
            )
        else:
            st.error("Required average/variability image files not found.")

        st.markdown("---")


def show_difference_image(output_dir: Path):
    """Displays difference between average healthy and infected leaf images."""

    if st.checkbox("Difference Between Average Healthy & Infected Leaves"):
        diff_img_path = output_dir / "avg_diff.png"

        if diff_img_path.exists():
            st.warning(
                "⚠️ The difference image reveals subtle but clear "
                "variations, but not obvious to the naked eye. "
                "Further automated feature extraction may help."
            )
            st.image(str(diff_img_path),
                     caption="Difference Between Average Images")
        else:
            st.error("Difference image not found."
                     "Please generate it during preprocessing.")

        st.markdown("---")


def show_image_montage(data_dir: Path):
    """Displays a montage of sample images from the validation set."""

    if st.checkbox("Image Montage"):
        st.write("Click 'Create Montage' to show a montage of random images.\n"
                 "\nSelect 'healthy' or 'diseased' to filter images.")

        if not data_dir.exists():
            st.error(f"Validation directory not found: {data_dir}")
            return

        class_lbls = sorted([f.name for f in data_dir.iterdir() if f.is_dir()])
        if not class_lbls:
            st.error("No class subdirectories found in validation set.")
            return

        label_to_display = st.selectbox("Select Class Label",
                                        options=class_lbls)

        if st.button("Create Montage"):
            image_montage(
                dir_path=data_dir,
                label_to_display=label_to_display,
                nrows=8,
                ncols=3,
                figsize=(10, 25),
            )

        st.markdown("---")


def image_montage(dir_path: Path, label_to_display: str,
                  nrows: int, ncols: int, figsize=(10, 10)):
    """
    Generates and displays a montage of random images from a selected class.
    """
    sns.set_style("white")

    class_dir = dir_path / label_to_display
    if not class_dir.exists():
        st.error(f"⚠️ Selected label folder not found: {class_dir}")
        return

    images_list = list(class_dir.glob("*"))
    total_images = len(images_list)
    required_images = nrows * ncols

    if total_images < required_images:
        st.warning(
            f"Only {total_images} images found,"
            f" but {required_images} requested."
        )
        return

    selected_images = random.sample(images_list, required_images)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    plot_positions = list(itertools.product(range(nrows), range(ncols)))

    for i, img_path in enumerate(selected_images):
        img = imread(img_path)
        row, col = plot_positions[i]

        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{img.shape[1]}px × {img.shape[0]}px")
        axes[row, col].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
