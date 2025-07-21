import streamlit as st
from pathlib import Path
from src.dashboard.vda_utils import (
    show_average_images,
    show_difference_image,
    show_image_montage
)


def page_visual_diagnosis_assistant_body():
    """
    Streamlit UI for visualizing 
    healthy vs. mildew-infected cherry leaves.
    """

    st.title("🍒 Visual Diagnosis Assistant")
    st.info(
        "This study explores **Healthy vs. Mildew-Infected** cherry leaves. "
        "The goal is to understand patterns before applying machine learning."
    )

    version = "v1"
    output_dir = Path("outputs") / version
    data_dir = Path("inputs/datasets/raw/cherry-leaves/validation")

    show_average_images(output_dir)
    show_difference_image(output_dir)
    show_image_montage(data_dir)



