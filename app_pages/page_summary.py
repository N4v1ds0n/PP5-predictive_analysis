import streamlit as st
import glob


def page_summary_body():
    """Enhanced Project Summary Page with clear layout, improved visuals, and
    interactive elements."""

    st.title("🌿 Project Summary")

    # ─── Problem and Solution Overview ───────────────────────────────────────
    st.markdown("## The Challenge")

    st.markdown("""
    Farmy & Foods, a leading agricultural company, is facing major difficulties
    with:

    - **Manual detection of powdery mildew**, taking **~30 minutes per tree**
    - Thousands of cherry trees across **multiple plantations**
    - A time-intensive process that does not scale

    **The solution?** An **AI-powered image recognition model** that:
    - Classifies leaves as **Healthy** or **Infected**
    - Provides **instant predictions**, saving **hours of labor**
    - Enables future **expansion to other crops**
    """)

    st.divider()

    # ─── Sample Images ──────────────────────────────────────────────────────
    st.markdown("## Sample Cherry Leaf Images")

    healthy_images = glob.glob(
        "inputs/datasets/raw/cherry-leaves/validation/healthy/*.JPG")
    diseased_images = glob.glob(
        "inputs/datasets/raw/cherry-leaves/validation/diseased/*.JPG")

    if healthy_images and diseased_images:
        image_paths = [healthy_images[0], diseased_images[0]]
        captions = ["✅ Healthy Leaf", "⚠️ Mildew-Infected Leaf"]

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_paths[0],
                     caption=captions[0], use_container_width=True)
        with col2:
            st.image(image_paths[1],
                     caption=captions[1], use_container_width=True)
    else:
        st.warning("⚠️ No sample images found! Please check dataset paths.")

    st.caption("Note: Infected leaves show **white dot patterns "
               "or discoloration**.")

    st.divider()

    # ─── Business Requirements ───────────────────────────────────────────────
    st.markdown("## Business Requirements")

    st.markdown("""
    This project directly addresses **three key business needs**:

    1️⃣ **Distinguish** between healthy and mildew-infected cherry leaves
    2️⃣ **Automate** leaf classification using a **deep learning model**
    3️⃣ **Provide a clear prediction report** with each model output —
    ensuring transparency, trust, and usability
    """)

    st.divider()

    # ─── Dataset Overview ────────────────────────────────────────────────────
    st.markdown("## Dataset Overview")

    st.markdown("""
    The dataset used in this project contains **over 4,000 cherry leaf
    images**, labeled as:

    - **Healthy**
    - **Diseased (Powdery Mildew)**

    Source:
    [Kaggle - Dataset](https://www.kaggle.com/codeinstitute/cherry-leaves)

    For this study:
    - The dataset was split into **training**, **validation**, and **test**
    folders
    - **Training**: 70% of images
    - **Validation**: 15% of images
    - **Test**: 15% of images
    """)

    st.divider()

    # ─── External Resources ──────────────────────────────────────────────────
    with st.expander("🔗 Additional Resources and References"):
        st.markdown(
            "- 📖 [Project README (GitHub)](https://github.com/"
            "N4v1ds0n/PP5-predictive_analysis/blob/main/README.md)\n"
            "- 🌿 [Wikipedia: Powdery Mildew]("
            "https://en.wikipedia.org/wiki/Powdery_mildew)"
        )
