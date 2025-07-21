import streamlit as st


def page_working_hypothesis_body():
    """Displays working hypotheses, validation strategies, and conclusions."""

    st.title("🔍 Project Hypotheses & Insights")
    st.write(
        "This section outlines our initial assumptions, how we tested them, "
        "and the conclusions drawn from exploratory data analysis and modeling."
    )

    # Hypothesis 1
    st.header("Hypothesis 1: Visual Signs of Infection Are Detectable")
    st.success(
        "Cherry leaves infected with powdery mildew exhibit distinct visual features "
        "such as discoloration, patchiness, and texture irregularities that can be "
        "distinguished from healthy leaves."
    )

    st.info(
        "**Validation Methods:**\n"
        "- 📊 **Mean & Standard Deviation Images** to visualize class-level texture patterns\n"
        "- 🧭 **PCA** to examine class separability in feature space\n"
        "- t-SNE to visualize high-dimensional data in 2D\n"
        "- UMAP to explore local structure and class clusters "
    )

    st.warning(
        "**Findings & Conclusion:**\n"
        "- Mildew-infected leaves exhibit localized brightness and textural noise\n"
        "- PCA revealed moderate but usable class separation\n"
        "- t-SNE reveals better-defined local groupings, especially for the healthy class (red), which appears more compact and clustered toward the lower part of the graph.\n"
        "- In the UMAP, diseased and healthy samples form distinct clusters, with some overlap. This implies features contain highly discriminative information."
        "✅ **Conclusion:** Visual symptoms are present and exploitable for classification."
    )

    # Hypothesis 2
    st.header("Hypothesis 2: A CNN Can Reliably Detect Mildew")
    st.success(
        "A convolutional neural network (CNN), trained on cherry leaf images, "
        "can learn distinguishing features and achieve up to **100% accuracy** when predicting on the test set. "
        "in predicting whether a leaf is healthy or infected."
    )

    st.info(
        "**Validation Methods:**\n"
        "- 🤖 **Model Training & Evaluation** using accuracy, precision, recall, F1-score\n"
        "- 📉 **Confusion Matrix** to examine classification errors\n"
        "- 📈 **ROC Curve & AUC Score** to assess model confidence and reliability\n"
        "- 🔍 **Checking for Dataset Overlaps** to prevent data leakage\n"
        "- 🔄 **K-Fold Cross-Validation** to ensure robustness\n"
    )

    st.warning(
        "**Findings & Conclusion:**\n"
        "- The trained CNN achieved an accuracy of **≥99%** on the test set\n"
        "- False negatives were minimal, supporting reliability in detection\n"
        "- ROC AUC score **≥0.99**, confirming strong model performance\n\n"
        "✅ **Conclusion:** The CNN is well-suited for image-based mildew detection."
    )