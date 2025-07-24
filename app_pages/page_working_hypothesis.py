import streamlit as st


def page_working_hypothesis_body():
    """
    Displays working hypotheses, validation strategies, and conclusions.
    """

    st.title("🔍 Project Hypotheses & Insights")
    st.write(
        """
        This section outlines initial assumptions, how they were validated,
        and the conclusions drawn from EDA and modeling.
        """
    )

    # Hypothesis 1
    st.header("Hypothesis 1: Visual signs of infection are detectable")
    st.warning(
        """
        **Statement**:\n
        Cherry leaves infected with powdery mildew exhibit distinct visual
        features e.g. discoloration, patchiness, and texture irregularities
        that can be distinguished from healthy leaves.
        """
    )

    st.info(
        """
        **Validation Methods:**\n
        - Manual inspection of the dataset confirmed consistent and distinct
        mildew markings.\n
        - **Mean & Standard Deviation Images** to visualize class-level
        texture patterns\n
        - **PCA** to examine class separability in feature space\n
        - t-SNE to improve non-linear separability\n
        - UMAP to explore local structure and class clusters
        """
    )

    st.success(
        """
        **Findings & Conclusion:**\n
        - Mildew-infected leaves exhibit localized brightness and
        textural noise\n
        - PCA revealed moderate but usable class separation\n
        - t-SNE reveals better-defined local groupings, especially for the
        healthy class, which appears more compact and clustered toward the
        lower part of the graph.\n
        - In the UMAP, diseased and healthy samples form distinct clusters,
        with some overlap. This implies features contain highly
        discriminative information.\n\n
        **Conclusion:** Visual symptoms are present and exploitable for
        classification.
        """
    )

    # Hypothesis 2
    st.header("Hypothesis 2: A softmax output neuron is the better choice "
              "despite of binary dataset"
              )
    st.warning(
        """
        **Statement**:\n
        Although for a binary classification problem the single output neuron
        setup with a sigmoid function returning 1 or 0 is the default approach.
        However there are indications, that especially for gradient based
        optimization a categorical approach with two softmax Neurons might be
        better. This is because categorical loss can sometimes offer better
        learning curves due to gradient stability.
        """
    )

    st.info(
        """
        **Validation Methods:**\n
        - Comparison of sigmoid vs. softmax output layers by running a
        kfold for each setup to compare accuracy and time consumption\n
        """
    )

    st.success(
        """
        **Findings & Conclusion:**\n
        - the models produced well-calibrated probability scores.\n
        - The softmax approach showed overall better performance\n\n
        **Conclusion:** The kfold duel showed that the softmax approach
        is not only reaching a higher accuracy, but also training faster
        on average, making it the objectively better choice for our model.
        """
    )

    # Hypothesis 3
    st.header("Hypothesis 3: A CNN can reliably detect mildew")
    st.warning(
        """
        **Statement**:\n
        A Convolutional Neural Network (CNN) can learn the relevant
        spatial patterns in the leaf images to distinguish between
        healthy and infected samples with high accuracy.
        """
    )

    st.info(
        """
        **Validation Methods:**\n
        - **Model Training & Evaluation** using accuracy, precision, recall,
        F1-score\n
        - **Confusion Matrix** to examine classification errors\n
        - **ROC Curve & AUC Score** to assess model confidence and
        reliability\n
        - **Checking for Dataset Overlaps** to prevent data leakage\n
        - **K-Fold Cross-Validation** to ensure robustness\n
        """
    )

    st.success(
        """
        **Findings & Conclusion:**\n
        - The trained CNN achieved an accuracy of **≥99%** on the test set\n
        - False negatives were minimal, supporting reliability in detection\n
        - ROC AUC score **1**, confirming strong model performance\n\n
        **Conclusion:** The model can make very reliable predictions on
        the test data and distinguish between healthy and mildew-infected
        leaves.
        """
    )

    # Hypothesis 4
    st.header("Hypothesis 4: The model might have difficulties to generalize "
              "beyond the training domain")
    st.warning(
        """
        **Statement**:\n
        A model trained on clean, uniformly staged leaf images might have
        problems to effectively extraploate to to real-world conditions where
        leaves appear in varied lighting, angles, and backgrounds. If the
        leaves are not photographed under the same conditions the model might
        be misclassifying.
        """
    )

    st.info(
        """
        **Validation Methods:**\n
        - A set of real-world test images (generated by AI, but realistic)
        were used to evaluate model robustness.
        """
    )

    st.success(
        """
        **Findings & Conclusion:**\n
        - The model suffers from poor extrapolation due to the highly uniform
        dataset.\n
        - For field-ready performance, substantial improvements should be
        made.\n
        **Conclusion:**The models performance is good in a very specific setup.
        To create a robust flexible model for real-world field use the model
        needs dataset diversification (adding naturally captured leaf images
        to the training set), domain adaptation techniques, or synthetic data
        generation that mimics field conditions.

        """
    )
