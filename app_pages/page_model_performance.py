# app_pages/page_ml_performance_metrics.py

import streamlit as st
import pandas as pd
from src.dashboard.performance_utils import (
    load_test_metrics,
    load_csv,
    load_drt_figure,
    display_image_centered,
    center_component,
)


def page_ml_performance_body():
    version = "v1"
    st.title("Model Evaluation Dashboard")

    st.info(
        "Explore the model’s performance via metrics and visualizations.")

    # --- Class Distribution ---
    st.subheader("Dataset Split & Class Distribution")
    center_component(lambda: st.image(
        f"outputs/{version}/labels_distribution.png",
        caption="Class Distribution in Train, Validation, and Test Sets",
        use_container_width=True
    ))
    center_component(lambda: st.info(
        """
        The data is perfectly balanced, which will help to reduce model bias.
        The train, validation, and test sets are split as follows:
        - Train: 70%
        - Validation: 15%
        - Test: 15%\n
        The size of the dataset is moderate which means augmentation will be
        applied. However augmenting validation and test sets is not
        recommended, as it can cause artifacts and will also impair
        the models ability to adapt to real world use.
        """
    ))

    st.divider()

    # --- PCA ---
    st.subheader("Feature Space (PCA)")
    pca_df = load_csv(f"outputs/{version}/pca_results.csv",
                      error_msg="PCA data not found.")
    if pca_df is not None:
        center_component(lambda: st.plotly_chart(load_drt_figure(pca_df)))
        center_component(lambda: st.info(
            """
            Partial class separation suggests overlapping but
            distinguishable features. There's a slight tendency for each class
            to lean to one side (healthy toward the right, diseased more
            spread left), but it's not strongly separated. This suggests that
            linear combinations of features are not sufficient for clear class
            separation.
            """
        ))

    st.divider()

    # --- t-SNE ---
    st.subheader("Feature Space (t-SNE)")
    tsne_df = load_csv(f"outputs/{version}/tsne_results.csv",
                       error_msg="t-SNE data not found.")
    if tsne_df is not None:
        center_component(lambda: st.plotly_chart(load_drt_figure(tsne_df)))
        center_component(lambda: st.info(
            """
            t-SNE reveals better-defined local groupings, especially
            for the healthy class, which appears more compact and clustered
            toward the lower part of the graph.
            """
        ))

    st.divider()

    # --- UMAP ---
    st.subheader("Feature Space (UMAP)")
    umap_df = load_csv(f"outputs/{version}/umap_results.csv",
                       error_msg="UMAP data not found.")
    if umap_df is not None:
        center_component(lambda: st.plotly_chart(load_drt_figure(umap_df)))
        center_component(lambda: st.info(
            """
            Class Separation: The "diseased" and "healthy" samples form
            distinct clusters, with some outliers of the healthy class
            overlapping to the diseased cluster. This implies the features
            contain overall highly discriminative information.
            Non-linear Structure: UMAP, like t-SNE, preserves local and some
            global structure, but does so with better preservation of global
            topology. The fact that the groups remain separate in this
            projection suggests: The use of a Convoluted Neural Network
            (CNN) to train a prediction model.
            """
        ))

    st.divider()

    # --- Classification Report ---
    st.subheader("Classification Reports")
    col1, col2 = st.columns(2)

    train_report = load_csv(
        f"outputs/{version}/classification_report_train.csv")
    test_report = load_csv(
        f"outputs/{version}/classification_report_test.csv")

    with col1:
        st.markdown("#### Train Set")
        if train_report is not None:
            st.dataframe(train_report, height=300)

    with col2:
        st.markdown("#### Test Set")
        if test_report is not None:
            st.dataframe(test_report, height=300)

    center_component(lambda: st.info(
        """
        The model demonstrates exceptional performance, achieving nearly
        perfect precision, recall, and F1-scores across the training,
        validation, and test sets, with 100% accuracy on both the validation
        and test sets. These results suggest that the model generalizes
        extremely well, indicating it has likely captured the underlying
        patterns in the data without overfitting.
        """
    ))

    st.divider()

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    col1, col2, col3 = st.columns(3)
    with col1:
        display_image_centered(
            f"outputs/{version}/confusion_matrix_test.png",
            "Test Set Confusion Matrix", width=350)
    with col2:
        display_image_centered(
            f"outputs/{version}/confusion_matrix_train.png",
            "Train Set Confusion Matrix", width=350)
    with col3:
        display_image_centered(
            f"outputs/{version}/confusion_matrix_val.png",
            "Validation Set Confusion Matrix", width=350)

    center_component(lambda: st.info(
        """
        Our model performs exceptionally well across all sets,
        achieving perfect accuracy on both the validation and test
        sets, while slightly underperforming on the training set.
        This could suggest strong generalization, though perfect
        accuracy on unseen data may also point to potential data
        leakage or overly simplistic test examples. Further evaluation
        with more diverse test data is recommended to confirm
        robustness.
        """
    ))

    st.divider()

    # --- Learning Curves ---
    st.subheader("Model Learning Curves")
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            f"outputs/{version}/model_training_acc.png",
            caption="Accuracy Over Epochs", use_container_width=True)
    with col2:
        st.image(
            f"outputs/{version}/model_training_losses.png",
            caption="Loss Over Epochs", use_container_width=True)

    center_component(lambda: st.info(
        """
        **Accuracy Curve** - The training and validation accuracy curves
        show a steady and rapid increase within the first few epochs,
        with both eventually converging above 99%. This indicates that
        the model has learned to classify the data very effectively and
        is capable of fitting both the training and validation sets well.

        **Loss Curve** - Both training and validation losses exhibit a
        clear downward trend, with training loss reaching near zero and
        validation loss staying consistently low. The absence of
        significant spikes or divergence suggests the model has not
        overfit to the training data.

        **Generalization** - The narrow and stable gap between training
        and validation accuracy/loss across epochs is a strong indicator
        of good generalization. The model performs well not only on
        training data but also on unseen validation samples — a key
        trait for real-world deployment.
        """
    ))

    st.divider()

    # --- ROC Curve ---
    st.subheader("ROC Curve")
    display_image_centered(
        f"outputs/{version}/roc_curve.png", "ROC Curve", width=700)

    center_component(lambda: st.info(
        """
        AUC = 1.0 across all sets means the model perfectly distinguishes
        between the classes in each set. This suggests excellent
        generalization and the model is likely very well-suited to the
        problem (perhaps due to distinct patterns between classes).\n

        However, perfect AUC scores could also indicate:\n

        - Overfitting (but it's not likely as the model performs
        better on test and validation sets than on the training set)\n
        - Data leakage — if information from the test/val sets
        unintentionally influences training (e.g., during
        preprocessing).\n
        - Or we have an overly simple problem — like when diseased
        vs. healthy leaves have extremely obvious visual differences.\n

        The random score of 0.5 shows that we don't have any error
        in the allocation of classes.
        """
    ))

    st.divider()

    # --- Final Metrics ---
    st.subheader("Final Test Metrics")
    test_eval = load_test_metrics(version)
    if test_eval and len(test_eval) == 2:
        df = pd.DataFrame({
            "Metric": ["Loss", "Accuracy"],
            "Value": [round(test_eval[0], 4), round(test_eval[1], 4)]
        })
        center_component(lambda: st.table(df))
    else:
        st.error("Evaluation metrics not available or improperly formatted.")

    # --- Summary ---
    st.subheader("Summary & Deployment Readiness")
    st.success(
        """
        Model shows excellent generalization, stable training,
        and reliable predictions. It is ready for deployment
        and testing in early mildew detection in the field.
        """
    )
    st.markdown(
        "See [README](https://github.com/N4v1ds0n/PP5-predictive_analysis"
        "/blob/main/README.md) for more.")
