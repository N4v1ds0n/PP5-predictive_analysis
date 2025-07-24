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
        "Explore the model’s performance across metrics and visualizations.")

    # --- Class Distribution ---
    st.subheader("Dataset Split & Class Distribution")
    center_component(lambda: st.image(
        f"outputs/{version}/labels_distribution.png",
        caption="Class Distribution in Train, Validation, and Test Sets",
        use_container_width=True
    ))
    center_component(lambda: st.info(
        "- Balanced distribution reduces model bias."
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
            - Partial class separation suggests overlapping but
            distinguishable features.
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
            - t-SNE reveals better-defined local groupings, especially
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
            - The 'diseased' and 'healthy' samples form distinct clusters,
            with minimal overlap. This implies your features contain highly
            discriminative information.
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
        "- Strong performance and generalization seen in F1 scores."
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
        "- Mostly correct predictions, minor false positives."
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
        "- High convergence, minimal overfitting = solid training dynamics."
    ))

    st.divider()

    # --- ROC Curve ---
    st.subheader("ROC Curve")
    display_image_centered(
        f"outputs/{version}/roc_curve.png", "ROC Curve", width=700)

    center_component(lambda: st.info(
        "- AUC = 1 → perfect discriminative ability."
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
        and reliable predictions. It is ready for real-world deployment
        in early mildew detection.
        """
    )
    st.markdown(
        "See [README](https://github.com/N4v1ds0n/PP5-predictive_analysis"
        "/blob/main/README.md) for more.")
