import streamlit as st
import pandas as pd
import base64
import pickle
from pathlib import Path
import plotly.express as px

def load_pickle(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load pickle: {file_path} — {e}")
        return None


def load_test_metrics(version):
    result = load_pickle(f"outputs/{version}/eval.pkl")
    if isinstance(result, dict):
        return list(result.values())
    return result


def load_csv(file_path, error_msg="CSV file not found."):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(error_msg)
        return None


def encode_image_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        st.error(f"Failed to encode image: {path} — {e}")
        return None


def display_image_centered(path, caption, width=700):
    b64 = encode_image_to_base64(path)
    if b64:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{b64}" width="{width}">
                <p style="font-size:14px;">{caption}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def center_component(callable_obj):
    cols = st.columns([1, 6, 1])
    with cols[1]:
        callable_obj()


def load_drt_figure(pca_df):
    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color="Label",
        title="PCA Projection of Cherry Leaves",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        width=700, height=500
    )
    return fig