import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.dashboard.data_management import load_pkl_file


def plot_predictions_probabilities(pred_proba, pred_class, chart_key=None):
    """
    Plot prediction probability results with optional unique Streamlit key.
    """
    prob_per_class = pd.DataFrame(
        data=[0, 0],
        index={'diseased': 0, 'healthy': 1}.keys(),
        columns=['Probability']
    )

    prob_per_class.loc[pred_class] = pred_proba
    for x in prob_per_class.index.to_list():
        if x not in pred_class:
            prob_per_class.loc[x] = 1 - pred_proba
    prob_per_class = prob_per_class.round(3)
    prob_per_class['Diagnostic'] = prob_per_class.index

    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y='Probability',
        range_y=[0, 1],
        width=600, height=300,
        template='seaborn'
    )

    st.plotly_chart(fig, key=chart_key or f"chart_{np.random.randint(1_000_000)}")


def resize_input_image(img, version):
    """
    Resizes and normalizes input image based on training data dimensions.

    Args:
        img: PIL Image instance.
        version: Model version to load shape from.

    Returns:
        Numpy array of shape (1, height, width, channels).
    """
    image_shape = load_pkl_file(file_path=f"outputs/{version}/img_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)

    my_image = np.expand_dims(img_resized, axis=0)/255

    return my_image


def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction over live images
    """
    model = load_model(f"outputs/{version}/mildew_detector.h5")

    pred_proba = model.predict(my_image)[0, 0]

    target_map = {v: k for k, v in {'diseased': 0, 'healthy': 1}.items()}
    pred_class = target_map[pred_proba > 0.5]
    if pred_class == target_map[0]:
        pred_proba = 1 - pred_proba

    st.write(
        f"The predictive analysis indicates the sample leaf is "
        f"**{pred_class.lower()}** with a probability of **{pred_proba:.2%}**.")

    return pred_proba, pred_class


def load_test_evaluation(version):
    return load_pkl_file(f'outputs/{version}/eval.pkl')
