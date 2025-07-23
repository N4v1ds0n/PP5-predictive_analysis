from pathlib import Path
import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dropout, Dense, InputLayer
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc


def get_split_counts(data_dir,
                     splits=('train', 'validation', 'test'),
                     labels=('healthy', 'diseased')):
    """
    Get image counts for each class per split.
    """
    data = {'Set': [], 'Label': [], 'Frequency': []}
    data_dir = Path(data_dir)

    for split in splits:
        for label in labels:
            path = data_dir / split / label
            count = len(os.listdir(path)) if path.exists() else 0
            data['Set'].append(split)
            data['Label'].append(label)
            data['Frequency'].append(count)
            print(f"* {split} - {label}: {count} images")

    return pd.DataFrame(data)


def plot_split_distribution(df_freq, save_path=None):
    """
    Visualize class distribution across dataset splits.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_freq, x='Set', y='Frequency', hue='Label')
    plt.title("Dataset Class Distribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_augmented_samples_grid(generators_dict, class_indices):
    """
    Plots a 3x2 grid with one 'healthy' and one 'diseased' sample from each
    generator.

    Parameters:
        generators_dict (dict): Dictionary of {name: generator}
        class_indices (dict): Mapping of class names to integers
        (e.g., {'diseased': 0, 'healthy': 1})
    """
    # Reverse class_indices for integer -> label mapping
    label_map = {v: k for k, v in class_indices.items()}

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    axes = axes.flatten()

    idx = 0
    for set_name, generator in generators_dict.items():
        collected = {0: None, 1: None}  # store one image per class
        while any(v is None for v in collected.values()):
            images, labels = next(generator)
            for img, label in zip(images, labels):
                # Handle both scalar and one-hot labels
                if hasattr(label, '__len__') and not isinstance(label, str):
                    label_id = int(np.argmax(label))  # categorical
                else:
                    label_id = int(label)  # binary

                if label_id in collected and collected[label_id] is None:
                    collected[label_id] = img

                if all(v is not None for v in collected.values()):
                    break
        # Plot images in fixed class order: 0 first (usually diseased), then 1
        for class_id in sorted(collected):
            ax = axes[idx]
            ax.imshow(collected[class_id])
            ax.set_title(f"{set_name.title()}\nLabel: {label_map[class_id]}")
            ax.axis("off")
            idx += 1

    plt.tight_layout()
    plt.suptitle("Augmented & Rescaled Samples (Train / Validation / Test)",
                 fontsize=16, y=1.05)
    plt.show()


def build_custom_cnn(
    shape=(128, 128, 3),
    num_classes=1,
    base_filters=32,
    conv_blocks=3,
    dense_units=64,
    dropout_rate=0.3,
    learning_rate=1e-4
):
    model = Sequential()
    model.add(InputLayer(shape=shape))

    # Add convolutional blocks
    filters = base_filters
    for _ in range(conv_blocks):
        model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D())
        filters *= 2  # Double filters per block

    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(num_classes,
                    activation='sigmoid' if num_classes == 1 else 'softmax'))

    if num_classes == 1:
        loss_fn = 'binary_crossentropy'
    else:
        loss_fn = 'categorical_crossentropy'

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model


def plot_training_curves(history,
                         output_dir=None,
                         save_figures=True,
                         dpi=150,
                         style="whitegrid"):
    """
    Plots loss and accuracy curves from model training history.

    Parameters:
        history (tensorflow.keras.callbacks.History or dict): History object or dict
        with 'loss', 'val_loss', etc. output_dir (str, optional): Path to
        save figures. If None and save_figures=True, saves to current
        directory.
        save_figures (bool): Whether to save plots as image files.
        dpi (int): Resolution of saved figures.
        style (str): Seaborn style (e.g., "whitegrid", "darkgrid").

    Returns:
        None
    """
    if isinstance(history, dict):
        history_df = pd.DataFrame(history)
    else:
        history_df = pd.DataFrame(history.history)

    sns.set_style(style)

    # Plot Loss
    plt.figure(figsize=(8, 5))
    history_df[['loss', 'val_loss']].plot(style='.-', ax=plt.gca())
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if save_figures:
        save_path = os.path.join(output_dir or ".",
                                 "model_training_losses.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    history_df[['accuracy', 'val_accuracy']].plot(style='.-', ax=plt.gca())
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    if save_figures:
        save_path = os.path.join(output_dir or ".", "model_training_acc.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()


def collect_predictions(generator, model):
    x_data = []
    y_true = []
    y_probs = []


    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        x_data.extend(x_batch)
        y_true.extend(y_batch)

        probs = model.predict(x_batch, verbose=0)
        # Binary classification (sigmoid) → shape: (batch_size, 1)
        if probs.shape[-1] == 1:
            probs = probs.ravel()
        # Categorical classification (softmax) → shape: (batch_size, num_classes)
        # keep as-is

        y_probs.extend(probs)

    return np.array(x_data), np.array(y_true), np.array(y_probs)


def preprocess_labels_and_probs(y_true, y_probs, is_binary):
    if not is_binary:
        # Handle softmax output
        if y_probs.ndim == 2 and y_probs.shape[1] > 1:
            y_probs = y_probs[:, 1]  # Prob of class 1
        else:
            raise ValueError("Expected softmax probs with shape (N, num_classes)")

        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        elif y_true.ndim == 1:
            # Already class indices
            pass
        else:
            raise ValueError("Unexpected shape for y_true in multiclass classification")
    else:
        y_probs = y_probs.ravel()
        y_true = y_true.astype(int).ravel()

    return y_true, y_probs


def plot_roc(y_true_train, y_probs_train,
             y_true_val, y_probs_val,
             y_true_test, y_probs_test,
             is_binary, output_dir):
    
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import os

    y_true_train, y_probs_train = preprocess_labels_and_probs(y_true_train, y_probs_train, is_binary)
    y_true_val, y_probs_val = preprocess_labels_and_probs(y_true_val, y_probs_val, is_binary)
    y_true_test, y_probs_test = preprocess_labels_and_probs(y_true_test, y_probs_test, is_binary)

    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_probs_train)
    fpr_val, tpr_val, _ = roc_curve(y_true_val, y_probs_val)
    fpr_test, tpr_test, _ = roc_curve(y_true_test, y_probs_test)

    auc_train = auc(fpr_train, tpr_train)
    auc_val = auc(fpr_val, tpr_val)
    auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f"Train AUC = {auc_train:.2f}", color="blue")
    plt.plot(fpr_val, tpr_val, label=f"Validation AUC = {auc_val:.2f}", color="orange")
    plt.plot(fpr_test, tpr_test, label=f"Test AUC = {auc_test:.2f}", color="green")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (0.5)")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()

    save_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"ROC curve saved to {save_path}")


def create_dataframe_from_folders(data_dir):
    """
    Create a DataFrame with columns: filepath, label
    """
    filepaths = []
    labels = []

    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            filepaths.append(fpath)
            labels.append(label)

    return pd.DataFrame({'filename': filepaths, 'class': labels})


def run_kfold(data_dir, k=5, image_size=(128, 128), batch_size=32, epochs=10, mode='softmax'):
    assert mode in ['sigmoid', 'softmax'], "Mode must be 'sigmoid' or 'softmax'"

    df = create_dataframe_from_folders(data_dir)
    df = df[df['class'].isin(['diseased', 'healthy'])]

    if mode == 'sigmoid':
        # Expect exactly 2 classes
        unique_classes = df['class'].unique()
        if len(unique_classes) != 2:
            raise ValueError("Sigmoid mode requires exactly 2 classes.")
        class_mode = 'binary'
        num_classes = 1
    else:
        class_mode = 'categorical'
        num_classes = len(df['class'].unique())

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    training_times = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n🌀 Fold {fold+1}/{k} | Mode: {mode}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        datagen = ImageDataGenerator(rescale=1./255)

        train_gen = datagen.flow_from_dataframe(
            train_df,
            x_col='filename',
            y_col='class',
            target_size=image_size,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=True
        )

        val_gen = datagen.flow_from_dataframe(
            val_df,
            x_col='filename',
            y_col='class',
            target_size=image_size,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=False
        )

        model = build_custom_cnn(shape=(*image_size, 3), num_classes=num_classes)

        start_time = time.time()
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            verbose=1
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        training_times.append(elapsed)
        print(f"⏱️ Fold {fold+1} training time: {elapsed:.2f} seconds")

        val_loss, val_acc = model.evaluate(val_gen)
        print(f"✅ Fold {fold+1} - Val Accuracy: {val_acc:.4f}")
        fold_results.append({
            "fold": fold + 1,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "history": history.history,
            "train_time_sec": elapsed
        })

    print("\n📊 K-Fold Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}: Val Acc = {result['val_acc']:.4f}, Val Loss = {result['val_loss']:.4f}, Train Time = {result['train_time_sec']:.2f}s")
        
    accs = [r["val_acc"] for r in fold_results]
    print(f"\nAverage Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"⏱️ Average Training Time per Fold: {np.mean(training_times):.2f} seconds")
    print(f"🕒 Total Training Time: {np.sum(training_times):.2f} seconds")