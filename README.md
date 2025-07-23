# [Mildew Detector](https://detect-mildew-1e5e3ef17076.herokuapp.com/)
(Developer: Damian Droste)

[![GitHub commit activity](https://img.shields.io/github/commit-activity/t/n4v1ds0n/PP5-predictive_analysis)](https://www.github.com/n4v1ds0n/PP5-predictive_analysis/commits/main)
[![GitHub last commit](https://img.shields.io/github/last-commit/n4v1ds0n/PP5-predictive_analysis)](https://www.github.com/n4v1ds0n/PP5-predictive_analysis/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/n4v1ds0n/PP5-predictive_analysis)](https://www.github.com/n4v1ds0n/PP5-predictive_analysis)

![amiresponsive-screenshot](./assets/readme_img/amiresponsive.png)
[Page on AmIResponsive](https://ui.dev/amiresponsive?url=https://detect-mildew-1e5e3ef17076.herokuapp.com/)

...for your daily transactions, cashflow clarity, and categorical control.

Budget Ledger is a **terminal-based personal finance tracker** built in Python 3, 
designed to help users manage their income and expenses through an intuitive CLI 
interface.

[Link to live dashboard deployment (Heroku)](https://detect-mildew-1e5e3ef17076.herokuapp.com/)

---

## Table of Contents
1. [Dataset Description](#dataset-description)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and validation](#hypothesis-and-validation)
4. [Model Architecture](#model-architecture)
6. [Implementation of the Business Requirements](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
7. [ML Business case](#ml-business-case)
8. [Dashboard design](#dashboard-design-streamlit-app-user-interface)
9. [CRISP DM Process](#the-process-of-cross-industry-standard-process-for-data-mining)
9. [Testing](#testing)
10. [Bugs](#bugs)
11. [Deployment](#deployment)
12. [Technologies used](#technologies-used)
13. [Credits](#credits)

## Dataset Description

This dataset contains 4,208 high-quality images of individual cherry leaves, each classified as either healthy or infected with [powdery mildew](https://en.wikipedia.org/wiki/Powdery_mildew) — a widespread fungal disease affecting many plant species, including bitter cherry trees (Prunus spp.), the client's primary crop.

All images were captured under consistent, uniform conditions: the leaves are centered against a grey, grainy background, which provides strong visual contrast and supports reliable image analysis.
The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).

## Business Requirements

Farmy & Foods, a company in the agricultural sector, requested the development of a machine learning system capable of detecting powdery mildew — a fungal disease affecting cherry trees — from images of individual leaves.

The company currently relies on a manual inspection process: an employee examines several leaves per tree and applies treatment if mildew is found. This method is time-consuming (30 minutes per tree) and not scalable, given their thousands of cherry trees spread across multiple farms nationwide.

The client aims to automate this process using an AI-powered visual inspection system that can instantly identify whether a leaf is healthy or infected, thereby significantly reducing labor time and enabling early intervention to protect crop quality.

For full context, see the original business interview.
✅ Summary of Business Requirements:

    Visually differentiate healthy cherry leaves from those infected with powdery mildew.

    Automatically predict the health status of a leaf based on an image.

    Provide interpretable prediction reports for the examined leaves.


## Hypotheses and Validation

Hypothesis 1: Visual Signs of Infection Are Detectable

Hypothesis 2: A Sigmoid Activation Is Suitable for Binary Classification

Hypothesis 3: A CNN Can Reliably Detect Powdery Mildew

Hypothesis 4: The Model Can Generalize Beyond the Training Domain

### Hypothesis 1: Visual Signs of Infection Are Detectable

Statement: Powdery mildew creates distinct visual patterns—such as white, powdery spots—that can be reliably captured in leaf imagery.

Rationale: Since the disease alters the leaf's surface texture and coloration, it should be visually distinguishable from healthy leaves, particularly in a controlled imaging environment.

Validation:

    Manual inspection of the dataset confirmed consistent and distinct mildew markings.

    Exploratory Data Analysis (EDA) showed clustering of image features in principal component space, suggesting separability.

    Sample predictions were cross-checked with labeled images to confirm visual cues were being leveraged.


### Hypothesis 3: A Sigmoid Activation Is Suitable for Binary Classification

Statement: Using a sigmoid activation function in the output layer is appropriate for this binary classification task (healthy vs. infected).

Rationale: A sigmoid function outputs a probability between 0 and 1, making it ideal for binary outcomes. Unlike softmax, which is better suited for multi-class problems, sigmoid offers direct interpretability for two-class tasks.

Validation:

    Comparison of sigmoid vs. softmax output layers showed sigmoid resulted in simpler interpretation without sacrificing performance.

    The model produced well-calibrated probability scores.

    No class imbalance issues requiring threshold tuning were observed (due to balanced dataset).


### Hypothesis 2: A CNN Can Reliably Detect Powdery Mildew

Statement: A Convolutional Neural Network (CNN) can learn the relevant spatial patterns in the leaf images to distinguish between healthy and infected samples with high accuracy.

Rationale: CNNs are well-suited for image classification tasks as they extract hierarchical visual features such as edges, textures, and shapes.

Validation:

    The model achieved strong performance metrics (e.g., accuracy, precision, recall) on the validation and test sets.

    Learning curves showed convergence without significant overfitting.

    Grad-CAM heatmaps and model interpretability tools indicated focus on mildew regions.



### Hypothesis 4: The Model Can Generalize Beyond the Training Domain

Statement: A model trained on clean, uniformly staged leaf images can generalize effectively to real-world conditions where leaves appear in varied lighting, angles, and backgrounds.

Rationale: For the model to be practically useful in the field, it must handle natural variability — including non-isolated leaves, overlapping foliage, crumpling, varying lighting conditions, or inconsistent focus — none of which are present in the curated training dataset.

Validation:

    Real-world test images (taken in natural farm environments) were used to evaluate model robustness.

    Performance on these out-of-distribution (OOD) images was significantly lower, revealing overreliance on artificial features like consistent background, brightness, and contrast.

    Attempts to compensate using Test-Time Augmentation (TTA) and extensive data augmentation yielded only marginal improvements, underscoring the importance of representative training data.

Conclusion:
This hypothesis did not hold under current conditions. The model suffers from poor extrapolation due to the highly uniform dataset. For deployment-ready performance, substantial improvements must be made through:

    Dataset diversification (adding naturally captured leaf images),

    Domain adaptation techniques, or

    Synthetic data generation that mimics field conditions.

## Model Architecture

### Project Goal  
This project aims to detect **powdery mildew infections** in cherry leaves using a custom convolutional neural network (CNN). The dataset consists of uniformly captured, centered cherry leaf images, labeled as either *healthy* or *infected*. Given the clearly distinguishable visual cues and the binary nature of the classification task, a CNN is an ideal choice.

---

### Custom CNN Model Summary

The model was built using Keras' `Sequential` API and follows a simple yet effective structure:

- **Input**: RGB images resized to `128x128x3`
- **Convolutional Blocks**: 3 blocks of increasing filters (`32 → 64 → 128`) using `Conv2D` with ReLU and `MaxPooling2D` to downsample
- **Fully Connected Layer**: A single dense layer with 64 neurons
- **Regularization**: Dropout of 30% after flattening to reduce overfitting
- **Output**: 1 neuron with `sigmoid` activation for binary classification
- **Loss Function**: `binary_crossentropy`
- **Optimizer**: Adam with learning rate `1e-4`

> If the number of classes is changed (e.g., for multiclass disease types), the model automatically switches to a `softmax` output with `categorical_crossentropy`.

```python
# Pseudocode overview
model = build_custom_cnn(
    shape=(128, 128, 3),
    num_classes=1,
    base_filters=32,
    conv_blocks=3,
    dense_units=64,
    dropout_rate=0.3,
    learning_rate=1e-4
)
```

### ⚙️ Why This Architecture


| Component |	Justification |
|---|---|
|Conv2D (3x3 kernels)	| Efficient at capturing local features such as mildew texture and shape.|
|MaxPooling	| Reduces spatial size and computation, helps with feature generalization. |
|ReLU activation |	Speeds up convergence and avoids vanishing gradients. |
|Dropout (0.3)	| Helps regularize the model, important given the relatively small dataset size. |
|Sigmoid output	| Suitable for binary classification — interprets output as infection probability. |
|Adam optimizer	| Adaptive and widely used; balances speed and stability.  |


### Sigmoid vs. Softmax in Context

There is some debate between using sigmoid vs. softmax in binary classification:

Sigmoid + 1 output node:

- Interprets the output as the probability of being in class 1.

- Works well when classes are mutually exclusive.

- Computationally cheaper and more direct.

Softmax + 2 output nodes:

- Produces a full probability distribution.

- Technically overkill for binary classification but may help in multiclass extensions.

- Some argue it performs better in gradient-based optimization.

In this project, we use sigmoid, treating the task as binary classification with binary_crossentropy loss, which is appropriate and efficient.

### Training Configuration

    Epochs: 25 (with early stopping)

    Callbacks:

        EarlyStopping to prevent overfitting

        ModelCheckpoint to save the best model

    Metrics Tracked: Accuracy, validation loss

```python
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('path/to/save/mildew_detector.h5', save_best_only=True)
]

history = model.fit(
    train_set,
    validation_data=validation_set,
    epochs=25,
    callbacks=callbacks,
    verbose=1
)
```

## ML Business Case: Mildew Detector


### Objective

Develop a machine learning model to automatically classify whether a cherry leaf is infected with powdery mildew based on image data provided by the Farmy & Foody company. This is a supervised learning problem, framed as binary image classification.

### Problem Framing

- Type: Supervised Learning

- Task: Binary classification (Healthy vs. Diseased)

- Label Type: Single-label, mutually exclusive

- Input: RGB image of a cherry leaf

- Output:

    - A binary flag indicating infection status

    - A confidence score (probability between 0–1)

### Success Criteria

- Accuracy target: ≥ 87% on the test set

- Inference mode: Real-time / on-demand (no batch inference)

- Deployment target: Mobile/web application for field use by farmers

### Business Rationale

Currently, disease detection relies on manual leaf inspection, where a farmer spends ~30 minutes per tree, sampling and visually inspecting leaves. This process is:

- Time-consuming

- Prone to human error

- Scalable only with increased labor costs

By automating the diagnosis via a mobile app, we can offer:

- Faster diagnosis

- Consistent accuracy

- Reduced labor and inspection costs

- Scalable disease monitoring

### Data Source

- Dataset: Cherry Leaf Disease Dataset on Kaggle

- Provided by: Farmy & Foody

- Image Count: 4,208 cherry leaf images, 2 classes: Healthy and Powdery Mildew Infected, Format: 128x128 RGB JPEG images


## Debugging

### Fixed Bugs

| Bug | Fix |
|---|---|
|**Confusion matrix for model was off compared to performance statistics**|Wrote 'collect predictions functions that stores predictions in a list to avoid errors|
|**Confusion matrix for model was off compared to performance statistics**|Wrote 'collect predictions functions that stores predictions in a list to avoid errors|

### Unfixed Bugs

None known at time of submission.

---
## Deployment

The project is hosted on GitHub and deployed via Heroku.

### Create an App on Heroku

To deploy this project on Heroku, follow these steps:

- Ensure a requirements.txt file is present in the GitHub repository to specify all Python dependencies.

- Include a runtime.txt file to define the Python version (e.g., python-3.11.8) supported by the Heroku-20 stack.

- Push the latest code changes to GitHub.

- Log in to your Heroku dashboard and create a new app.

- Click Create New App, assign a unique name, and choose a region.

- Under the Settings tab, add the heroku/python buildpack.

- In the Deploy tab:

    - Choose GitHub as the deployment method.

    - Connect to your GitHub account and select the repository for this project.

- Select the branch to deploy, then click Deploy Branch.

- Optionally, enable Automatic Deploys for continuous deployment or use Manual Deploy.

- Monitor the build logs as dependencies are installed and the app is built.

- Once deployed, the app will be available at a URL similar to: https://your-app-name.herokuapp.com/

- If the slug size exceeds the limit, add non-essential files to a .slugignore file to reduce build size. This might include packages you are only using in your notebooks, you can remove them from your requirements file and install them from a notebook cell.

## Forking the Repository

If you want to fork this project repository you are very welcome to do so:

- Go to the GitHub Repository.

- Log into your GitHub account.

- Click the Fork button at the top-right corner.

- Select the destination (e.g., your own GitHub account).

- You now have a personal copy of the repository to modify without affecting the original.

## Cloning the Repository Locally

If you want to work on your fork locally, you can clone the project to your machine:

- Navigate to the GitHub Repository.

- Click the Code button and choose one of the cloning methods (HTTPS, GitHub CLI, ZIP).

- To clone via HTTPS:

    - Copy the URL under Clone with HTTPS

    - Open a terminal or Git Bash

    - Navigate to the desired directory

    - Run:

    git clone https://github.com/*yourhandle*/PP5-predictive_analysis.git

- Press Enter and wait for the process to complete.

- For additional help, refer to the GitHub cloning guide.

---

## Technologies used

### Platforms
- [Heroku](https://en.wikipedia.org/wiki/Heroku) Deployment
- [Jupiter Notebook](https://jupyter.org/) Data acquisition, exploration and training area
- [Kaggle](https://www.kaggle.com/) Data source
- [GitHub](https://github.com/): Repo storage
- [VSCode](https://code.visualstudio.com/) Local IDE


### Languages
- [Python](https://www.python.org/)
- [Markdown](https://en.wikipedia.org/wiki/Markdown)
  
### Main Data Analysis and Machine Learning Libraries
<pre>
numpy==1.26.1
pandas==2.1.1
matplotlib==3.8.0
seaborn==0.13.2
plotly==5.17.0
Pillow==10.0.1
streamlit==1.40.2
joblib==1.4.2
tensorflow-cpu==2.16.1
scikit-learn==...
</pre>

## Credits




### Acknowledgements

Thanks to [Code Institute](https://codeinstitute.net/global/), especially Kay Welfare and my mentor Mo Shami. 

### Deployed version at [Mildew Detector](https://detect-mildew-1e5e3ef17076.herokuapp.com/)