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
2. [ML Business case](#ml-business-case)
3. [Business Requirements](#business-requirements)
4. [Implementation of Business Requirements](#Implementation-of-Business-Requirements)
5. [Hypothesis and validation](#hypotheses-and-validation)
6. [User Stories](#user-stories)
7. [Dashboard design](#dashboard-design)
8. [CRISP DM](#crisp-dm-approach)
9. [Model Architecture](#model-architecture)
10. [Testing](#testing)
11. [Bugs](#bugs)
12. [Deployment](#deployment)
13. [Technologies used](#technologies-used)
14. [Credits](#credits)

## Dataset Description

This dataset contains 4,208 high-quality images of individual cherry leaves, each classified as either healthy or infected with [powdery mildew](https://en.wikipedia.org/wiki/Powdery_mildew) — a widespread fungal disease affecting many plant species, including bitter cherry trees (Prunus spp.), the client's primary crop.

All images were captured under consistent, uniform conditions: the leaves are centered against a grey, grainy background, which provides strong visual contrast and supports reliable image analysis.
The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).
---

## ML Business Case

The Mildew Detector

### Problem statement

Farmy & Foods, a major player in the agricultural sector, faces operational challenges in managing powdery mildew across its cherry plantations. The current reliance on manual inspection—taking approximately 30 minutes per tree—is time-intensive and unsustainable at scale. As a result, delayed detection can hinder timely treatment, negatively affecting both yield and crop quality.


### Objective

Develop a machine learning model to automatically classify whether a cherry leaf is infected with powdery mildew based on image data provided by the Farmy & Foody company. This is a supervised learning problem, framed as binary image classification. The goal:

- reduce inspection times through automation
- improve accuracy compared to manual inspection
- make inspection scaleable for larger areas

### Problem Framing

- Type: Supervised Learning

- Task: Binary classification (Healthy vs. Diseased)

- Label Type: Single-label, mutually exclusive

- Input: RGB image of a cherry leaf

- Output:

    - A binary flag indicating infection status

    - A confidence score (probability between 0–1)

### Success Criteria

- Accuracy target: ≥ 97% on the test set

- Inference mode: Real-time / on-demand

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
---

## Business Requirements

Farmy & Foods, a company in the agricultural sector, requested the development of a machine learning system capable of detecting powdery mildew — a fungal disease affecting cherry trees — from images of individual leaves.

The company currently relies on a manual inspection process: an employee examines several leaves per tree and applies treatment if mildew is found. This method is time-consuming (30 minutes per tree) and not scalable, given their thousands of cherry trees spread across multiple farms nationwide.

The client aims to automate this process using an AI-powered visual inspection system that can instantly identify whether a leaf is healthy or infected, thereby significantly reducing labor time and enabling early intervention to protect crop quality.

Summary of Business Requirements:

- Visually differentiate healthy cherry leaves from those infected with powdery mildew.

- Automatically predict the health status of a leaf based on an image.

- Provide interpretable prediction reports for the examined leaves.


## Implementation of Business Requirements

### Business Requirement 1: Visual Differentiation

Key result:
Differentiate healthy from diseased leaves through image analysis.

Data processing Tasks:

- Image normalization
- PCA was used to check feature separability between healthy and infected leaves.
- Check balancing and verify class distribution to avoid biased learning.

Outcome: Ensured the dataset was visually and statistically robust for effective classification.


### Business Requirement 2: Disease Detection & Classification by ML model

Key result:
Train an ML model capable of classifying cherry leaves as healthy or diseased.

Data preparation and ML training tasks:

- Training set augmentation for model robustnesss.
- Resizing images for consistency across the set.
- Model Tuning (Sigmoid or Softmax)
- Evaluate performance (track accuracy, precision, recall, F1-score, and ROC curves).
- Check for overfitting by compared training vs. validation vs. test accuracy.

Outcome: Trained an excellingly performing and efficient model based on Softmax that meets the needs of the business requirement.


### Business Requirement 3: Model deployment with clear documentation and prediction tool providing a report


ML performance visualization Tasks:


- Confusion Matrices and other preformance markers visualize performance over all sets
- Generate prediction reports showing predicted class labels, confidence scores.
- Deploy streamlit dashboard to provide findings and functionalities.

Outcome: The deployed model provides insights and functionality for users of the dashboard.


Conclusion:

Each task perforem in this project was tailored to fulfill part of the business requirements
- Data processing and visualisation confirmed data quality
- Augmentation, and preprocessing as well as model selection and evaluation ensure a well trained model.
- performance evaluation, reporting and interactivity create an insightful and functional deployment.



## Hypotheses and Validation

Hypothesis 1: Visual signs of mildew infection are detectable

Hypothesis 2: A softmax output neuron is the better choice despite of binary dataset

Hypothesis 3: A CNN can reliably detect powdery mildew

Hypothesis 4: The model might have difficulties to Generalize Beyond the Training Domain

### Hypothesis 1: Visual signs of infection are detectable

Statement: Powdery mildew creates distinct visual patterns—such as white, powdery spots—that can be reliably captured in leaf imagery.

Rationale: Since the disease alters the leaf's surface texture and coloration, it should be visually distinguishable from healthy leaves, particularly in a controlled imaging environment.

Validation:

- Manual inspection of the dataset confirmed consistent and distinct mildew markings.

- Create averaged images for each class and a standard deviation image -> differences detectable

- Exploratory Data Analysis (EDA) showed clustering of image features in principal component space, suggesting separability.
  - PCA, t-SNE and UMAP showed increasingly distinguishable clusters, which suggests good class seperability.

Conclusion:
Mildew-infected leaves exhibit localized brightness and textural noise. The PCA revealed moderate but usable class separation, t-SNE reveals better-defined local groupings, especially for the healthy class, which appears more compact and clustered toward the lower part of the graph. In the UMAP, diseased and healthy samples form distinct clusters, with some overlap. This implies features contain highly discriminative information.



### Hypothesis 2: A softmax output neuron is the better choice despite of binary dataset

Statement: Although for a binary classification problem the single output neuron setup with a sigmoid function returning 1 or 0 is the default approach. However there are indications, that especially for gradient based optimization a categorical approach with two softmax Neurons might be better. This is because categorical loss can sometimes offer better learning curves due to gradient stability.

Rationale: A sigmoid function outputs a probability between 0 and 1, making it ideal for binary outcomes. But softmax, which is usually better suited for multi-class problems, might lead to more accurate results due to better gradient stability.

Validation:

- Comparison of sigmoid vs. softmax output layers by running a ffold for each setup to compare accuracy and time consumption

- The models produced well-calibrated probability scores.

Conclusion:
The kfold duel showed that the softmax approach is not only reaching a higher accuracy, but also training faster on average, making it the objectively better choice for our model.
    


### Hypothesis 3: A CNN can reliably detect powdery mildew

Statement: A Convolutional Neural Network (CNN) can learn the relevant spatial patterns in the leaf images to distinguish between healthy and infected samples with high accuracy.

Rationale: CNNs are well-suited for image classification tasks as they extract hierarchical visual features such as edges, textures, and shapes.

Validation:

- The model achieved strong performance metrics (e.g., accuracy, precision, recall) on the validation and test sets.

- Learning curves showed convergence without significant overfitting.

- Confusion matrices, classification report, ROC curce and AUC score showed excelling results the AUC score was 1 which is a perfect score

Conclusion:
The model can make very reliable predictions on the test data and distinguish between healthy and mildew-infected leaves.


### Hypothesis 4: The model might have difficulties to Generalize Beyond the Training Domain

Statement: A model trained on clean, uniformly staged leaf images might have problems to effectively extraploate to to real-world conditions where leaves appear in varied lighting, angles, and backgrounds. If the leaves are not photographed under the same conditions the model might be misclassifying.

Rationale: For the model to be practically useful in the field, it should handle natural variability — including non-isolated leaves, overlapping foliage, crumpling, varying lighting conditions, or inconsistent focus — none of which are present in the curated training dataset. Otherwise the model will still be useful, but much more difficult to apply.

Validation:

- A set of real-world test images (generated by AI, but realistic) were used to evaluate model robustness.

- Performance on these out-of-distribution (OOD) images was significantly lower, revealing overreliance on artificial features like consistent background, brightness, and contrast.

- Attempts to compensate using Test-Time Augmentation (TTA) and extensive data augmentation yielded only marginal improvements, underscoring the importance of representative training data.

Conclusion:
The model suffers from poor extrapolation due to the highly uniform dataset. For field-ready performance, substantial improvements should be made through:

- Dataset diversification (adding naturally captured leaf images),

- Domain adaptation techniques, or

- Synthetic data generation that mimics field conditions.

## User Stories

### User Story 1: I as a user want a summary page that provides me with all the information needed to understand the scope of the project and classification model.

Acceptance criteria:

- Clear statement of the project scope
- Stating business requirements
- Overview of dataset 

### User Story 2: I as a user want to visually differentiate healthy & infected leaves

Acceptance Criteria

- Display averaged and variability images as well 
- Display standard deviation image between healthy and diseased leaves.
- view a montage of healthy or infected leaves for reference

### User Story 3: I as a user want to use the classification model for AI-powered infection prediction

Acceptance Criteria

- Detector page with the option to upload my own pictures and predict chance of infection
- clear statement of my results
- results downloadable

### User Story 4: I as a user want to know more about the models capabilities and stats.

Acceptance Criteria

- dashboard with model metrics
- graphs and tables clearly explained
- summarized for general understanding

### User Story 5: I as a user want to learn about the working hypotheses that went into this project and formed the results.

Acceptance Criteria

    - clear statement of working hypotheses
    - stating validation methods
    - stating results


## Dashboard Design

### Page 1: Summary

This page conveys a quick overview of the project, what to wexpect on the dashboard. It recaps the business case and the approaches for a solution.

**Page contents:**

- Introduction to the challenge behind the project
- Sample images from the dataset (one of each class)
- Business requirements
- Overview of the dataset and its use in this project
- Additional ressources

For details take a look at the screenshots below:

<details><summary>Screenshots</summary>
<img src="assets/readme_img/features/summary_1.png">
<img src="assets/readme_img/features/summary_2.png">
</details>

### Page 2: Diagnosis Assistant

This page gives a visual itroduction into the data set. It helps to see the differences between classes and understand the implications of the images for a prospective model training.

**Page contents:**

- Average and Variability images of the dataset.
- Figures demonstrating the differences between images of different classes.
- The possibility to create an image montage to see a variety of healthy or diseased leaves.

For details take a look at the screenshots below:

<details><summary>Screenshots</summary>
<img src="assets/readme_img/features/diag_ass_1.png">
<img src="assets/readme_img/features/diag_ass_2.png">
<img src="assets/readme_img/features/diag_ass_3.png">
<img src="assets/readme_img/features/diag_ass_4.png">
<img src="assets/readme_img/features/diag_ass_5.png">
</details>

### Page 3: Mildew Detector

This page provides a functional tool to upload images and have them tested by the model. It will provide a prediction and classification report, which can be downloaded.

**Page contents:**

- Explanation of tool functionality.
- Link to set of example images to test the image on.
- Upload widget.
- prediction report.
- download button

For details take a look at the screenshots below:

<details><summary>Screenshots</summary>
<img src="assets/readme_img/features/mild_det_1.png">
<img src="assets/readme_img/features/mild_det_2.png">
<img src="assets/readme_img/features/mild_det_3.png">
</details>

### Page 4: Model performance Metrics

This page provides the user with a detailed deep dive into the evaluation and the metrics of the model trained for this project.

**Page contents:**

- Dataset Split & Class Distribution
- Feature Space (PCA)
- Feature Space (t-SNE)
- Feature Space (UMAP)
- Classification Reports
- Confusion Matrices
- Model Learning Curves
- ROC curve
- Concluding metrics

For details take a look at the screenshots below:

<details><summary>Screenshots</summary>
<img src="assets/readme_img/features/perf_metr_1.png">
<img src="assets/readme_img/features/perf_metr_2.png">
<img src="assets/readme_img/features/perf_metr_3.png">
<img src="assets/readme_img/features/perf_metr_4.png">
<img src="assets/readme_img/features/perf_metr_5.png">
<img src="assets/readme_img/features/perf_metr_6.png">
<img src="assets/readme_img/features/perf_metr_7.png">
<img src="assets/readme_img/features/perf_metr_8.png">
</details>

### Page 5: Working Hypotheses

This page gives a detailed view on the working hypotheses that were pursued during this project. Each Hypothesis is stated and validated.

**Page contents:**

- Hypothesis 1
- Hypothesis 2
- Hypothesis 3
- Hypothesis 4

For details take a look at the screenshots below:

<details><summary>Screenshots</summary>
<img src="assets/readme_img/features/hypo_1.png">
<img src="assets/readme_img/features/hypo_2.png">
<img src="assets/readme_img/features/hypo_3.png">
<img src="assets/readme_img/features/hypo_4.png">
</details>

## CRISP-DM Approach

This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology to ensure a structured and iterative approach to developing a deep learning model for powdery mildew detection in cherry leaves.

### 1. Business Understanding

- Defined the business objective: automate early detection of diseased leaves to reduce manual inspection overhead.

- Success criteria: high classification accuracy, ease of deployment, and actionable output for end users.

### 2. Data Understanding

- Acquired a labeled image dataset from Farmy & Foods.

- Performed exploratory analysis, image inspection, PCA, t-SNE and UMAP to evaluate class separability and identify potential data quality issues.

### 3. Data Preparation

Preprocessing steps included:

- Resizing, normalization, and data augmentation of training set (e.g., flips, zoom) to enhance generalization.

- Dataset split: 70% train, 15% validation, 15% test, maintaining class balance.

### 4. Modeling

- Implemented a Convolutional Neural Network (CNN) optimized for binary classification using Softmax activation and categorical crossentropy.

- adjusted hyperparameters (e.g., learning rate, batch size, optimizer).

- Monitored performance using validation loss, accuracy, and ROC-AUC.

### 5. Evaluation

Evaluated model on the held-out test set using:

- Classification report (precision, recall, F1-score)

- Confusion matrix and ROC curves

- Met performance benchmarks (≥ 90% test accuracy) defined in the business case.

- Tested for data-leakage

- Tested predictions on real world image examples

### 6. Deployment & Monitoring

- Deployed the model in a Streamlit web app for real-time predictions.

- Integrated automated prediction reports with class probabilities.

- Notebooks can be used as a retraining pipeline for continuous improvement if new data is acquired.

### Conclusion

Following the CRISP-DM process enabled a clear, reproducible path from problem definition to deployment. The result is a scalable, interpretable, and production-ready mildew detection tool tailored for use in precision agriculture.


## Model Architecture

### Project Goal  
This project aims to detect **powdery mildew infections** in cherry leaves using an ML model. The dataset consists of uniformly captured, centered cherry leaf images, labeled as either *healthy* or *infected*. Given the clearly distinguishable visual cues that can also be grouped in tests such as PCA, t-SNE and UMAP a CNN is an ideal choice.

---

### Custom CNN Model Summary

To keep all options in the field of CNNs open a custom function for model creation was drafted which can be adjusted to fit binary and multiclass models.

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
or 
|Softmax output | Also suitable for binary classification if properly adjusted, can produce higher accuracy|
|Adam optimizer	| Adaptive and widely used; balances speed and stability.  |


### Sigmoid vs. Softmax in Context

There is some debate between using sigmoid vs. softmax in binary classification:

Sigmoid + 1 output node:

- Interprets the output as the probability of being in class 1.

- Works well when classes are mutually exclusive.

- Computationally cheaper and more direct.

Softmax + 2 output nodes:

- Produces a full probability distribution.

- Can be a technical overkill for binary classification but may help in gradient based classification.

- Arguably performs better in gradient-based optimization.

In this project, we use softmax, treating the task as categorical classification with categorical_crossentropy loss, which yields better accuracy and despite higher technical complexity trains faster.

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

## Testing

### Manual testing

#### User Story 1: I as a user want a summary page that provides me with all the information needed to understand the scope of the project and classification model.

| **Feature** | **Action** | **Expected Result** | **Actual Result** |
|-------------|------------|---------------------|-------------------|
| Summary Page | Dashboard starts with summary page, if not already there, click on sidebar and choose 'Summary Page' | get redirected to Summary page to find all the information you need | Works as expected |

<details><summary>Screenshots</summary>
<img src="assets/readme_img/testing/summary_1.png">
<img src="assets/readme_img/features/summary_2.png">
</details>


### User Story 2: I as a user want to visually differentiate healthy & infected leaves

| **Feature** | **Action** | **Expected Result** | **Actual Result** |
|-------------|------------|---------------------|-------------------|
| Diagnosis Assistant | On the dashboard click on sidebar and choose 'Diagnosis Assitant' click through the different radiator button options| Get redirected to the Diagnosis Assistant and browse through the options to receive the information you need | Works as expected |

<details><summary>Screenshots</summary>
<img src="assets/readme_img/testing/diag_ass_1.png">
<img src="assets/readme_img/testing/diag_ass_2.png">
<img src="assets/readme_img/testing/diag_ass_3.png">
<img src="assets/readme_img/testing/diag_ass_4.png">
<img src="assets/readme_img/testing/diag_ass_5.png">
</details>


### User Story 3: I as a user want to use the classification model for AI-powered infection prediction

Acceptance Criteria

| **Feature** | **Action** | **Expected Result** | **Actual Result** |
|-------------|------------|---------------------|-------------------|
| Mildew Detector | On the dashboard click on sidebar and choose 'Mildew Detector' drag (an) image(s) from your desktop on the widget, or click the browse button to select it from a directory, receive a prediction report from the model. Click on the button to download your report.| Get redirected to the Mildew Detector upload an image and receive report receive a csv file with report upon downloading it. | Works as expected |

<details><summary>Screenshots</summary>
<img src="assets/readme_img/testing/mild_det_1.png">
<img src="assets/readme_img/testing/mild_det_2.png">
<img src="assets/readme_img/testing/mild_det_3.png">
</details>

### User Story 4: I as a user want to know more about the models capabilities and stats.

| **Feature** | **Action** | **Expected Result** | **Actual Result** |
|-------------|------------|---------------------|-------------------|
| Model Performance Metrics | On the dashboard click on sidebar and choose 'Model Performance Metrics'| Get redirected to the Model Performance Metrics page and find detailed information on the model, its evaluation and metrics | Works as expected |

<details><summary>Screenshots</summary>
<img src="assets/readme_img/testing/perf_metr_1.png">
<img src="assets/readme_img/features/perf_metr_2.png">
<img src="assets/readme_img/features/perf_metr_3.png">
<img src="assets/readme_img/features/perf_metr_4.png">
<img src="assets/readme_img/features/perf_metr_5.png">
<img src="assets/readme_img/features/perf_metr_6.png">
<img src="assets/readme_img/features/perf_metr_7.png">
<img src="assets/readme_img/features/perf_metr_8.png">
</details>

### User Story 5: I as a user want to learn about the working hypotheses that went into this project and formed the results.

| **Feature** | **Action** | **Expected Result** | **Actual Result** |
|-------------|------------|---------------------|-------------------|
| Working Hypotheses | On the dashboard click on sidebar and choose 'Working Hypotheses'| Get redirected to the Working Hypotheses page Page and read all about the working hypotheses drafted for this project and the validation process and conlusions. | Works as expected |

<details><summary>Screenshots</summary>
<img src="assets/readme_img/testing/hypo_1.png">
<img src="assets/readme_img/features/hypo_2.png">
<img src="assets/readme_img/features/hypo_3.png">
<img src="assets/readme_img/features/hypo_4.png">
</details>

### Validator Testing

PEP8 compliance ensured using flake8 extension for VScode



## Debugging

### Fixed Bugs

| Bug | Fix |
|---|---|
|**Confusion matrix for model was off compared to performance statistics**|Wrote 'collect predictions functions that stores predictions in a list to avoid errors|
|**Training the CNN with a softmax output returned an error where it could not process the dataset**|Had to set the class mode of the dataset augmentation to categorical instead of binary|
|**Code cell could not find test-image folder**| removed hidden space typed in folder name|
|**Screenshots were not showing in the readme**| moving a folder hadchanged the filepath, changing it back solved it|

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

The Code Institute Walkthrough project one and the Business case were a huge inspiration

[shields.io](https://shields.io/) was used to create the informative badges on top of the readme

I used Copilot for code refactoring, and chatgpt was really helpful creating sample images for model testing.


### Acknowledgements

I would like to thank my wife for keeping me sane and on track and my daughter for leading of track when I needed it!

Thanks to [Code Institute](https://codeinstitute.net/global/), especially Kay Welfare and my mentor Mo Shami. 

### Deployed version at [Mildew Detector](https://detect-mildew-1e5e3ef17076.herokuapp.com/)