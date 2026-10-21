# PyTorch vs TensorFlow: Income Classification

## Project Overview

This project implements and compares deep learning models built using **PyTorch** and **TensorFlow/Keras** to predict income categories from the Adult Income dataset.

The project evaluates both frameworks using baseline neural networks and applies **Optuna hyperparameter optimization** to investigate whether tuning improves model performance.

## Objectives

- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess the dataset.
- Handle missing values and encode categorical features.
- Split the data into training and testing sets.
- Build and train neural networks using PyTorch and TensorFlow.
- Evaluate model performance using classification metrics.
- Optimize hyperparameters using Optuna.
- Compare baseline and tuned models to determine which performs better.

## Dataset

**Dataset:** Adult Income Dataset (`adult_dataset.csv`)

The dataset contains demographic and employment-related information used to classify individuals into income categories.

- **Target variable:** `income`
- **Task:** Binary classification
- **Input features:** Demographic, educational, employment, and other personal attributes.

The dataset is expected to be available locally or through Google Drive. Update the dataset path in the notebook to match your environment.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- PyTorch
- TensorFlow / Keras
- Optuna
- Jupyter Notebook / Google Colab

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Inspect dataset structure and dimensions.
- Examine descriptive statistics and data types.
- Identify missing values and duplicate records.

### 2. Data Preprocessing

- Remove duplicate rows.
- Handle missing numerical values using `KNNImputer`.
- Handle missing categorical values using `SimpleImputer`.
- Apply label encoding and one-hot encoding.
- Separate features (`X`) and target (`y`).
- Split the data into training and testing sets using an 80:20 ratio with stratification.

### 3. PyTorch Implementation

Build a feedforward neural network using `torch.nn` with:

- Two hidden layers using ReLU activation.
- Dropout regularization.
- Adam optimizer.
- Cross-entropy loss.
- A custom training loop and `DataLoader`.

### 4. TensorFlow Implementation

Build a comparable feedforward neural network using TensorFlow/Keras with:

- Two hidden layers using ReLU activation.
- Dropout regularization.
- Adam optimizer.
- Sparse categorical cross-entropy loss.
- Keras model training and validation.

### 5. Hyperparameter Optimization with Optuna

Optuna is used to search for better model configurations independently for both frameworks.

The hyperparameters explored include:

- Hidden-layer sizes
- Learning rate
- Dropout rate
- Batch size

The optimization objective is **validation macro-F1 score**. Each framework uses 10 Optuna trials, with a limited number of epochs per trial to keep computation practical.

### 6. Model Evaluation and Comparison

The notebook compares four configurations:

1. PyTorch baseline
2. PyTorch with Optuna tuning
3. TensorFlow baseline
4. TensorFlow with Optuna tuning

The evaluation metrics include:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score

Classification reports are also generated to examine performance across individual income classes.

## Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn torch tensorflow optuna
```

## How to Run

1. Clone or download this repository.
2. Open `Assesment_Pytorch_vs_Tensorflow_fixed.ipynb` in Jupyter Notebook or Google Colab.
3. Upload `adult_dataset.csv` or connect Google Drive.
4. Update the dataset path in the data-loading cell.
5. Run all notebook cells in order.
6. Review the baseline evaluation, Optuna tuning results, final comparison table, and generated conclusion.

**Note:** Deep learning training and hyperparameter tuning may take additional time depending on the available hardware.

## Results and Findings

The notebook generates evaluation results for all four model configurations after execution.

The final comparison examines whether Optuna improves accuracy and macro-F1 for each framework and compares the resulting models.

Actual metric values are generated at runtime and are intentionally not reported here before the notebook has been executed.

## Conclusion

This project demonstrates how the same income-classification task can be implemented using PyTorch and TensorFlow/Keras. It also explores the role of automated hyperparameter optimization in improving neural network performance.

The best model should be selected based on its held-out test performance, with particular attention to macro-F1 and the classification report rather than accuracy alone.

## Future Improvements
