# Loan Prediction Classification

A machine learning project that predicts whether a loan application will be approved (`Y`) or rejected (`N`) based on applicant profile, income, credit history, and property details.

## Project Summary

This repository contains an end-to-end classification workflow built around the classic loan approval dataset. The notebook follows the full ML pipeline:

1. Load the training and test datasets
2. Clean missing values
3. Encode categorical variables into numeric form
4. Explore feature relationships with correlation plots
5. Compare multiple classification models
6. Train a final Logistic Regression model
7. Generate the submission file

The final prediction file is saved as [`output.csv`](output.csv).

## Business Goal

The goal is to estimate loan approval decisions automatically so that financial teams can make faster, more consistent screening decisions.

## Dataset

The dataset is stored in [`Dataset/`](Dataset/):

- [`train.csv`](Dataset/train.csv) - labeled training data
- [`test.csv`](Dataset/test.csv) - unlabeled test data used for prediction
- [`Sample_Submission.csv`](Dataset/Sample_Submission.csv) - reference submission format

### Target Variable

- `Loan_Status`
  - `Y` = loan approved
  - `N` = loan rejected

### Main Features

- `Loan_ID`
- `Gender`
- `Married`
- `Dependents`
- `Education`
- `Self_Employed`
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`
- `Property_Area`

## Methodology

The notebook keeps the original project logic while using cleaner modern Python syntax:

- Missing values are handled with median, mean, and random binary imputation where the original notebook used random filling
- Categorical values are mapped to numeric labels
- The combined train/test dataset is used for preprocessing
- A train/test split is created from the labeled data
- Several models are compared with cross-validation
- Logistic Regression is used for the final prediction step

## Results

The generated submission in [`output.csv`](output.csv) contains predictions for 367 test applications.

- Approved loans: 281
- Rejected loans: 86

### Output Preview

| Loan_ID | Loan_Status |
| --- | --- |
| LP001015 | 1 |
| LP001022 | 1 |
| LP001031 | 1 |
| LP001035 | 0 |
| LP001051 | 1 |

`1` means approved, `0` means rejected.

## Visualizations

### Loan Status Distribution

![Loan Status Distribution](assets/loan_status_distribution.svg)

### Credit History vs Loan Status

![Credit History vs Loan Status](assets/credit_history_vs_status.svg)

### Feature Correlation Heatmap

![Feature Correlation Heatmap](assets/correlation_heatmap.svg)

### Predicted Loan Status Distribution

![Predicted Loan Status Distribution](assets/predicted_status_distribution.svg)

## Project Files

- [`Loan+Prediction.ipynb`](Loan+Prediction.ipynb) - main notebook
- [`Loan+Prediction.py`](Loan+Prediction.py) - notebook export
- [`output.csv`](output.csv) - final predictions
- [`assets/`](assets/) - visual summaries used in this README

## How To Run

1. Open [`Loan+Prediction.ipynb`](Loan+Prediction.ipynb) in Jupyter Notebook or JupyterLab.
2. Make sure the dataset files are present in [`Dataset/`](Dataset/).
3. Run the notebook cells from top to bottom.
4. The final prediction file will be written to [`output.csv`](output.csv).

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- seaborn
- matplotlib

## Notes

- The project is intentionally kept close to the original notebook logic.
- The README now includes the main output file and visual assets so recruiters can understand the project quickly without opening the notebook first.
