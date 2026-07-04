# Credit Card Approval Prediction

An end-to-end machine learning project that predicts whether a credit card applicant is likely to be **Approved** or **Rejected**.

## Project Overview

This project covers the complete ML lifecycle:

- Data collection from `application_record.csv` and `credit_record.csv`.
- Dataset merge using `ID`.
- Binary target creation from repayment status.
- Data cleaning, preprocessing, and feature engineering.
- Exploratory data analysis.
- Model training and evaluation.
- Model comparison across Logistic Regression, Decision Tree, Random Forest, XGBoost, and additional models.
- Saved best model with Joblib.
- Flask web app for single and batch predictions.
- Testing, documentation, design diagrams, and demonstration notes.

## Rubric Folder Structure

```text
Credit Card Approval Prediction/
|-- Brainstorming/
|-- Requirement Analysis/
|-- Project Design Phase/
|-- Planning/
|-- Project Development Phase/
|-- Project Testing/
|-- Project Documentation/
|-- Project Demonstration/
|-- data/
|-- models/
|-- notebooks/
|-- reports/
|-- static/
|-- templates/
|-- tests/
|-- app.py
|-- predict.py
|-- preprocessing.py
|-- train.py
|-- Review-Project.ps1
|-- START_WEBSITE.bat
|-- requirements.txt
|-- Procfile
|-- runtime.txt
|-- Dockerfile
|-- .env.example
`-- README.md
```

## Installation

Fast Windows start:

```powershell
.\START_WEBSITE.bat
```

Then open:

```text
http://127.0.0.1:5000
```

Full Windows project review:

```powershell
powershell -ExecutionPolicy Bypass -File .\Review-Project.ps1
```

This review command creates or repairs the website environment, installs the required packages, checks important project files, runs the automated tests, and starts the website at `http://127.0.0.1:5000` if everything passes.

Manual setup:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If VS Code reports that the environment has no `pip` or cannot resolve the interpreter, run `Review-Project.ps1` or `START_WEBSITE.bat`. They detect a broken website environment, rename it safely, and create a fresh one. This project also includes `.vscode/settings.json` pointing VS Code to the project environment.

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The project expects:

- `data/application_record.csv`
- `data/credit_record.csv`

The target is generated from `credit_record.csv`:

- Serious overdue statuses `2`, `3`, `4`, or `5` become **Rejected**.
- Clean or minor statuses such as `C`, `X`, `0`, or `1` become **Approved**.

This conversion is implemented in `preprocessing.py`.

## How to Train

```powershell
python train.py
```

Training creates:

- `models/best_model.pkl`
- `models/model_metadata.pkl`
- `models/preprocessor.pkl`
- `reports/model_comparison.csv`
- Evaluation plots in `reports/plots/`

## Models and Metrics

The training script compares:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine
- KNN
- Naive Bayes
- Gradient Boosting

Metrics include accuracy, precision, recall, F1 score, ROC-AUC, confusion matrix, classification report, and cross-validation F1.

## How to Run the Flask App

```powershell
python run.py
```

Open:

```text
http://127.0.0.1:5000
```

## Web App Features

- Home page.
- Signup and login.
- Single applicant prediction.
- Batch CSV prediction.
- Prediction history.
- Download prediction as PDF.
- Download prediction as Excel.
- Model information page.
- Feature reference page.
- Custom 404 page.
- Input validation and error handling.

## Testing

```powershell
pytest
```

The automated tests cover validation, credit-rule behavior, downloads, history management, login/signup behavior, and Google login configuration.

## Deployment

The project includes `Procfile`, `runtime.txt`, and `Dockerfile`.

For local configuration, copy `.env.example` to `.env` and set a strong `SECRET_KEY`. Do not commit `.env`.

## Important Limitation

The included dataset is a small sample. The project demonstrates the full workflow and app integration, but a production credit decision system would require a larger validated dataset, fairness review, monitoring, security hardening, and compliance checks.
