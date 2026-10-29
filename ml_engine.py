"""
ML Engine for Loan Prediction System
Provides data preprocessing, model training, evaluation metrics, and inference.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH = os.path.join(DATASET_DIR, "test.csv")

def load_raw_data():
    """Load raw training and test datasets."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def preprocess_data(train_df, test_df=None):
    """
    Robust preprocessing matching the repository logic with deterministic imputation
    and enhanced financial indicators.
    """
    train = train_df.copy()
    test = test_df.copy() if test_df is not None else None
    
    # Target encoding
    if 'Loan_Status' in train.columns:
        train['Loan_Status'] = train['Loan_Status'].map({'Y': 1, 'N': 0, 1: 1, 0: 0})
    
    # Store IDs
    train_ids = train['Loan_ID'] if 'Loan_ID' in train.columns else None
    test_ids = test['Loan_ID'] if test is not None and 'Loan_ID' in test.columns else None
    
    # Imputation values based on training data
    impute_vals = {
        'Gender': train['Gender'].mode()[0] if not train['Gender'].dropna().empty else 'Male',
        'Married': train['Married'].mode()[0] if not train['Married'].dropna().empty else 'Yes',
        'Dependents': '0',
        'Self_Employed': 'No',
        'LoanAmount': train['LoanAmount'].median(),
        'Loan_Amount_Term': train['Loan_Amount_Term'].mode()[0] if not train['Loan_Amount_Term'].dropna().empty else 360.0,
        'Credit_History': 1.0,
        'Property_Area': 'Semiurban'
    }

    def clean_df(df):
        d = df.copy()
        d['Gender'] = d['Gender'].fillna(impute_vals['Gender']).map({'Male': 1, 'Female': 0}).fillna(1).astype(int)
        d['Married'] = d['Married'].fillna(impute_vals['Married']).map({'Yes': 1, 'No': 0}).fillna(1).astype(int)
        d['Dependents'] = d['Dependents'].fillna(impute_vals['Dependents']).map({'0': 0, '1': 1, '2': 2, '3+': 3}).fillna(0).astype(int)
        d['Education'] = d['Education'].map({'Graduate': 1, 'Not Graduate': 0}).fillna(1).astype(int)
        d['Self_Employed'] = d['Self_Employed'].fillna(impute_vals['Self_Employed']).map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        d['Property_Area'] = d['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0}).fillna(1).astype(int)
        
        d['ApplicantIncome'] = pd.to_numeric(d['ApplicantIncome'], errors='coerce').fillna(0)
        d['CoapplicantIncome'] = pd.to_numeric(d['CoapplicantIncome'], errors='coerce').fillna(0)
        d['LoanAmount'] = pd.to_numeric(d['LoanAmount'], errors='coerce').fillna(impute_vals['LoanAmount'])
        d['Loan_Amount_Term'] = pd.to_numeric(d['Loan_Amount_Term'], errors='coerce').fillna(impute_vals['Loan_Amount_Term'])
        d['Credit_History'] = pd.to_numeric(d['Credit_History'], errors='coerce').fillna(impute_vals['Credit_History']).astype(float)
        
        return d

    clean_train = clean_df(train)
    clean_test = clean_df(test) if test is not None else None
    
    return clean_train, clean_test, impute_vals

FEATURE_COLS = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

def get_models():
    """Returns dictionary of candidate models with pipelines for standard scaling where needed."""
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=7))
        ]),
        'Linear Discriminant Analysis': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearDiscriminantAnalysis())
        ]),
        'Support Vector Machine (SVC)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, kernel='rbf', random_state=7))
        ]),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=7),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=7),
        'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=7),
        'K-Nearest Neighbors (KNN)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ]),
        'Gaussian Naive Bayes': GaussianNB()
    }

def train_and_evaluate_all():
    """
    Trains all models on the dataset, evaluates cross-validation and validation metrics,
    and returns a comprehensive results dictionary.
    """
    raw_train, raw_test = load_raw_data()
    clean_train, clean_test, _ = preprocess_data(raw_train, raw_test)
    
    X = clean_train[FEATURE_COLS]
    y = clean_train['Loan_Status']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    
    models = get_models()
    results = {}
    leaderboard = []
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    
    for name, model in models.items():
        # Cross validation on full training set
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        
        # Fit on train split
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, probs)
        else:
            auc = 0.5
            probs = preds
            
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        cm = confusion_matrix(y_val, preds)
        
        # Full fit for inference
        full_model = get_models()[name]
        full_model.fit(X, y)
        
        results[name] = {
            'model': full_model,
            'split_model': model,
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'test_acc': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'roc_auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'y_val': y_val.tolist(),
            'y_preds': preds.tolist(),
            'y_probs': probs.tolist()
        }
        
        leaderboard.append({
            'Model': name,
            '10-Fold CV Accuracy': f"{cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)",
            'Validation Accuracy': f"{acc*100:.2f}%",
            'Precision': f"{prec*100:.2f}%",
            'Recall': f"{rec*100:.2f}%",
            'F1 Score': f"{f1:.3f}",
            'ROC AUC': f"{auc:.3f}",
            '_cv_val': cv_scores.mean(),
            '_val_acc': acc
        })
        
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by='_val_acc', ascending=False).reset_index(drop=True)
    
    return {
        'results': results,
        'leaderboard': leaderboard_df,
        'raw_train': raw_train,
        'raw_test': raw_test,
        'clean_train': clean_train,
        'clean_test': clean_test,
        'feature_cols': FEATURE_COLS
    }

def get_feature_importance(model_dict, model_name='Logistic Regression'):
    """Extract feature importance or regression coefficients."""
    model_obj = model_dict['results'][model_name]['model']
    
    if hasattr(model_obj, 'named_steps'):
        clf = model_obj.named_steps['clf']
    else:
        clf = model_obj
        
    if hasattr(clf, 'coef_'):
        weights = clf.coef_[0]
        imp_df = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'Weight': weights,
            'Absolute_Impact': np.abs(weights)
        }).sort_values(by='Absolute_Impact', ascending=False)
        return imp_df, 'Coefficient (Log-Odds Impact)'
    elif hasattr(clf, 'feature_importances_'):
        imp = clf.feature_importances_
        imp_df = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'Weight': imp,
            'Absolute_Impact': imp
        }).sort_values(by='Absolute_Impact', ascending=False)
        return imp_df, 'Feature Importance (Gini)'
    else:
        return pd.DataFrame({'Feature': FEATURE_COLS, 'Weight': np.ones(len(FEATURE_COLS))}), 'Uniform'

def predict_single_applicant(model, applicant_dict):
    """
    Predict loan decision for a single applicant record.
    Returns decision, probability, and risk factors.
    """
    df = pd.DataFrame([applicant_dict])
    clean_df, _, _ = preprocess_data(df)
    features = clean_df[FEATURE_COLS]
    
    prob = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else float(model.predict(features)[0])
    pred = 1 if prob >= 0.5 else 0
    
    # Financial metrics
    app_inc = float(applicant_dict.get('ApplicantIncome', 0))
    co_inc = float(applicant_dict.get('CoapplicantIncome', 0))
    total_income = app_inc + co_inc
    loan_k = float(applicant_dict.get('LoanAmount', 100))
    term_months = float(applicant_dict.get('Loan_Amount_Term', 360))
    
    # Rough EMI formula (annual interest rate ~ 8.5%)
    r = (8.5 / 12) / 100
    n = max(term_months, 12)
    loan_principal = loan_k * 1000
    if r > 0:
        emi = (loan_principal * r * ((1 + r)**n)) / (((1 + r)**n) - 1)
    else:
        emi = loan_principal / n
        
    emi_ratio = (emi / total_income * 100) if total_income > 0 else 100.0
    loan_to_income = (loan_principal / (total_income * 12)) if total_income > 0 else 999.0
    
    # Risk factor audit
    positives = []
    cautions = []
    
    if applicant_dict.get('Credit_History', 1) == 1:
        positives.append("Excellent Credit History: Strong past record of on-time loan repayments (+40% approval driver)")
    else:
        cautions.append("No/Poor Credit History: Significant historical risk factor (rejection driver)")
        
    if applicant_dict.get('Property_Area') == 'Semiurban':
        positives.append("Semiurban Property: Highest historical approval cohort (76.8% approval rate)")
        
    if emi_ratio <= 35:
        positives.append(f"Comfortable EMI Burden: Estimated monthly EMI of ${emi:,.0f} is {emi_ratio:.1f}% of total income")
    elif emi_ratio <= 50:
        cautions.append(f"Moderate EMI Burden: Estimated monthly EMI of ${emi:,.0f} accounts for {emi_ratio:.1f}% of total income")
    else:
        cautions.append(f"Elevated Debt-to-Income: Monthly EMI of ${emi:,.0f} requires {emi_ratio:.1f}% of household monthly income")
        
    if co_inc > 0:
        positives.append(f"Co-Applicant Added: Additional monthly income of ${co_inc:,.0f} strengthens debt coverage")
    else:
        if app_inc < 3000 and loan_k > 120:
            cautions.append("Single Applicant with High Leverage: Adding an eligible coapplicant could bolster application strength")
            
    risk_tier = "Prime (Low Risk)" if prob >= 0.75 else ("Near-Prime (Moderate Risk)" if prob >= 0.50 else "High Risk / Subprime")
    
    return {
        'prediction': pred,
        'status_label': 'APPROVED' if pred == 1 else 'REJECTED',
        'probability': prob,
        'risk_tier': risk_tier,
        'emi': emi,
        'total_income': total_income,
        'emi_ratio': emi_ratio,
        'loan_to_income': loan_to_income,
        'positives': positives,
        'cautions': cautions
    }

if __name__ == "__main__":
    print("Testing ML Engine...")
    data = train_and_evaluate_all()
    print("Leaderboard:\n", data['leaderboard'])
    sample = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Semiurban'
    }
    pred = predict_single_applicant(data['results']['Logistic Regression']['model'], sample)
    print("\nSample Decision:", pred)

