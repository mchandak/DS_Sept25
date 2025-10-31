import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

os.chdir(r"D:\Manoj\1ExcelR\Latest content 16th Aug 2025\Material\Day 28- Final project with Deployment\My")

# 1. Load data
df = pd.read_csv("loan prediction.csv")   # adjust path if needed
df.info()
# Drop Loan_ID
df = df.drop("Loan_ID", axis=1)

# Separate features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ===================== IDENTIFY COLUMNS =====================
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# ===================== PREPROCESSING PIPELINES =====================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ===================== CREATE MODEL PIPELINE =====================
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# ===================== TRAIN TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ===================== TRAIN MODEL =====================
clf.fit(X_train, y_train)

# ===================== PREDICTIONS =====================
y_pred = clf.predict(X_test)

# ===================== EVALUATION METRICS =====================
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:")
print(classification_report(y_test, y_pred,labels=['Y', 'N']))

cm = confusion_matrix(y_test, y_pred,labels=['Y', 'N'])
print("\n Confusion Matrix:\n", cm)

# ===================== SAVE MODEL =====================

joblib.dump(clf, "model.pkl")
print("Saved model to model.pkl")


# ===================== Coefficients of Model =====================
# Extract logistic regression model from pipeline
log_reg_model = clf.named_steps['model']

# Get feature names after one-hot encoding
ohe = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
feature_names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))

# Combine feature names and coefficients
coefficients = log_reg_model.coef_[0]
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort by absolute value of coefficient 
coef_df = coef_df.sort_values(by="Coefficient", ascending=False)
print("\n Coefficient DataFrame :\n", coef_df.head(10))
