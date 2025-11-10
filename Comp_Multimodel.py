import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os
# Load data
os.chdir(r"D:\Manoj\1ExcelR\Latest content 16th Aug 2025\Material\Day 28- Final project with Deployment\My")

# 1. Load data
df = pd.read_csv("loan prediction.csv") 

df = df.drop('Loan_ID', axis=1)  # Drop ID column

# Target variable mapping
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
y = df['Loan_Status']
X = df.drop('Loan_Status', axis=1)

# Categorical and numerical columns
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if X[col].dtype != 'object']

# Impute missing values
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

# One-hot encode categoricals
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Standardize numerical features
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Models to benchmark
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)   
    print(f'--- {name} ---','Accuracy:', f'{acc:.4f}')
   
 


