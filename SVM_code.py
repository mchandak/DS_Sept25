import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate a synthetic non-linear dataset with 5 continuous variables
df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\Pharma_Industry.csv")
df.shape

# Convert to DataFrame for reference
X = df.iloc[:,0:3]
y = df["Drug Response"]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define SVM models with different kernels
svm_linear = SVC(kernel='linear', C=1.0)
svm_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_rbf = SVC(kernel='rbf', gamma=0.5, C=1.0)

# Train and evaluate SVM with linear kernel
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)

# Train and evaluate SVM with polynomial kernel
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)

# Train and evaluate SVM with RBF kernel
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

# Display results
print(f"Linear SVM Accuracy: {accuracy_linear:.2f}")
print(f"Polynomial SVM Accuracy: {accuracy_poly:.2f}")
print(f"RBF SVM Accuracy: {accuracy_rbf:.2f}")











