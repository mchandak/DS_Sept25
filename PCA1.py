# PCA on Breast Cancer Wisconsin Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir("D:\\Manoj\\1ExcelR\\Data")
# Load dataset
data = pd.read_csv("breast-cancer-wisconsin-data.csv")

# Display initial info
print("Shape:", data.shape)
print("\nColumns:\n", data.columns)
print("\nMissing values:\n", data.isnull().sum())

data.drop(['id'], axis=1, inplace=True)

# Separate features and target
X = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']

# Encode target (optional, for plotting)
y_encoded = y.map({'M': 1, 'B': 0})

# Standardize the data (very important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=None)   # Keep all components
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_var = pca.explained_variance_ratio_
cum_explained_var = np.cumsum(explained_var)

# Print explained variance
print("\nExplained Variance Ratio (first 10 components):\n", explained_var[:10])
print("\nCumulative Explained Variance:\n", cum_explained_var)

# Plot cumulative explained variance
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cum_explained_var)+1), cum_explained_var, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.grid(True)
plt.show()

# Choose 2 components for visualization
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(data=X_pca_2, columns=['PC1', 'PC2'])
pca_df['Diagnosis'] = y

# Visualize 2D PCA
plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Diagnosis', data=pca_df, palette=['green','red'])
plt.title('PCA: Breast Cancer Dataset (2 Components)')
plt.show()


#########################
#Part 2
# PCA + Random Forest Classification
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

os.chdir("D:\\Manoj\\1ExcelR\\Data")
# Load dataset
data = pd.read_csv("breast-cancer-wisconsin-data.csv")

# Drop ID column if present

data.drop(['id'], axis=1, inplace=True)

# Encode target
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# 2️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3️⃣ Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4️⃣ Apply PCA (choose components that explain ~95% variance)
pca = PCA(0.95)  # retain 95% of the variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original feature count: {X.shape[1]}")
print(f"Reduced feature count (PCA): {pca.n_components_}")
print("\nExplained Variance Ratio per component:\n", pca.explained_variance_ratio_)

# 5️⃣ Train Random Forest on PCA data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)

# 6️⃣ Predict and Evaluate
y_pred = rf.predict(X_test_pca)

print("\n✅ Model Evaluation on PCA Data")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7️⃣ Optional: Plot Feature Importance (on PCA components)
importances = rf.feature_importances_
plt.figure(figsize=(8,5))
plt.bar(range(len(importances)), importances)
plt.xlabel("Principal Components")
plt.ylabel("Importance")
plt.title("Feature Importance (PCA Components)")
plt.show()


#Part 3
# Compare Random Forest accuracy with different PCA component counts (5 to 10)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

os.chdir("D:\\Manoj\\1ExcelR\\Data")
# Load dataset
data = pd.read_csv("breast-cancer-wisconsin-data.csv")

# Drop unnecessary columns
data.drop(['id'], axis=1, inplace=True)

# Encode target
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2️⃣ Try PCA with components from 2 to 10
results = []

for n in range(2, 11):
    # Apply PCA
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    start = time.time()
    rf.fit(X_train_pca, y_train)
    y_pred = rf.predict(X_test_pca)
    
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    results.append({'Components': n, 'Accuracy': acc})

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\nRandom Forest Accuracy with Different PCA Components (5–10):\n")
print(results_df)




