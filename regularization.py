# Step 1: Import necessary libraries
import pandas as pd
import numpy as np

# Step 2: Load the dataset
df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\bangalore house price.csv")
df.info()

print(df.shape)
# Step 3: Define features and target
X = df.drop(columns=['price'])  # All columns except the target
y = df['price']

# Step 4: Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize and apply Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

lr_pipeline = make_pipeline(StandardScaler(), LinearRegression())
lr_pipeline.fit(X_train, y_train)

# Evaluate Linear Regression
y_train_pred = lr_pipeline.predict(X_train)
y_test_pred = lr_pipeline.predict(X_test)

from sklearn.metrics import r2_score
print("Linear Regression:")
print("Train R²:", r2_score(y_train, y_train_pred))
print("Test R² :", r2_score(y_test, y_test_pred))

# Step 6: Apply ShuffleSplit Cross-Validation
from sklearn.model_selection import ShuffleSplit, cross_val_score

shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
cv_scores = cross_val_score(lr_pipeline, X, y, cv=shuffle_split, scoring='r2')
#print("Cross-Validation R² Scores:", cv_scores)
print("Average CV R²:", np.round(cv_scores.mean(),2))

"""We apply ShuffleSplit, a kind of cross-validation that:

Randomly shuffles the data

*   Performs 5 different train-test splits
*   Reports R² scores on each split
*   This helps verify the model's consistency and generalization.

"""
# Step 7: Apply LassoCV to find optimal alpha
from sklearn.linear_model import Lasso, LassoCV

lasso_cv_pipeline = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=1000,alphas = None))
lasso_cv_pipeline.fit(X_train, y_train)

"""1. scikit-learn automatically sets alphas=None, which then triggers this logic inside LassoCV:
  “If alphas is None, use 100 values on a log scale between alpha_max and alpha_max * eps, where eps=0.001
2. max_iter=100, it defaults to 100 iterations (in scikit-learn). But if the loss hasn’t converged in 100 steps, you’ll get warning, hence better to keep highest number
"""

print("All alphas tried by LassoCV:", lasso_cv_pipeline.named_steps['lassocv'].alphas_)

# Extract best alpha
best_alpha = lasso_cv_pipeline.named_steps['lassocv'].alpha_
print("Best alpha from LassoCV:", best_alpha)

# Evaluate performance with best alpha
y_train_lasso = lasso_cv_pipeline.predict(X_train)
y_test_lasso = lasso_cv_pipeline.predict(X_test)
print("LassoCV Train R²:", np.round(r2_score(y_train, y_train_lasso),2))
print("LassoCV Test R² :", np.round(r2_score(y_test, y_test_lasso),2))

# Step 8: Plot alpha vs CV error
mse_path = lasso_cv_pipeline.named_steps['lassocv'].mse_path_.mean(axis=1)
alphas_tested = lasso_cv_pipeline.named_steps['lassocv'].alphas_
alphas_tested

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(alphas_tested, mse_path, marker='o')
plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best alpha: {best_alpha:.4f}')
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Cross-Validation MSE')
plt.title('LassoCV: Alpha vs. Mean Cross-Validation Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
*   We plot the mean cross-validation error (MSE) against different alphas.

*   A red line shows the best alpha (0.1758 in this case).
"""
# Step 9: Feature selection — count zeroed coefficients
lasso_model = lasso_cv_pipeline.named_steps['lassocv']
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso_model.coef_
})
zeroed_features = coef_df[coef_df['Coefficient'] == 0]
print("Number of features removed by LassoCV:", len(zeroed_features))

"""
*   We examine which coefficients were set to zero by Lasso.
*   These features are considered not useful
*   Lasso at α = 0.1758 removed 14 features (i.e., retained 93)
"""

# Step 10: Test with higher alpha to simplify model
for alpha_val in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    lasso_custom = make_pipeline(StandardScaler(), Lasso(alpha=alpha_val, max_iter=10000))
    lasso_custom.fit(X_train, y_train)

    y_test_pred = lasso_custom.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    non_zero_count = sum(lasso_custom.named_steps['lasso'].coef_ != 0)

    print(f"\nAlpha: {alpha_val}")
    print(f"Test R²: {r2_test:.4f}")
    print(f"Retained Features: {non_zero_count}")

# Step 11: Final model with alpha = 0.9 (recommended)
final_alpha = 0.9
final_lasso = make_pipeline(StandardScaler(), Lasso(alpha=final_alpha, max_iter=100))
final_lasso.fit(X_train, y_train)

final_coef = final_lasso.named_steps['lasso'].coef_
selected_features = X.columns[final_coef != 0]
print("\nFinal Selected Features (alpha = 0.9):", len(selected_features))

# Final evaluation
y_train_final = final_lasso.predict(X_train)
y_test_final = final_lasso.predict(X_test)

print("Final Lasso Train R²:", np.round(r2_score(y_train, y_train_lasso)*100,2))
print("Final Lasso Test R² :", np.round(r2_score(y_test, y_test_lasso)*100,2))

# print the feature names which are finalized from above after using alpha as 0.9

selected_features





