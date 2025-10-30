import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\sales.csv")
df.head()
df.info()

df.isnull().sum()
# Data transformation

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["ShelveLoc"] = LE.fit_transform(df["ShelveLoc"])
df["Urban"]     = LE.fit_transform(df["Urban"])
df["US"]        = LE.fit_transform(df["US"])
df["high"]      = LE.fit_transform(df["high"])
df.head()

# data partition
Y = df["high"]
X = df.drop(df[['high']],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.3)

#---------------------------------------------------------1
# !pip install xgboost
#xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=1,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'   # avoids warning messages
)

xgb_model.fit(X_train, Y_train)

y_pred_train_xgb = xgb_model.predict(X_train)
y_pred_test_xgb = xgb_model.predict(X_test)

training_accuracy_xgb = accuracy_score(Y_train, y_pred_train_xgb)
test_accuracy_xgb = accuracy_score(Y_test, y_pred_test_xgb)

print("XGBoost - Training Accuracy:", round(training_accuracy_xgb, 2))
print("XGBoost - Test Accuracy:", round(test_accuracy_xgb, 2))

from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid for XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 4, 6],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# Fit
grid_search.fit(X_train, Y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# Evaluate best model
best_xgb_model = grid_search.best_estimator_
y_pred_test_best = best_xgb_model.predict(X_test)
test_accuracy_best = accuracy_score(Y_test, y_pred_test_best)
print("Test Accuracy of Best Model:", round(test_accuracy_best, 2))

#LGBM
# !pip install lightgbm

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# LGBM Classifier
lgbm_model = LGBMClassifier(
    n_estimators=100,
    max_depth=1,
    learning_rate=0.1,
    random_state=42
)

lgbm_model.fit(X_train, Y_train)

y_pred_train_lgbm = lgbm_model.predict(X_train)
y_pred_test_lgbm = lgbm_model.predict(X_test)

training_accuracy_lgbm = accuracy_score(Y_train, y_pred_train_lgbm)
test_accuracy_lgbm = accuracy_score(Y_test, y_pred_test_lgbm)

print("LightGBM - Training Accuracy:", round(training_accuracy_lgbm, 2))
print("LightGBM - Test Accuracy:", round(test_accuracy_lgbm, 2))

from sklearn.model_selection import GridSearchCV

# Define parameter grid for LightGBM
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [-1, 3, 5]   # -1 means no limit
}

lgbm_model = LGBMClassifier(random_state=42)

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=lgbm_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, Y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Evaluate best model
best_lgbm_model = grid_search.best_estimator_
y_pred_test_best = best_lgbm_model.predict(X_test)

test_accuracy_best = accuracy_score(Y_test, y_pred_test_best)
print("Test accuracy of best LGBM model:", round(test_accuracy_best, 2))


