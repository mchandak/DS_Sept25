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

# fit the model using adaboost classifier

from sklearn.ensemble import AdaBoostClassifier

# AdaBoost Classifier
ab_model = AdaBoostClassifier()
ab_model.fit(X_train, Y_train)

y_pred_train_ab = ab_model.predict(X_train)
y_pred_test_ab = ab_model.predict(X_test)

training_accuracy_ab = accuracy_score(Y_train, y_pred_train_ab)
test_accuracy_ab = accuracy_score(Y_test, y_pred_test_ab)

print("AdaBoost - Training Accuracy:", np.round(training_accuracy_ab, 2))
print("AdaBoost - Test Accuracy:", np.round(test_accuracy_ab, 2))


# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1,10]
}

# Create a RandomForestClassifier
ab_model = AdaBoostClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=ab_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters and the best score
print("Best hyperparameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


# Evaluate the best model on the test set
best_ab_model = grid_search.best_estimator_
y_pred_test_best = best_ab_model.predict(X_test)
test_accuracy_best = accuracy_score(Y_test, y_pred_test_best)
print("Test accuracy of best model:", test_accuracy_best)


# Gradient boosting

from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=1,random_state=42,learning_rate=0.1)
gb_model.fit(X_train, Y_train)

y_pred_train_gb = gb_model.predict(X_train)
y_pred_test_gb = gb_model.predict(X_test)

training_accuracy_gb = accuracy_score(Y_train, y_pred_train_gb)
test_accuracy_gb = accuracy_score(Y_test, y_pred_test_gb)

print("Gradient Boosting - Training Accuracy:", np.round(training_accuracy_gb, 2))
print("Gradient Boosting - Test Accuracy:", np.round(test_accuracy_gb, 2))

# using grid search cv write down some combinations of n_estimators and max_samples and max_features

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1,10]
}

gb_model = GradientBoostingClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters and the best score
print("Best hyperparameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


# Evaluate the best model on the test set
best_gb_model = grid_search.best_estimator_
y_pred_test_best = best_gb_model.predict(X_test)
test_accuracy_best = accuracy_score(Y_test, y_pred_test_best)
print("Test accuracy of best model:", test_accuracy_best)

