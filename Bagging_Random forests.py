import pandas as pd
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# BAGGING CLASSIFIER

from sklearn.ensemble import BaggingClassifier

# Bagging Classifier
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='gini'),
                                  n_estimators=100, random_state=42,
                                  max_depth=3,min_samples_leaf=3)
bagging_model.fit(X_train, Y_train)

y_pred_train_bagging = bagging_model.predict(X_train)
y_pred_test_bagging = bagging_model.predict(X_test)

training_accuracy_bagging = accuracy_score(Y_train, y_pred_train_bagging)
test_accuracy_bagging = accuracy_score(Y_test, y_pred_test_bagging)

print("Bagging - Training Accuracy:", np.round(training_accuracy_bagging,2))
print("Bagging - Test Accuracy:", np.round(test_accuracy_bagging,2))


#---------------------------------------------------------2

from sklearn.ensemble import RandomForestClassifier

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42,
                                  max_depth=3,min_samples_leaf=3)
rf_model.fit(X_train, Y_train)

y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)

training_accuracy_rf = accuracy_score(Y_train, y_pred_train_rf)
test_accuracy_rf = accuracy_score(Y_test, y_pred_test_rf)

print("Random Forest - Training Accuracy:", np.round(training_accuracy_rf,2))
print("Random Forest - Test Accuracy:", np.round(test_accuracy_rf,2))

# Define the parameter grid
from sklearn.model_selection import GridSearchCV
param_grid = { 
           'n_estimators' : [ 50,100,200],
           'max_depth' : [ 5,7,9],
           'min_samples_leaf' : [ 2, 5, 10],
           'criterion':['gini','entropy']
           }

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters and the best score
print("Best hyperparameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Evaluate the best model on the test set
best_rf_model = grid_search.best_estimator_
y_pred_test_best = best_rf_model.predict(X_test)
test_accuracy_best = accuracy_score(Y_test, y_pred_test_best)
print("Test accuracy of best model:", round(test_accuracy_best,2))



