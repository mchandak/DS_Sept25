import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\Advertising_CL.csv")

# Display basic info and the first few rows
df.info()
df.head()

# Convert 'Timestamp' to datetime and extract new features
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M')
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6

# Select features (exclude non-numeric / ID-like columns)
X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male', 'Hour', 'Day']]

# Target variable
y = df['Clicked on Ad']

from sklearn.model_selection import train_test_split

# Split the data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Stratify splitting - proportion of each class in your target variable - mainly in classification model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Shapes of the splits
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize and fit the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# Make predictions on the training and test sets
y_train_pred = log_reg.predict(X_train)
y_test_pred = log_reg.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the accuracy scores
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

#==================================================================================
# shuffle split cross validation
#==================================================================================


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Set up ShuffleSplit cross-validator
from sklearn.model_selection import ShuffleSplit, cross_validate
shuffle_split = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)  #Split 10 times

# Perform cross-validation and collect both train and test scores
cv_results = cross_validate(model, X, y, cv=shuffle_split, scoring='accuracy', return_train_score=True)

# Extract train and test scores
train_scores = cv_results['train_score']
test_scores = cv_results['test_score']

# Show individual scores and their means
print("cross validation: train accuracy:" , np.round(train_scores.mean(),2))
print("cross validation: test accuracy:" , np.round(test_scores.mean(),2))

#=============================================================================
# K-Fold cross validation
#=============================================================================

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.model_selection import KFold, cross_validate
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and collect both train and test scores
cv_results = cross_validate(model, X, y, cv=shuffle_split, scoring='accuracy', return_train_score=True)

# Extract train and test scores
train_scores = cv_results['train_score']
test_scores = cv_results['test_score']

# Show individual scores and their means
print("cross validation: train accuracy:" , np.round(train_scores.mean(),2))
print("cross validation: test accuracy:" , np.round(test_scores.mean(),2))
