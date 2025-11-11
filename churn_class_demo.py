from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import os


# Set seeds for reproducibility
seed_value = 42
tf.random.set_seed(seed_value)

os.chdir("D:\\Manoj\\1ExcelR\\Data")
data = pd.read_csv("Churn_Modelling.csv")

# Select features (starting from column 3, excluding target)
data1 = data.iloc[:, 3:-1]

# One-hot encode categorical variables
data1 = pd.get_dummies(data1, columns=['Geography', 'Gender'], dtype=int)

# Drop one category from each categorical to avoid multicollinearity
data1.drop(['Geography_Spain', 'Gender_Male'], axis=1, inplace=True)

# Prepare X and y
X = data1.iloc[:, :].values
y = data.iloc[:, -1].values

# Split with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=10,
    stratify=y  # Maintains class distribution
)

# Critical for ANN - brings all features to similar scale
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
X_train_scaled.shape
# ============================================================================
# BUILD ANN ARCHITECTURE
# ============================================================================
classifier = Sequential()

# Input layer + First hidden layer
# Why 6 units? Rule of thumb: (input_dim + output_dim) / 2 = (11 + 1) / 2 ≈ 6
classifier.add(Dense(
    units=6, 
    activation='relu',
    input_dim=11,
    name='hidden_layer_1'
))

# Dropout to prevent overfitting (20% of neurons randomly deactivated)
classifier.add(Dropout(rate=0.2))

# Second hidden layer
classifier.add(Dense(
    units=6,
    activation='relu',
    name='hidden_layer_2'
))

# Dropout
classifier.add(Dropout(rate=0.2))

# Output layer
# Binary classification: 1 neuron with sigmoid activation
# Sigmoid outputs probability between 0 and 1
classifier.add(Dense(
    units=1,
    activation='sigmoid',
    name='output_layer'
))

print("\nModel Architecture:")
classifier.summary()

# ============================================================================
# STEP 6: COMPILE THE MODEL
# ============================================================================

# For binary classification:
# - Loss: binary_crossentropy (cross-entropy for binary problems)
# - Optimizer: adam (adaptive learning rate)

classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # For binary classification
    metrics=['accuracy']
)

# ============================================================================
# TRAIN THE MODEL
# ============================================================================

# Train with validation split for monitoring
classifier.fit(
    X_train_scaled, y_train,
    batch_size=50,
    epochs=50,  # Increased from 10 to 50 for better learning
    validation_split=0.2,  # Use 20% of training for validation
    verbose=1
)

# Get probability predictions
y_pred_proba = classifier.predict(X_test_scaled, verbose=0)

# Convert probabilities to binary predictions (threshold = 0.5)
y_pred = np.where(y_pred_proba > 0.5, 1, 0).flatten()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm

accuracy = accuracy_score(y_test, y_pred)
accuracy



