import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import os
import random
# Set seeds for reproducibility

seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Load data
os.chdir("D:\\Manoj\\1ExcelR\\Data")
df = pd.read_csv('mobiles_sales.csv')

features = ['Product_Views', 'Add_to_Cart_Count', 'Current_Price', 
            'Competitor_Price', 'Promotion_Active', 'Is_Holiday_or_Weekend', 
            'Stock_Availability', 'Sales_Count']

df = df[features]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # All features except target
        y.append(data[i+seq_length, -1])     # Target: Sales_Count
    return np.array(X), np.array(y)

sequence_length = 6  # Past 6 hours
X, y = create_sequences(scaled_data, sequence_length)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ------------------------
# Build the LSTM Model
# ------------------------
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

# Inverse transform the Sales_Count to original scale
sales_scaler = MinMaxScaler()
sales_scaler.min_, sales_scaler.scale_ = scaler.min_[-1], scaler.scale_[-1]

y_test_orig = sales_scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_orig = sales_scaler.inverse_transform(y_pred)


