from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Set seeds for reproducibility
seed_value = 42
tf.random.set_seed(seed_value)

# Load data
os.chdir("D:\\Manoj\\1ExcelR\\Data")
data1 = pd.read_csv('daily_energy_consumption.csv')

# Convert 'Date' to datetime and set as index
data1['Date'] = pd.to_datetime(data1['Date'], format='%d-%m-%Y')
data1.set_index('Date', inplace=True)

# Select the target column
consumption_data = data1[['Consumption (kWh)']]

# Normalize values to range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
consumption_scaled = scaler.fit_transform(consumption_data)

# Create sequences: past 30 days → next day
X = []
y = []
n_steps = 30

for i in range(n_steps, len(consumption_scaled)):
    X.append(consumption_scaled[i-n_steps:i, 0])
    y.append(consumption_scaled[i, 0])

X, y = np.array(X), np.array(y)

# Reshape X for RNN input: (samples, time_steps, features)
#samples → how many examples you’re training on
#time_steps → how many past points (days) the model looks back
#features → how many variables per day (in your case, just one — consumption)

X = X.reshape((X.shape[0], X.shape[1], 1))
print("X shape:", X.shape)
print("y shape:", y.shape)

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# Predict next day's consumption
last_30_days = consumption_scaled[-30:]
last_30_days = last_30_days.reshape((1, n_steps, 1))

predicted_scaled = model.predict(last_30_days)
predicted_kwh = scaler.inverse_transform(predicted_scaled)

print(f"Predicted next day's consumption: {predicted_kwh[0][0]:.2f} kWh")


######################################################







































