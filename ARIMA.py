import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Time series specific libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools

# Load the CSV file
df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\footfalls.csv")

df.info()
df.head(5)
df.isnull().sum()

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

# Convert 'Month' column to datetime format
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')

# Sort by date to ensure chronological order
df = df.sort_values('Month')

# Set 'Month' as index for time series operations
# This is required for time series analysis
df.set_index('Month', inplace=True)

# ============================================================================
# VISUALIZATION AND DECOMPOSITION
# ============================================================================
# Perform time series decomposition
# model='additive' assumes: Y(t) = Trend + Seasonal + Residual
# period=12 represents monthly seasonality (12 months in a year)
decomposition = seasonal_decompose(df['Footfalls'], model='additive', period=12)

# Create figure with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# PLOT 1 - All properties on axes[0]
axes[0].plot(df.index, df['Footfalls'], color='blue', linewidth=1.5, marker='.')
axes[0].set_title('Original Time Series - Footfalls', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Footfalls', fontsize=11)
axes[0].grid(True, alpha=0.3)

# PLOT 2 - All properties on axes[1]
axes[1].plot(decomposition.trend, color='green', linewidth=1.5)
axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Trend', fontsize=11)
axes[1].grid(True, alpha=0.3)

# PLOT 3 - All properties on axes[2]
axes[2].plot(decomposition.seasonal, color='orange', linewidth=1.5)
axes[2].set_title('Seasonal Component', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Seasonal', fontsize=11)
axes[2].grid(True, alpha=0.3)

# PLOT 4 - All properties on axes[3]
axes[3].plot(decomposition.resid, color='red', linewidth=1.5)
axes[3].set_title('Residual Component', fontsize=12, fontweight='bold')
axes[3].set_xlabel('Date', fontsize=11)
axes[3].set_ylabel('Residual', fontsize=11)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

# ============================================================================
# ADF TEST FOR STATIONARITY
# ============================================================================
from statsmodels.tsa.stattools import adfuller

def adf_test(series, title=''):
  
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')
    
    labels = ['ADF Statistic', 'p-value', '# Lags Used', '# Observations Used']
    out = dict(zip(labels, result[0:4]))
    
    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
    
    # Print results nicely
    for k, v in out.items():
        print(f'{k}: {v}')
    
    print('Stationary' if out['p-value'] < 0.05 else 'Non-stationary')
    
    return out

# Test original series
adf_result_original = adf_test(df['Footfalls'], 'Original Series')
print(adf_result_original['p-value'].round(2))

# Create first difference
# Differencing removes trend to make series stationary
df['Footfalls_diff1'] = df['Footfalls'].diff()
adf_result_diff1 = adf_test(df['Footfalls_diff1'], 'First Difference')
print(adf_result_diff1['p-value'].round(2))


# ============================================================================
# GRID SEARCH FOR OPTIMAL ARIMA PARAMETERS
# ============================================================================

print("\nPerforming exhaustive grid search...")

# Define parameter ranges
# p: AR terms [0,1,2,3]
# d: Differencing [0,1,2]
# q: MA terms [0,1,2,3]
p_values = range(0, 4)
d_values = range(0, 3)
q_values = range(0, 4)

best_aic = np.inf
best_order = None
best_model = None
results_list = []

# Iterate through all combinations
total = len(p_values) * len(d_values) * len(q_values)
print(f"Testing {total} ARIMA combinations...\n")

for p, d, q in itertools.product(p_values, d_values, q_values):
    try:
        # Fit ARIMA model with current parameters
        model = ARIMA(df['Footfalls'], order=(p, d, q))
        fitted_model = model.fit()
        
        aic = fitted_model.aic
         
        # Store results
        results_list.append({
            'order': (p, d, q),
            'AIC': aic
            
        })
        
        # Update best model if AIC is lower
        if aic < best_aic:
            best_aic = aic
            best_order = (p, d, q)
            best_model = fitted_model
        
        print(f"ARIMA{(p, d, q)} - AIC: {aic:8.2f}")
        
    except Exception as e:
        # Skip combinations that fail
        continue

print(f"BEST MODEL: ARIMA{best_order}")
print(f"AIC: {best_aic:.2f}")

# Use ARIMA(2,1,2) for better stability and interpretability
print("\nFitting ARIMA(2, 1, 2) model on full dataset...")

final_model = ARIMA(df['Footfalls'], order=(2, 1, 2))
final_fitted = final_model.fit()

# Generate 6-month ahead forecast
future_steps = 6
forecast_result = final_fitted.get_forecast(steps=future_steps)
future_forecast = forecast_result.predicted_mean
future_forecast 


#####################################

# ============================================================================
# ACF AND PACF ANALYSIS
# ============================================================================

# ACF: Autocorrelation Function - shows correlation with lagged values
# PACF: Partial Autocorrelation Function - shows direct correlation after removing intermediate lags
# These plots help determine p and q parameters

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# ACF for original series
plot_acf(df['Footfalls'].dropna(), lags=40, ax=axes[0, 0], color='blue')
axes[0, 0].set_title('ACF - Original Series', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# PACF for original series
plot_pacf(df['Footfalls'].dropna(), lags=40, ax=axes[0, 1], color='blue')
axes[0, 1].set_title('PACF - Original Series', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# ACF for first differenced series
plot_acf(df['Footfalls_diff1'].dropna(), lags=40, ax=axes[1, 0], color='green')
axes[1, 0].set_title('ACF - First Differenced Series', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# PACF for first differenced series
plot_pacf(df['Footfalls_diff1'].dropna(), lags=40, ax=axes[1, 1], color='green')
axes[1, 1].set_title('PACF - First Differenced Series', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()