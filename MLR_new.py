import pandas as pd
import numpy as np
import seaborn as sns

#Read the data
df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\Cars_4vars.csv")
df.shape
df.head()
df.info()

#check for missing values
df.isna().sum()

# Correlation Matrix
df.corr()
sns.pairplot(df)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

y_multi = df["MPG"]

# Define different models based on adding features step by step from highest correlation with MPG

# Model 1: Using only HP (highest correlation with MPG)
X_model1 = df[['HP']]
model1 = LinearRegression()
model1.fit(X_model1, y_multi)
y_pred1 = model1.predict(X_model1)
r2_model1 = r2_score(y_multi, y_pred1)
rmse_model1 = np.sqrt(mean_squared_error(y_multi, y_pred1))

# Model 2: Using HP + SP (next highest correlation)
X_model2 = df[['HP', 'SP']]
model2 = LinearRegression()
model2.fit(X_model2, y_multi)
y_pred2 = model2.predict(X_model2)
r2_model2 = r2_score(y_multi, y_pred2)
rmse_model2 = np.sqrt(mean_squared_error(y_multi, y_pred2))

# Model 3: Using HP + SP + VOL (next highest correlation)
X_model3 = df[['HP', 'SP', 'VOL']]
model3 = LinearRegression()
model3.fit(X_model3, y_multi)
y_pred3 = model3.predict(X_model3)
r2_model3 = r2_score(y_multi, y_pred3)
rmse_model3 = np.sqrt(mean_squared_error(y_multi, y_pred3))

# Model 4: Using HP + SP + VOL + WT (all variables)
X_model4 = df[['HP', 'SP', 'VOL', 'WT']]
model4 = LinearRegression()
model4.fit(X_model4, y_multi)
y_pred4 = model4.predict(X_model4)
r2_model4 = r2_score(y_multi, y_pred4)
rmse_model4 = np.sqrt(mean_squared_error(y_multi, y_pred4))

# Store results in a dataframe
model_results = pd.DataFrame({
    "Model": ["HP Only", "HP + SP", "HP + SP + VOL", "HP + SP + VOL + WT"],
    "R-Squared": [r2_model1, r2_model2, r2_model3, r2_model4],
    "RMSE": [rmse_model1, rmse_model2, rmse_model3, rmse_model4]
})

# Display the results
print(model_results)  

# Model 3: Using HP + SP + VOL (next highest correlation)
X_model3 = df[['HP',  'VOL']]
model3 = LinearRegression()
model3.fit(X_model3, y_multi)
y_pred3 = model3.predict(X_model3)
r2_model3 = r2_score(y_multi, y_pred3)
rmse_model3 = np.sqrt(mean_squared_error(y_multi, y_pred3))


#=============================================================================

# Define independent variable (X) as SP and dependent variable (Y) as HP
X_sp = df[['SP']]
y_hp = df['HP']

# Fit a linear regression model
model_hp_sp = LinearRegression()
model_hp_sp.fit(X_sp, y_hp)

# Predict HP using SP
y_hp_pred = model_hp_sp.predict(X_sp)

# Calculate R-squared for the model
r2_hp_sp = r2_score(y_hp, y_hp_pred)

# Calculate VIF using R-squared
vif_hp_sp = 1 / (1 - r2_hp_sp)

# Display the results
print("R square",r2_hp_sp)
print("VIF: ",vif_hp_sp)


#=============================================================================

# Define independent variable (X) as VOL and dependent variable (Y) as WT
X_vol = df[['VOL']]
y_wt = df['WT']

# Fit a linear regression model
model_hp_sp = LinearRegression()
model_hp_sp.fit(X_vol, y_wt)

# Predict HP using SP
y_wt_pred = model_hp_sp.predict(X_vol)

# Calculate R-squared for the model
r2_vol_wt = r2_score(y_wt, y_wt_pred)

# Calculate VIF using R-squared
vif_vol_wt = 1 / (1 - r2_vol_wt)

# Display the results
print("R square",r2_vol_wt)
print("VIF: ",vif_vol_wt)

#=============================================================================

# Define independent variable (X) as VOL and dependent variable (Y) as HP
X_vol = df[['VOL']]
y_hp = df['HP']

# Fit a linear regression model
model_hp_sp = LinearRegression()
model_hp_sp.fit(X_vol, y_hp)

# Predict HP using SP
y_hp_pred = model_hp_sp.predict(X_vol)

# Calculate R-squared for the model
r2_vol_hp = r2_score(y_hp, y_hp_pred)

# Calculate VIF using R-squared
vif_vol_hp = 1 / (1 - r2_vol_hp)

# Display the results
print("R square",r2_vol_hp)
print("VIF: ",vif_vol_hp)


#=======================================================

# Model 2: Using HP + VOL (next highest correlation)
X_best = df[['HP', 'VOL']]
model5 = LinearRegression()
model5.fit(X_best, y_multi)
y_pred2 = model5.predict(X_best)
r2_model2 = r2_score(y_multi, y_pred2)
rmse_model2 = np.sqrt(mean_squared_error(y_multi, y_pred2))


import matplotlib.pyplot as plt
import seaborn as sns

# Compute residuals for the HP + VOL model
residuals = y_multi - y_pred2

# Plot residuals vs. predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred2, y=residuals, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted MPG")
plt.ylabel("Residuals")
plt.title("Residual Plot for Homoscedasticity Check")
plt.show()



















