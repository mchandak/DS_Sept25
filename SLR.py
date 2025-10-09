
import pandas as pd
df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\experience_vs_salary.csv")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Defining the independent (X) and dependent (y) variables
X = df[["Experience (Years)"]]  # Independent variable (Years of Experience)
y = df["Salary ($)"]  # Dependent variable (Salary)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X, y)

# Making predictions using the same dataset (without data partitioning)
y_pred = model.predict(X)

# Calculating R-squared (R²) and Mean Squared Error (MSE)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

r2, mse

model.coef_
model.intercept_

x=12
y1 =  model.intercept_ + model.coef_ * x
y1
