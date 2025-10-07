import numpy as np
import pandas as pd
df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\\Bulbs_Lifespan.csv")

from statsmodels.stats.weightstats import ztest


# Given data
population_mean = 1000  # Hypothesized population mean
alpha = 0.05  # Level of significance

# Step 1: Perform a one-sample Z-test
z_stat, p_value = ztest(df["Lifespan (Hours)"], value=population_mean, alternative='two-sided')

# Step 2: Decision
if p_value < alpha:
    conclusion = "Reject the null hypothesis: The average lifespan is significantly different from 1000 hours."
else:
    conclusion = "Fail to reject the null hypothesis: There is no significant difference in the average lifespan."

# Output results
print(f"Z-Statistic: {z_stat:.2f}")
print(f"P-Value: {p_value:.4f}")
print(f"Conclusion: {conclusion}")


