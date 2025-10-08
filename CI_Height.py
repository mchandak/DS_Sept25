import pandas as pd

# Load data
df = pd.read_csv(r'D:\Manoj\1ExcelR\Data\Height.csv')

# Basic stats
mean_height = df['Height_cm'].mean()
std_height = df['Height_cm'].std()
n = len(df)

# 95% Confidence Interval
from scipy.stats import t
import numpy as np

t_crit = t.ppf(0.975, df=n-1)
margin = t_crit * (std_height / np.sqrt(n))

ci_lower = mean_height - margin
ci_upper = mean_height + margin

print(f"Mean height: {mean_height:.2f} cm")
print(f"95% Confidence Interval: {ci_lower:.2f} cm to {ci_upper:.2f} cm")

#We are 95% confident that the true population mean height lies 
#between [171.44 cm and 173.29 cm].