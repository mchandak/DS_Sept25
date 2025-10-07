
import pandas as pd
from statsmodels.stats.weightstats import ztest

# Load the dataset
df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\ecommerce_two_sample_ztest_dataset.csv")

# Separate the data into two groups
email_group = df[df['Group'] == 'Email Marketing']['Purchase_Value']
social_group = df[df['Group'] == 'Social Media Ads']['Purchase_Value']

email_group.describe()
social_group.describe()

# Perform the two-sample Z-test
z_stat, p_value = ztest(email_group, social_group, alternative='two-sided')
z_stat
p_value
# Define the level of significance
alpha = 0.05

# Decision based on p-value
if p_value < alpha:
    conclusion = "Reject the null hypothesis: There is a significant difference in average purchase value."
else:
    conclusion = "Fail to reject the null hypothesis: There is no significant difference in average purchase value."

# Output results
print(f"Z-Statistic: {z_stat:.2f}")
print(f"P-Value: {p_value:.4f}")
print(f"Conclusion: {conclusion}")
