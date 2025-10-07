"""
Created on Mon Jan 27 20:49:03 2025
"""
import numpy as np
import pandas as pd
sample = pd.read_csv(r"D:\Manoj\1ExcelR\Data\Bulbs_Lifespan.csv")

import pandas as pd
from scipy.stats import ttest_1samp

# Hypothesized population mean
pop_mean = 1000  # Null hypothesis: mean = 1000

# Perform the one-sample t-test
t_statistic, p_value = ttest_1samp(sample, pop_mean)

# Significance level
alpha = 0.05

#=============================================================================
# Results
print("T-Statistic: ",np.round(t_statistic,4))
print("p_value: ",np.round(p_value,4))

if p_value < alpha:
    print("Reject the null hypothesis: The mean lifespan is significantly different from 1000 hours.")
else:
    print("Fail to reject the null hypothesis: No significant difference from 1000 hours.")
    
#=============================================================================    
# For Alternative Hypothesis: Less Than (𝐻1:𝜇<1000)
    
if t_statistic < 0:  # Test statistic must point in the correct direction
    one_tailed_p_value = p_value / 2
    if one_tailed_p_value < alpha:
        print("Reject the null hypothesis: The mean lifespan is significantly less than 1000 hours.")
    else:
        print("Fail to reject the null hypothesis: No significant evidence that the mean lifespan is less than 1000 hours.")
else:
    print("Fail to reject the null hypothesis: Test statistic does not support H1 (mean < 1000).")

#=============================================================================
# For Alternative Hypothesis: Less Than (𝐻1:𝜇>1000)
if t_statistic > 0:  # Test statistic must point in the correct direction
    one_tailed_p_value = p_value / 2
    if one_tailed_p_value < alpha:
        print("Reject the null hypothesis: The mean lifespan is significantly greater than 1000 hours.")
    else:
        print("Fail to reject the null hypothesis: No significant evidence that the mean lifespan is greater than 1000 hours.")
else:
    print("Fail to reject the null hypothesis: Test statistic does not support H1 (mean > 1000).")


#=============================================================================
