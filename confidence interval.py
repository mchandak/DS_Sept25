"""
Created on Thu Apr 10 12:44:17 2025
"""

import pandas as pd
df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\Lungcapdata.csv")
df

df["LungCap"].describe()
df["LungCap"].mean()
#-------------------------------------------------------------------------------


from scipy import stats

# 90%
df_ci = stats.norm.interval(0.90, loc=df["LungCap"].mean(),
                                  scale=df["LungCap"].std())

print("I am 90% confident that mean will be lies under this interval:",df_ci)

#--------------------------------------------------------------------------------
df_ci = stats.norm.interval(0.95,
                                  loc=df["LungCap"].mean(),
                                  scale=df["LungCap"].std())

print("I am 95% confident that mean will be lies under this interval:",df_ci)

#--------------------------------------------------------------------------------
# 99%
df_ci = stats.norm.interval(0.99,
                                  loc=df["LungCap"].mean(),
                                  scale=df["LungCap"].std())

print("I am 99% confident that mean will be lies under this interval:",df_ci)
#--------------------------------------------------------------------------------
