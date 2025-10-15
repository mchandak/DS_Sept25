
import pandas as pd
df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\Diab.csv")
df

df.info()

df_cat = df[["Gender",'Diabetic']]
df_cat

df_cont = df.iloc[:,2:6]
df_cont

# standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(df_cont)
SS_X = pd.DataFrame(SS_X)
SS_X.columns = ["OGTT", "DBP", "BMI","Age"]
SS_X

# scaling/normalizatin
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
MM_X = MM.fit_transform(df_cont)
MM_X = pd.DataFrame(MM_X)
MM_X.columns = ["OGTT", "DBP", "BMI","Age"]
MM_X

#============================================================
# label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df_cat["Gender_LE"] = LE.fit_transform(df_cat["Gender"])
df_cat["Diabetic_LE"] = LE.fit_transform(df_cat["Diabetic"])
df_cat.head()

#============================================================
# OneHotencoding
from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder()

df_g = pd.DataFrame(OHE.fit_transform(df_cat[["Gender"]]).toarray())
df_g.columns = ['Female','Male']
df_g

df_d = pd.DataFrame(OHE.fit_transform(df_cat[["Diabetic"]]).toarray())
df_d.columns = ['No','Yes']
df_d

#============================================================
























































