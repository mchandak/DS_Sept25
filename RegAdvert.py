import pandas as pd
import matplotlib.pyplot as plt

#Loading data
data = pd.read_csv(r"D:\Manoj\1ExcelR\Data\Advertising.csv")

data.isnull().sum()

#Simple linear regression
# Input variable
X=data.iloc[:,1:2].values
# Output variable
y=data.iloc[:,4].values

#Split Data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,
                        test_size=.25,random_state=1)
#Model Building & Prediction
from sklearn.linear_model import LinearRegression
reg =LinearRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
#Model evaluation

reg.score (X_train, y_train)
reg.score (X_test, y_test)
#coefficient & intercept values
print(reg.coef_)
print(reg.intercept_)

#calculating y for new value of x
#y = b0 + b1x       (y=mx+c)
x=100
y1 =  reg.intercept_ + reg.coef_ * x
y1


#Scatter plot
plt.scatter(X_train,y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title('TV Advertising')
plt.xlabel('TV')
plt.ylabel('Sales')


plt.scatter(X_test,y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('TV Advertising')
plt.xlabel('TV')
plt.ylabel('Sales')

####
