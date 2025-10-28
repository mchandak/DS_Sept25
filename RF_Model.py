import pandas as pd
import matplotlib.pyplot as plt
import os

#Loading dataset

os.chdir(r"D:\Manoj\1ExcelR\Data")
data = pd.read_csv("loan prediction.csv")

# Pre-processing
data.isnull().sum()

#Fill in with most frequent category (mode)
data['Gender'].fillna(data['Gender'].mode()[0],inplace= True)
data['Married'].fillna(data['Married'].mode()[0],inplace= True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace= True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace= True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace= True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace= True)

data['LoanAmount'].fillna(data['LoanAmount'].median(),inplace= True)

#Dummy variable
data1= data.iloc[:,1:-1]       # Exclude id and status column
data1 = pd.get_dummies(data1, columns=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area'],dtype=int)

X = data1.values
y = data.iloc[:,-1].values
#Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='gini',n_estimators=100,max_depth=3,
                                    min_samples_leaf=5,random_state=10)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print('Training Accuracy : {:.3f}'.format(classifier.score(X_train, y_train)))
print('Testing Accuracy : {:.3f}'.format(classifier.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred , labels= ('Y','N'))
print(cm)

#Cross Validationn
from sklearn.model_selection import cross_val_score,KFold
kf = KFold(5,shuffle=True, random_state= 1)
accuracies = cross_val_score(classifier,X,y,cv=kf)
print('{:.3f}'.format(accuracies.mean()))

############################

#Gridsearch
from sklearn.model_selection import GridSearchCV
param_grid = { 
           'n_estimators' : [ 50,100,200],
           'max_depth' : [ 5,7,9],
           'min_samples_leaf' : [ 2, 5, 10],
           'criterion':['gini','entropy']
           }

CV_rfc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= kf)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_score_.round(5))
y_pred = CV_rfc.predict(X_test)
print('Testing Accuracy : {:.3f}'.format(CV_rfc.score(X_test, y_test)))
print (CV_rfc.best_params_)

classifier1 = RandomForestClassifier(criterion= 'gini',n_estimators=50,max_depth=5,min_samples_leaf=2,random_state=10)
classifier1.fit(X_train, y_train)

y_pred = classifier1.predict(X_test)

print('Training Accuracy : {:.3f}'.format(classifier1.score(X_train, y_train)))
print('Testing Accuracy : {:.3f}'.format(classifier1.score(X_test, y_test)))

###########################

#Feature importance
dict1 = {'feature':data1.columns,'importance':classifier.feature_importances_}
df1 = pd.DataFrame(dict1)
df1.sort_values('importance',inplace=True)
df1.plot.barh('feature','importance',legend=False)
plt.tight_layout()

##########################












