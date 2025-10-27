import pandas as pd

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
data1 = pd.get_dummies(data1, columns=['Gender','Married','Dependents','Education','Self_Employed',
                                       'Credit_History','Property_Area'],dtype=int)

X = data1.values
y = data.iloc[:,-1].values
#Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion="gini", max_depth=3,min_samples_leaf=5,random_state = 10)
#clf = DecisionTreeClassifier(criterion="gini",max_depth=10,min_samples_leaf=1,random_state = 10)     # Overfitting
#clf = DecisionTreeClassifier(criterion="gini",random_state = 10)     # Overfitting

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3,min_samples_leaf=5,random_state = 10)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Training Accuracy : {:.3f}'.format(clf.score(X_train, y_train)))
print('Testing Accuracy : {:.3f}'.format(clf.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred,labels=('Y','N'))
print(cm)


#Cross Validationn

from sklearn.model_selection import cross_val_score,KFold
kf = KFold(5,shuffle=True, random_state= 1)
accuracies = cross_val_score(clf,X,y,cv=kf)
print('{:.3f}'.format(accuracies.mean()))



##########
#Optional
#!pip install graphviz
#!pip install pydotplus
#conda install graphviz

from sklearn import tree
import pydotplus as pdtp
from io import StringIO

#Chart
len2= len(data1.columns)
features = list(data1.columns[0:len2])

dot_data= StringIO();
tree.export_graphviz(clf,out_file=dot_data,feature_names=features,impurity = False)
graph = pdtp.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('treedemo.pdf')

df1= pd.DataFrame(X_train,columns=data1.columns)
df1['Loan_Status']=y_train
df1.to_csv('TreeTrain.csv')

