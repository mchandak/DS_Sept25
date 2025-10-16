import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("D:\\Manoj\\1ExcelR\\Data")
data = pd.read_csv('loan prediction.csv')

data.isnull().sum()

data.info()
# Pre-processing
data.isnull().sum()

#Fill in with most frequent category (mode)

data['Gender'].fillna(data['Gender'].mode()[0],inplace= True)
data['Married'].fillna(data['Married'].mode()[0],inplace= True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace= True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace= True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace= True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace= True)

data['LoanAmount'].describe()
data['LoanAmount'].fillna(data['LoanAmount'].median(),inplace= True)

#Dummy variable
data1= data.iloc[:,1:-1]       # Exclude id and status column
data1 = pd.get_dummies(data1, columns=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area'],dtype=int)

X = data1.values
y = data.iloc[:,-1].values
#Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)     #fit_transform on X_train
X_test = sc.transform(X_test)           #transform on X_test


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)         #k is hyperparameter
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Training Accuracy : {:.3f}'.format(knn.score(X_train, y_train)))
print('Testing Accuracy : {:.3f}'.format(knn.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred,labels=('Y','N'))

print(cm)

########

#Cross Validationn

from sklearn.model_selection import cross_val_score,KFold             
kf = KFold( n_splits=5,shuffle=True, random_state= 42)               # Data not scaled
accuracies = cross_val_score(knn,X,y,cv=kf)
print('{:.3f}'.format(accuracies.mean()))

#Pipeline                                                   # Data scaled
from sklearn.pipeline import make_pipeline
clf = make_pipeline(sc, knn)
accuracies = cross_val_score(clf,X,y,cv=kf)
print('{:.3f}'.format(accuracies.mean()))


###############
# Elbow method to get optimum value of k

neighbors=range(1,11)
k_score=[]
for n in neighbors:
    knn1=KNeighborsClassifier(n_neighbors=n)
    clf1=make_pipeline(sc,knn1)
    accuracies1=cross_val_score(clf1,X,y,cv=kf)
    k_score.append(1-accuracies1.mean())
    
k_score
np.array(k_score).round(4)

plt.plot(neighbors,k_score)
plt.ylabel('Error')
plt.xlabel('Number of Neighbors')
plt.locator_params(axis='x', nbins=20)


#Model using optimized k value
from sklearn.pipeline import make_pipeline
knn1 = KNeighborsClassifier(n_neighbors = 7)
clf1 = make_pipeline(sc, knn1)
accuracies = cross_val_score(clf1,X,y,cv=kf)
print('{:.3f}'.format(accuracies.mean()))

##########################














# Optimization
#from sklearn.model_selection import GridSearchCV
#param_grid = { 'n_neighbors': [3,5,7,9,11,13,15]}
#grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
#grid.fit(X_train, y_train)
#print(grid.best_score_.round(5))

# allscores=grid.cv_results_['mean_test_score']
# print(allscores)
