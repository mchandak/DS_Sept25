import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Logistic regression
#Loading dataset

os.chdir("D:\\Manoj\\1ExcelR\\Data")
data = pd.read_csv('loan prediction.csv')

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 100)

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)     #fit_transform on X_train
X_test = sc.transform(X_test)           #transform on X_test


# Model training
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# Prediction
y_pred = log_reg.predict(X_test)

#y_prob = log_reg.predict_proba(X_test)

print('Training Accuracy : {:.3f}'.format(log_reg.score(X_train, y_train)))
print('Testing Accuracy : {:.3f}'.format(log_reg.score(X_test, y_test)))

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)
print(cm1)
cm2 = confusion_matrix(y_test, y_pred, labels=['Y', 'N'])
print(cm2)

#acc= (112+13)/(112+1+28+13)
#round(acc,3)

# df1 = pd.DataFrame()
# df1['Y_Test'] = y_test
# df1['Y_Pred'] = y_pred
# df1.to_csv("TestandPred.csv", index= False)


from sklearn.metrics import recall_score, precision_score, f1_score

#Other ML Performance metrics

#Recall (Sensitivity)  (TP/(TP+FN))
recall1= cm2[0,0]/(cm2[0,0]+cm2[0,1])
print("Recall:", np.round(recall1,2))
recall2 = recall_score(y_test,y_pred,pos_label='Y')
print("Recall:", np.round(recall2,2))

# Calculate precision
#np.round(cm2[0,0]/(cm2[0,0]+cm2[1,0]),2)
precision = precision_score(y_test,y_pred,pos_label='Y')
print("Precision:", np.round(precision,2))

# Calculate F1-score
f1 = f1_score(y_test,y_pred,pos_label='Y')
print("F1-score:", np.round(f1,2))

# Calculate specificity - TN / (TN + FP)
np.round(cm2[1,1]/(cm2[1,1]+cm2[1,0]),2)

from sklearn.metrics import roc_curve, auc
# Predictions on the test set
y_prob = log_reg.predict_proba(X_test)[:,1]

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob,pos_label='Y')
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print(f"AUC score: {roc_auc}")

######################

#Cross Validationn

from sklearn.model_selection import cross_val_score,KFold
log_reg1 = LogisticRegression(max_iter=200)
kf = KFold(5,shuffle=True, random_state= 1)
accuracies = cross_val_score(log_reg1,X,y,cv=kf)
#accuracies = cross_val_score(log_reg1,X,y,cv=5)
print('{:.3f}'.format(accuracies.mean()))

#################



# #Pipeline                                                   # Data scaled
# from sklearn.pipeline import make_pipeline
# clf = make_pipeline(sc, log_reg)
# accuracies = cross_val_score(clf,X,y,cv=10)
# print('{:.5f}'.format(accuracies.mean()))

######################




import math

y = -10
math.exp(-y)
p = 1 / (1 + math.exp(-y))
print(p)


x= 10 ** -5
print(format(x, ".10f"))  









