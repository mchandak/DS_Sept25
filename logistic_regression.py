import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"D:\Manoj\1ExcelR\Data\Advertising_CL.csv")

# Display basic info and the first few rows
df.info()
df.head()

# Correlation heatmap of numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

"""Here’s what we can observe from the EDA visuals:

**Correlation Heatmap:**

*   Daily Time Spent on Site and Daily Internet Usage have a negative correlation with each other.

*   Clicked on Ad is negatively correlated with Daily Internet Usage and positively correlated with Daily Time Spent on Site and Age.

*   No multicollinearity issues are apparent.


"""

# Distribution of Age
plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.show()

"""**Distribution of Age:**

Most users are clustered between 25–45 years, with a slight right skew.
"""

# Distribution of Area Income
plt.figure(figsize=(6, 4))
sns.histplot(df['Area Income'], bins=20, kde=True)
plt.title('Distribution of Area Income')
plt.xlabel('Area Income')
plt.show()

"""**Distribution of Area Income:**

The income distribution appears fairly normal, centered around $55,000.
"""

# Boxplot of Daily Time Spent on Site by Clicked on Ad
plt.figure(figsize=(6, 4))
sns.boxplot(x='Clicked on Ad', y='Daily Time Spent on Site', data=df)
plt.title('Daily Time Spent on Site vs Clicked on Ad')
plt.show()

"""
**Daily Time Spent on Site vs Clicked on Ad:**
Users who clicked on the ad tend to have lower daily time spent on site, which is interesting and worth exploring further."""

# Boxplot of Daily Internet Usage by Clicked on Ad
plt.figure(figsize=(6, 4))
sns.boxplot(x='Clicked on Ad', y='Daily Internet Usage', data=df)
plt.title('Daily Internet Usage vs Clicked on Ad')
plt.show()

"""**Daily Internet Usage vs Clicked on Ad:**

Those who clicked generally have lower daily internet usage, while non-clickers tend to have higher internet use.
"""

# Convert 'Timestamp' to datetime and extract new features
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M')
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6

# Check the new columns
df[['Timestamp', 'Hour', 'Day']].head()

"""We've successfully extracted two new features from Timestamp:

**Hour:** The hour of the day (0–23).

**Day:** The day of the week (0=Monday, 6=Sunday).
"""

# Clicked on Ad by Hour
plt.figure(figsize=(8, 4))
sns.countplot(x='Hour', hue='Clicked on Ad', data=df)
plt.title('Ad Clicks by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.show()


# Clicked on Ad by Day of Week
plt.figure(figsize=(8, 4))
sns.countplot(x='Day', hue='Clicked on Ad', data=df)
plt.title('Ad Clicks by Day of Week')
plt.xlabel('Day of Week (0=Monday)')
plt.ylabel('Count')
plt.show()

"""**Ad Clicks by Day of Week:**

Clicks are fairly evenly distributed across the week, with no major peaks or drops by weekday.


"""

# Define feature set and target
X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male', 'Hour', 'Day']]
y = df['Clicked on Ad']

from sklearn.linear_model import LogisticRegression
# Initialize and fit Logistic Regression model
logmodel = LogisticRegression()
logmodel.fit(X, y)

# Predictions
predictions = logmodel.predict(X)

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

accuracy = accuracy_score(y, predictions)
print("Accuracy score:", accuracy)

conf_matrix = confusion_matrix(y, predictions)
conf_matrix

class_report = classification_report(y, predictions)
print(class_report)


import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

# Assuming 'y' and 'predictions' are already defined from the previous code

# Calculate recall
recall = recall_score(y, predictions)
print("Recall:", np.round(recall,2))

# Calculate precision
precision = precision_score(y, predictions)
print("Precision:", np.round(precision,2))

# Calculate F1-score
f1 = f1_score(y, predictions)
print("F1-score:", np.round(f1,2))

# Calculate specificity
tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
specificity = tn / (tn + fp)
print("Specificity:", np.round(specificity,2))

from sklearn.metrics import roc_curve, auc

# Predictions on the test set
predictions = logmodel.predict(X)
y_pred_proba = logmodel.predict_proba(X)[:,1]

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
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

##########################

conf_matrix = confusion_matrix(y, predictions,labels=[1, 0])
conf_matrix

class_report = classification_report(y, predictions,labels=[1, 0])
print(class_report)


import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

# Assuming 'y' and 'predictions' are already defined from the previous code

# Calculate recall
recall = recall_score(y, predictions,pos_label=1)
print("Recall:", np.round(recall,2))

# Calculate precision
precision = precision_score(y, predictions,pos_label=1)
print("Precision:", np.round(precision,2))

# Calculate F1-score
f1 = f1_score(y, predictions,pos_label=1)
print("F1-score:", np.round(f1,2))


