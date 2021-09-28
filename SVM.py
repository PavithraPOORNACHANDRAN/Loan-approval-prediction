#Import the Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Create DataFrame
df = pd.read_csv('Dataset.csv')

#Data Analysis
df.shape
df.dtypes
df.head()
df = df.drop('Loan_ID', axis=1)
df.shape
df.tail()
df.sample(10)

# Describe the dataset
df.describe()

# check the unique labels of the target variable (Outcome)
df['Loan_Status'].unique()

# print the no.of labes for each class
print(df.Loan_Status.value_counts())

# convert into binary output
# mapping the target variable class into binary
df['Loan_Status']=df['Loan_Status'].map({'Y':0,'N':1})

# check the unique labels of the target variable(Outcome)
df['Loan_Status'].unique()

# print the no.of labes for each class
print(df.Loan_Status.value_counts())

# visulaize the class labels of target variable(Dataset)
import seaborn as sns
import matplotlib.pyplot as plt
f,ax=plt.subplots(1,2,figsize=(10,5))
df['Loan_Status'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Loan_Status')
ax[0].set_ylabel('')
sns.countplot('Loan_Status',data=df,ax=ax[1])
ax[1].set_title('Loan_Status')
LD, NLD = df['Loan_Status'].value_counts()
print('Number of people will get the loan (0): ',LD)
print('Number of people willnot get the loan (1): ',NLD)
plt.grid()
plt.show()
df.dtypes

#Feature Engineering
df=pd.concat([df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Loan_Status']],
                      pd.get_dummies(df.Gender, drop_first=True), 
                      pd.get_dummies(df.Married, drop_first=True), 
                      pd.get_dummies(df.Dependents, drop_first=True), 
                      pd.get_dummies(df.Education, drop_first=True), 
                      pd.get_dummies(df.Self_Employed, drop_first=True), 
                      pd.get_dummies(df.Property_Area, drop_first=True),], axis=1)
df.head()
df.shape

# Check the Number of Rows after Removing Duplicates
df.shape
df.drop_duplicates(inplace=True)
df.shape

# Check Columns that has NULLs
df.isnull().sum()
df.dropna(inplace= True)
df.isnull().sum()
df.shape

#Create Dependent and Independent Variables
target_name = 'Loan_Status'
# Separate object for target feature
y = df[target_name]

# Separate Object for Input Features
X = df.drop(target_name, axis=1)
X.shape
X.head()
y.shape
y.head()

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
MMX = scaler.fit_transform(X)

#Split the Dataset to Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(MMX, y, test_size=0.30, random_state=1)
X_train.shape, y_train.shape,X_test.shape, y_test.shape

#Support Vector Machine
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)

# predict the target variable
svc_y_pred=svc.predict(X_test)
svc_train_score = round(svc.score(X_train, y_train) * 100, 2)
print('Random Forest Score on Train data: \n', svc_train_score)
svc_Test_score = round(svc.score(X_test, y_test) * 100, 2)
print('Random Forest Score on test data: \n', svc_Test_score)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,svc_y_pred))

from sklearn.metrics import confusion_matrix
import pylab as pl
cm = confusion_matrix(y_test,svc_y_pred)
pl.matshow(cm)
pl.title('Confusion matrix of Support Vector Machine')
pl.colorbar()
pl.show()

svc_acc_score=round(accuracy_score(y_test,svc_y_pred)*100)
print('Accuracy: \n', svc_acc_score)

print("precision Score  is:", precision_score(y_test,svc_y_pred))

print("recall_Score is:",recall_score(y_test,svc_y_pred)*100)

print('f1_score is :',f1_score(y_test, svc_y_pred))

from sklearn.metrics import classification_report
print('Classification Report: \n', classification_report(y_test,svc_y_pred))

from sklearn.calibration import CalibratedClassifierCV
svm = SVC()
svc = CalibratedClassifierCV(svm)
svc.fit(X_train, y_train)
svc.predict_proba(X_test)
svc_y_pred_prob = svc.predict_proba(X_test)[:,1]
FPR,TPR, thresholds = roc_curve(y_test, svc_y_pred_prob)

# create plot
plt.plot(FPR,TPR, label='ROC curve of svc')
plt.plot([0, 1], [0, 1],label='regr')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVC')
plt.xlim([-0.02, 1])
plt.ylim([0, 1.02])
plt.legend(loc="lower right")
plt.grid()
plt.show()

from sklearn.metrics import roc_auc_score
svc_auc = round(roc_auc_score(y_test,svc_y_pred_prob)*100,2)
print('roc_auc_score :',svc_auc)
