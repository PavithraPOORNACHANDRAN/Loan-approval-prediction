
## Loan Approval Prediction using Machine Learning Models

Now a day’s bankers and financial industries are giving a good service. The efficient of the industry is fully depends on the ability to determine the credit risk. Before giving the loan to borrowers, the bank needs to check whether the borrower is eligible to take a loan or not. Identifying the customer status is a challenging task for bankers and financial industry. 

So, this problem is tackled by using machine learning models. We did experiment on the previous records of the customer to whom get the loan and based on the records and experience the model was trained and validated by using machine learning of classification algorithms. We prepared our data using Jupyter Notebook and use various models to predict the target variable.

## Installation

The Code is written in Python 3.7. If you don't have Python installed you can find it here. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. 

We were used Python for this course along with the below-listed libraries.

- Numpy
- Pandas
- Matplotlib
- scikit-learn
## Dataset

The data set include 13 attributes such as Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. The data sets contain 615 records which is unfiltered data. The data can be accessed from Kaggle.com 

If you want to try with other dataset, you can get from the bellow link:

https://www.kaggle.com/search?q=loan+prediction+dataset

  
## Methodologies

Our proposed Methodologies are following in this way:
- Data Acquisition
- Exploratory Data Analysis (EDA)
- Data Pre-Processing
- Feature Engineering
- Build the Model
- Evaluate the Model

## Model Building
For getting comparison, We were build the following models for the same dataset:
- Logistic Regression
- Random Forest
- Suppoer Vector Machine

We evaluate the models performance by using the metrics called Accuracy, Precision, Recall, F1 Score, Area Under the Curve (AUC), ROC Curve which plots the true positive rates against false positive rates.

According to these metrics, Support Vector Machine model has the high accuracy and the Random Forest model has the lowest accuracy. Support Vector Machine has given 85% accurate classification result. 


## License

[MIT](https://choosealicense.com/licenses/mit/)

  