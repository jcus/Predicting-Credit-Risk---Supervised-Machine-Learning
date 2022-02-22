# Predicting Credit Risk - Supervised Machine Learning 

Building a machine learning (ML) model to predict whether a loan from LendingClub will become high risk or not. 
To make predication on those given set of samples, Supervised Machine Learning algorithm searches of patterns with the value labels assigned to data points. Through some ML algorithms for supervised learning include SVM for classification problems, Linear Regression for regression problems, and Random Forest for regression, and Classification problems. Since those given data contains annotations with output classes form the cardinal out classes. We hereby apply supervised ML of those algorithms to test and score the risk level from the LendingClub dataset for the loan.


## Background :computer:

A peer-to-peer lending services company-- LendingClub allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

Using this data to create machine learning models to classify the risk level of given loans. Specifically, comparing the Logistic Regression model and Random Forest Classifier.


## Process - Feature Engineering :computer:

### Retrieve, Identify the data
Data source: in folder `Resources/Generator` , there is a notebook [GenerateData.ipynb](/Resources/Generator/GenerateData.ipynb) download data from LendingClub and output two CSVs. In original dataset, 2.2% of loans are categorized as high risk.
Prepare data: To get a truly accurate model, these dataset have been Undersampled to give an even number of high risk and low risk loans. Undersampling, Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are techniques are also chould been choosed from. Make sure the data is clean, secured, and governed. 

* `2019loans.csv`
* `2020Q1loans.csv`

Using an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).

### Nominal to numerical- Convert categorical data
Create a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. 
Similarly, create a testing set from the 2020 loans, also using `pd.get_dummies()`. 
Note: There are categories in the 2019 loans that do not exist in the testing set. If you fit a model to the training set and try to score it on the testing set as is, you will get an error. Use code to fill in the missing categories in the testing set. 


## Ensemble Methods :computer:

Training: Apply machine learning algorithms to create general Linear Models. In this project, we use sklearn models like logistic regression, random forests to classify the training data. Also, applying offen used bagging methods for ensemble, like RandomForest. 

### Consider the models- test and score
Creating and comparing two models on this data: a logistic regression, and a random forests classifier. 
Evaluate the models to find the best performing algorithm. Make a prediction of model with better performing. Which method we use, depends on the model type and the feature values.

Before we create, fit, and score the models, there is a prediction markdown cells in Jupyter Notebook as to which model we think will perform better, and provide justification.

### Fit a LogisticRegression model and RandomForestClassifier model
Create a LogisticRegression model, fit it to the data, and print the model's score. Do the same for a RandomForestClassifier. You may choose any starting hyperparameters you like. Which model performed better? How does that compare to your prediction? Write down your results and thoughts.


### Revisit the Preprocessing: Scale the data 
The data going into these models was never scaled, an important step in preprocessing. Use `StandardScaler` algorithm to scale the training and testing sets. Before re-fitting the LogisticRegression and RandomForestClassifier models on the scaled data, there is another prediction and provide justification about how we think scaling will affect the accuracy of the models. 

Fit and score the LogisticRegression and RandomForestClassifier models on the scaled data. How do the model scores compare to each other, and to the previous results on unscaled data? How does this compare to your prediction? Write down your results and thoughts.

### Further steps- Deploy and assess predictiona

Deploy: Machine learning algorithms create models that can be deployed to both cloud and on-premises applications as needed.
Predict: After deployment, we could start making predictions based on new, incoming data.
Assess predictions: Assess the validity of your predictions. The information we gather from analyzing the validity of
predictions is then fed back into the machine learning cycle to help improve accuracy.


## References :computer:

LendingClub (2019-2020) _Loan Stats_. Retrieved from: [https://resources.lendingclub.com/](https://resources.lendingclub.com/)

- - -

© 2021 Trilogy Education Services, a 2U, Inc. brand. All Rights Reserved.
