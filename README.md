# HealthRadar
Scripts for training a model to assess personalized cancer risks.

## Preprocessing
Convert raw data from web to CSV file.
```
raw_to_csv.py
```
Select and modify features based on manual curation and missing data.
```
preprocessing.py
```
Scale numerical data, convert categorical data to binary data.
```
transformation.py
```

## Model fitting
Train model using stochastic gradient descent (logistic regression and linear SVM), logistic regression, and kernel approximation + linear SVM.
```
learning_SGD.py
learning_LR.py
learning_kernel.py
```
