# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8. Evaluate Model: Calculate and display accuracy with metrics.accuracy_score. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: D R NETHRANJAN CHOWDARY
RegisterNumber:  212225100031
*/
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics

file = 'spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

data = pd.read_csv('spam.csv', encoding='Windows-1252')
x = data["v2"].values
y = data["v1"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
```

## Output:
![image](https://github.com/user-attachments/assets/ba42ca2d-85a5-4795-b40d-63df37236a81)


![image](https://github.com/user-attachments/assets/22edcf48-dd25-4513-867d-6c7430da985d)


![image](https://github.com/user-attachments/assets/618324db-35cd-4c45-8469-466a22c5cf86)


![image](https://github.com/user-attachments/assets/81bc5619-eb01-41f1-9218-e95a64875d42)


![image](https://github.com/user-attachments/assets/6836f245-d141-4e8b-8bfa-e2d8eac18a76)


![image](https://github.com/user-attachments/assets/3da73419-4d11-45bf-b361-b1fd344e732e)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
