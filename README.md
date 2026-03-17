# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries and load the spam email dataset
2. Convert the email text into numerical features using TF-IDF Vectorizer.
3. Split the dataset into training data and testing data, then train the Support Vector Machine (SVM) classifier.
4. Predict whether the email is Spam or Ham and evaluate the model accuracy.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: SHAJIVE KUMAR J
RegisterNumber: 212225230258 
```
~~~
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv(f"C:/Users/acer/Downloads/spam.csv", encoding="latin-1")

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n")
print(cm)
~~~

## Output:

<img width="786" height="444" alt="Screenshot 2026-03-17 082328" src="https://github.com/user-attachments/assets/570868cb-8f79-48b0-8f93-cbaf14c05061" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
