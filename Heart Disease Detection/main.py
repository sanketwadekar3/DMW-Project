
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('heart.csv')
X = dataset[['cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
Y = dataset[['target']]
    
#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

knn_classifier = KNeighborsClassifier(n_neighbors=20)
knn = knn_classifier.fit(X_train, Y_train.values.ravel())
K_pred = knn_classifier.predict(X_test)

rf_classifier = RandomForestClassifier(n_estimators=20, random_state = 0)
rf = rf_classifier.fit(X_train, Y_train.values.ravel())
RF_pred = rf_classifier.predict(X_test)

nb_classifier = GaussianNB()
nb = nb_classifier.fit(X_train, Y_train.values.ravel())
NB_pred = nb_classifier.predict(X_test)

print("---------------------------------------------------------------")
print("\nHeart Disease Detection\n")
print("---------------------------------------------------------------")

print("\nKNN Result\n")
print(confusion_matrix(Y_test, K_pred))
print(classification_report(Y_test, K_pred))
knn_accuracy = accuracy_score(Y_test, K_pred)
print("---------------------------------------------------------------")

print("\nRandom Forest Result\n")
print(confusion_matrix(Y_test, RF_pred))
print(classification_report(Y_test, RF_pred))
rf_accuracy = accuracy_score(Y_test, RF_pred)
print("---------------------------------------------------------------")

print("\nNaive Bayes Result\n")
print(confusion_matrix(Y_test, NB_pred))
print(classification_report(Y_test, NB_pred))
nb_accuracy = accuracy_score(Y_test, NB_pred)
print("---------------------------------------------------------------")

y = [knn_accuracy,rf_accuracy,nb_accuracy]
#y = [f(i) for i in y]	
x = range(3)
width = 1/1.5
plt.bar(x,y,width,color='blue')
plt.xticks(range(3),('KNN','Random Forest','Naive Bayes'))
print(x)
print(y)
plt.legend()
plt.show()
