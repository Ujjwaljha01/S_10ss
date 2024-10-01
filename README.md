Pratical- 8 : 
Aim:- Implement the K-NN algorithm for classification or regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

plt.style.use('ggplot')

# Load the data
df = pd.read_csv("D:/Ai practical/diabetes.csv")
print(df.head(),"\n")
print(df.shape,"\n")
print(df.dtypes,"\n")

# Prepare the data
x = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# KNN model training and evaluation
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    train_accuracy[i] = knn.score(x_train, y_train)
    test_accuracy[i] = knn.score(x_test, y_test)

# Plotting the accuracy for different values of k
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# Predict probabilities for ROC AUC score calculation
knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(x_train, y_train)
y_pred_prob = knn.predict_proba(x_test)[:, 1]

# Grid Search for optimal hyperparameters
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
print("\n\n",knn_cv.fit(x, y))

print("\n\n",knn_cv.best_score_)
print("\n\n",knn_cv.best_params_)
output:-







