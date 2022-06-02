import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
plt.style.use('ggplot')

accuracyFile = open('accuracy_result.txt','w')
accuracyFile.close()

# Load the dataset
df = pd.read_csv('wpbc.csv')
#df.columns = ['ID number', 'Class', 'radius(mean)', 'texture(mean)', 'perimeter(mean)', 'area(mean)', 'smoothness(mean)', 'compactness(mean)', 'cancavity(mean)', 'concave points(mean)','symmetry(mean)','fractal dimension(mean)',    \
#'radius(standard error)', 'texture(standard error)', 'perimeter(standard error)', 'area(standard error)', 'smoothness(standard error)', 'compactness(standard error)', 'cancavity(standard error)', 'concave points(standard error)','symmetry(standard error)','fractal dimension(standard error)', \
#'radius(worst)', 'texture(worst)', 'perimeter(worst)', 'area(worst)', 'smoothness(worst)', 'compactness(worst)', 'cancavity(worst)', 'concave points(worst)','symmetry(worst)','fractal dimension(worst)']

df.columns = ['ID number', 'Class', 'time','radius(mean)', 'texture(mean)', 'perimeter(mean)', 'area(mean)', 'smoothness(mean)', 'compactness(mean)', 'cancavity(mean)', 'concave points(mean)','symmetry(mean)','fractal dimension(mean)',    \
'radius(standard error)', 'texture(standard error)', 'perimeter(standard error)', 'area(standard error)', 'smoothness(standard error)', 'compactness(standard error)', 'cancavity(standard error)', 'concave points(standard error)','symmetry(standard error)','fractal dimension(standard error)', \
'radius(worst)', 'texture(worst)', 'perimeter(worst)', 'area(worst)', 'smoothness(worst)', 'compactness(worst)', 'cancavity(worst)', 'concave points(worst)','symmetry(worst)','fractal dimension(worst)','tumor size','lymph node status']
df = df.drop(columns= 'ID number')
df = df.replace('?', np.NaN)
df = df.dropna()

# Let's create numpy arrays for features and target
#X = df.drop('Class',axis=1).values
features=['radius(mean)', 'texture(mean)','time']
X = df[features].values
y = df['Class'].values

# Change the values of y into 0 and 1 only
for c in range(len(y)): 
    if y[c] == 'R':
        y[c] = 0
    elif y[c] == 'N':
        y[c] = 1

y = y.astype('int')

# Create a test set of size of about 40% of the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

# Setup arrays to store training and test accuracies
neighbors = np.arange(1, 30)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the model
    knn.fit(X_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)

print(neighbors)
print(train_accuracy)
print(test_accuracy)

# Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model
knn.fit(X_train,y_train)

#knn.score(X_test,y_test)

# Let us get the predictions using the classifier we had fit above
#y_pred = knn.predict(X_test)

model = RandomForestClassifier(n_estimators = 100, random_state = 0, n_jobs = -1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix :')
print(cm)
print()

TP = cm[0][0]
FN = cm[0][1]
FP = cm[1][0]
TN = cm[1][1]

accuracy = (TP+TN)/(TP+TN+FP+FN)
# Measurement
print('Sensitivity :', TP/(TP+FN))
print('Specificity :', TN/(TN+FP))
print('Precision \t:', TP/(TP+FP))
print('Recall \t\t:', TP/(TP+FN))
print('Accuracy \t:', accuracy)
print(model.score(X_test,y_test))
print(knn.score(X_test,y_test))

with open("accuracy_result.txt","a") as f:
    f.write(f"{accuracy}\n")

    