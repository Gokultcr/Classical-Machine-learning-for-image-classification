
# Experimenting with Classical Machine Learning Algorithms for Image Classification

This script explores various classical machine learning algorithms for image classification. The algorithms used in this experiment include Random Forest, K Nearest Neighbors, Decision Tree, Gradient Boost, Support Vector Machine, Gaussian Naive Bayes, and Multi-Layer Perceptron (MLP). 

It's important to note that only training and testing datasets are used in this experiment, and the algorithms are utilized with predefined parameters. Additionally, only training and testing accuracies are displayed.

### Data Preparation

The dataset is imported in pickle format, and the labels and dataset are stored in separate variables. The dataset is then split into training and testing sets.

### Random Forest Algorithm

Random Forest algorithm is used with 40 estimators

```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=40)
classifier.fit(xtrain, ytrain)
```
### K Nearest Neighbors (KNN) Algorithm

KNN algorithm is tested with varying numbers of neighbors (1 to 10)

```python
from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(xtrain, ytrain)
    training_accuracy.append(knn.score(xtrain, ytrain))
    test_accuracy.append(knn.score(xtest, ytest))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')
```

### Decision Tree Algorithm

Decision Tree algorithm is used

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(xtrain, ytrain)
print("Accuracy on training set: {:.3f}".format(tree.score(xtrain, ytrain)))
print("Accuracy on test set: {:.3f}".format(tree.score(xtest, ytest)))
```

### Gradient Boost Algorithm

Gradient Boost algorithm is utilized

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0)
gb.fit(xtrain, ytrain)
print("Accuracy on training set: {:.3f}".format(gb.score(xtrain, ytrain)))
print("Accuracy on test set: {:.3f}".format(gb.score(xtest, ytest)))
```

### Support Vector Machine (SVM) Algorithm

Support Vector Machine algorithm is applied

```python
from sklearn.svm import SVC

svc = SVC()
svc.fit(xtrain, ytrain)
print("Accuracy on training set: {:.2f}".format(svc.score(xtrain, ytrain)))
print("Accuracy on test set: {:.2f}".format(svc.score(xtest, ytest)))
```

### Gaussian Naive Bayes Algorithm

Gaussian Naive Bayes algorithm is used

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(xtrain, ytrain)
y_pred_nb = model.predict(xtest)
print(y_pred_nb)
```

### Multi-Layer Perceptron (MLP) Algorithm

MLP algorithm is applied

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=42)
mlp.fit(xtrain, ytrain)
print("Accuracy on training set: {:.2f}".format(mlp.score(xtrain, ytrain)))
print("Accuracy on test set: {:.2f}".format(mlp.score(xtest, ytest)))
```

## Conclusion
This script provides an overview of various classical machine learning algorithms for image classification and their performance on the given dataset.

