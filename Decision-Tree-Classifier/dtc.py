#step:1 :- import libraries

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#step:2 :-import dataset
iris =load_iris()

label_names = iris['target_names']
labels = iris['target']
feature_names = iris['feature_names']
features = iris['data']

print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])

#step:3 :- divide dataset into training dataset and testing dataset
# Here I take 20% original dataset as test dataset
train,test,train_labels,test_labels = train_test_split(features,labels,
                                                        test_size = 0.20,
                                                        random_state = 42)

dtc = DecisionTreeClassifier()
model = dtc.fit(train,train_labels)

preds = dtc.predict(test)
print("Prediction")
print(preds)

# Evaluating models accuracy
print("accuracy")
print(accuracy_score(test_labels,preds))
