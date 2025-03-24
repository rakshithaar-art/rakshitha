
Handwritten Digit Prediction using Linear Model

Build a machine learning model to classify a handwritten digit using a linear model.

1... # Imports required packages
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

Retrieving Data


2... # Fetches the data set and caches it
# NOTE: This step may take several seconds to complete as it downloads 
# the data from web and then caches it locally
mnist = fetch_openml("mnist_784", as_frame=False)
3... # [OPTIONAL] Checks the available bunch objects
print(mnist.keys())

Exploratory Data Analysis (EDA)

4... # Finds the shape of the data
print(mnist.data.shape)
5... # Similarly, finds the shape of the target
print(mnist.target.shape)

6... # Let's view target of first few observations
print(mnist.target[:20])

7... # [OPTIONAL] Let's plot first 100 handwritten digits in a 10x10 subplots in a figure
for idx, image_data in enumerate(mnist.data[:100]):
    plt.subplot(10, 10, idx + 1)
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap = "binary")
    plt.axis("off")
plt.show()

Modeling

8... # Converts target type from 'char' to 'integer'
mnist.target = mnist.target.astype(int)

9... # Split the data set into train and test data set.
# NOTE: Data set is already shuffled, and no further shuffling is being performed here.
# As recommended for this dataset, train and test set ratio maintained is 60000:10000.
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42, stratify=mnist.target)
10... # Checks values of the few of the labels
y_train

11...# Instantiate scaler object to scale train set (with mean 0 and standard deviation 1)
std_scaler = StandardScaler()
# Scales the feature values
X_train_scaled = std_scaler.fit_transform(X_train)

12... # Transforms the test data using learned scaler
X_test_scaled = std_scaler.transform(X_test)

13... # Initialize the classifier
sgd_clf = SGDClassifier(n_jobs=-1, random_state=42)
# Fits the model on train set
# NOTE: This step may take few minutes to complete
sgd_clf.fit(X_train_scaled, y_train)

14... # Performs predictions on the already scaled test set
predictions_test = sgd_clf.predict(X_test_scaled)
print(predictions_test)

15... # Takes just one prediction and compares with the actual label
print("Prediction:", predictions_test[0])
print("Actual Label:", y_test[0])

Decision Function


16... # Predict confidence score over decision function
decision_scores = sgd_clf.decision_function([X_test_scaled[0]])
print(decision_scores)

17... # The class with the highest score gets predicted.
print("Prediction:", sgd_clf.classes_[decision_scores.argmax()])
print("Actual Label:", y_test[0])

18... # Prints prediction performance of the model on test set
print("Prediction Performance (Accuracy) of SGD Classifier: {:.1f}%".format(accuracy_score(y_test, predictions_test) * 100))

Error Analysis

19... # Performs cross-validation predictions
# NOTE: This step may take several minutes to complete
cv_predictions = cross_val_predict(
    sgd_clf, X_train_scaled, y_train, cv=5, n_jobs=-1, verbose=3, method="predict")
    
20... # Plots the confusion matrix
ConfusionMatrixDisplay.from_predictions(y_train, cv_predictions)
plt.title("Confusion Matrix")
plt.show()

21... # Plots the normalized confusion matrix by dividing each value by the total number of images in the
# corresponding (true) class (i.e., divide by the row’s sum).
    
ConfusionMatrixDisplay.from_predictions(y_train, cv_predictions, normalize="true", values_format=".0%")
plt.title("Confusion Matrix [Row-Normalized]")
plt.show()

22... # Renders the confusion matrix to emphasis more on incorrect predictions than the correct predictions
sample_weight = (cv_predictions != y_train) 
ConfusionMatrixDisplay.from_predictions(
    y_train, cv_predictions, sample_weight=sample_weight, normalize="true", values_format=".0%")
plt.title("Confusion Matrix [Error-Normalized by Row]")
plt.show()

