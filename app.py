import streamlit as st
from sklearn.datasets import load_iris, load_digits
#loading datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load datasets
iris = load_iris()
digits = load_digits()

# Function to train/evaluate classifiers
def evaluate_classifier(X_train, y_train, X_test, y_test, classifier_name):
    if classifier_name == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000)
    elif classifier_name == "Naive Bayes":
        classifier = GaussianNB()
    elif classifier_name == "MLP Classifier":
        classifier = MLPClassifier(max_iter=1000)
    else:
        st.error("Invalid classifier selection.")
        return None

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return classifier, accuracy

# Streamlit UI
st.title("Machine Learning Classifier App")

# Sidebar for dataset and model selection
st.sidebar.title("Settings")
dataset_choice = st.sidebar.radio("Choose a dataset:", ["IRIS", "Digits"])
classifier_choice = st.sidebar.selectbox("Choose a classifier:", ["Logistic Regression", "Naive Bayes", "MLP Classifier"])

# Split data into features and target outside the if blocks
X_iris, y_iris = iris.data, iris.target
X_digits, y_digits = digits.data, digits.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)

# Load selected dataset
if dataset_choice == "IRIS":
    X_train, X_test, y_train, y_test = X_train_iris, X_test_iris, y_train_iris, y_test_iris
elif dataset_choice == "Digits":
    X_train, X_test, y_train, y_test = X_train_digits, X_test_digits, y_train_digits, y_test_digits

# Create input fields for feature values
st.subheader("Enter Feature Values:")
input_values = []
for i in range(X_train.shape[1]):
    value = st.number_input(f"Feature {i+1}", step=0.1)
    input_values.append(value)

input_array = [input_values]

if st.button("Predict"):
    classifier, accuracy = evaluate_classifier(X_train, y_train, X_test, y_test, classifier_choice)

    if classifier is not None:
        prediction = classifier.predict(input_array)

        st.subheader("Prediction Results:")
        st.write(f"Selected Classifier: {classifier_choice}")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Predicted Class: {prediction}")
