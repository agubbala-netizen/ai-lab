# Binary Classification using Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Sample Dataset
# Features: [Hours_Studied, Attendance]
# Label: 0 = Fail, 1 = Pass
# -----------------------------

X = [
    [2, 60],
    [4, 70],
    [6, 80],
    [8, 90],
    [1, 50],
    [3, 65],
    [7, 85],
    [5, 75]
]

y = [0, 0, 1, 1, 0, 0, 1, 1]

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# Create Decision Tree Classifier
# -----------------------------
model = DecisionTreeClassifier(criterion="gini")

# Train the model
model.fit(X_train, y_train)

# -----------------------------
# Test the model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# -----------------------------
# User Input for Prediction
# -----------------------------
print("\n--- Predict Student Result ---")
hours = int(input("Enter hours studied: "))
attendance = int(input("Enter attendance percentage: "))

prediction = model.predict([[hours, attendance]])

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
