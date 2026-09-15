print("Python file is running!")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
data = pd.read_csv("dataset/credit_data.csv")

# Feature engineering from financial history
data["debt_to_income"] = data["debt"] / data["income"]
data["loan_to_income"] = data["loan_amount"] / data["income"]

# Display the dataset
print(data)

# Display information
print("\nDataset Information:")
print(data.info())

# Display first 5 rows
print("\nFirst 5 Rows:")
print(data.head())

# Define features (X) and target (y)
X = data.drop("creditworthy", axis=1)
y = data["creditworthy"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))