import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")

# Convert categorical columns to numeric using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed!")
print("Accuracy:", accuracy)

import pickle

# Save the trained model
with open("heart_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as heart_model.pkl")
