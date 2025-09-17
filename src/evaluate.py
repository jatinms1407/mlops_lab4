import joblib, json, numpy as np
from sklearn.metrics import accuracy_score

# Load trained model
model = joblib.load("models/model.pkl")

# Fake test dataset (just for lab demo)
X_test = np.random.rand(50, 5)
y_test = np.random.randint(0, 2, 50)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
metrics = {"accuracy": float(acc)}

# Save metrics to JSON
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Wrote metrics.json:", metrics)
