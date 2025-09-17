import joblib, os
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.random.rand(200, 5)
y = np.random.randint(0, 2, 200)

model = LogisticRegression(max_iter=200)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Saved model to models/model.pkl")
