"""
Simple ML flow: load a public dataset, train a classifier, report results.

Dataset: sklearn's built-in digits (handwritten 0-9, 8x8 images, 1797 samples).
Model:   Logistic Regression.

Run:  python simple_ml.py
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the data
digits = load_digits()
X, y = digits.data, digits.target
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes\n")

# 2. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples\n")

# 3. Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model trained.\n")

# 5. Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Test accuracy: {acc:.4f}\n")
print("Per-class results:")
print(classification_report(y_test, preds))
print("Confusion matrix:")
print(confusion_matrix(y_test, preds))
