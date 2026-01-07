import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.model.logistic_regression import LogisticRegression
from src.utils.standard_scaler import StandardScaler


# Load the dataset
data_path = Path("input/breast_cancer.parquet")
df = pd.read_parquet(data_path)

X = df.drop(columns=['target']).values
y = df['target'].values

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features (fit on training data only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Cross-validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model = LogisticRegression(solver='gradient_descent', max_iter=1000)
    model.fit(X_fold_train, y_fold_train)

    fold_score = model.score(X_fold_val, y_fold_val)
    cv_scores.append(fold_score)
    print(f"Fold {fold} accuracy: {fold_score:.4f}")

print(f"\nMean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Final evaluation on test set
model_path = Path("output/model")
model_path.mkdir(parents=True, exist_ok=True)

# Train and save gradient descent model
gd_model = LogisticRegression(solver='gradient_descent', max_iter=1000)
gd_model.fit(X_train, y_train)
gd_score = gd_model.score(X_test, y_test)
print(f"\nGradient Descent:")
print(f"  Test accuracy: {gd_score:.4f}")
print(f"  Iterations to converge: {gd_model.n_iter_}")
joblib.dump(gd_model, model_path / "logistic_regression_gradient_descent.joblib")
print(f"  Model saved to {model_path / 'logistic_regression_gradient_descent.joblib'}")

# Train and save Newton's method model
newton_model = LogisticRegression(solver='newton', max_iter=1000)
newton_model.fit(X_train, y_train)
newton_score = newton_model.score(X_test, y_test)
print(f"\nNewton's Method:")
print(f"  Test accuracy: {newton_score:.4f}")
print(f"  Iterations to converge: {newton_model.n_iter_}")
joblib.dump(newton_model, model_path / "logistic_regression_newton.joblib")
print(f"  Model saved to {model_path / 'logistic_regression_newton.joblib'}")
