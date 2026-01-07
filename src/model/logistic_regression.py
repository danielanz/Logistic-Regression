import numpy as np


class LogisticRegression:
    """
    Logistic Regression classifier with gradient descent and Newton's method solvers.

    Parameters
    ----------
    solver : str, default='gradient_descent'
        Optimization algorithm to use. Options: 'gradient_descent', 'newton'.
    learning_rate : float, default=0.01
        Learning rate for gradient descent (ignored for Newton's method).
    max_iter : int, default=1000
        Maximum number of iterations for the solver.
    tol : float, default=1e-4
        Tolerance for convergence.
    fit_intercept : bool, default=True
        Whether to add an intercept term.
    """

    def __init__(
        self,
        solver: str = 'gradient_descent',
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True
    ):
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

        # Attributes set during fit
        self.coef_ = None  # Coefficients (weights)
        self.intercept_ = None  # Intercept term
        self.n_iter_ = None  # Number of iterations until convergence
        self.classes_ = None  # Unique class labels
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        result = np.zeros_like(z)
        
        pos_mask = z >= 0
        neg_mask = ~pos_mask
        
        result[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
        result[neg_mask] = np.exp(z[neg_mask]) / (1 + np.exp(z[neg_mask]))
        
        return result

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to feature matrix."""
        return np.column_stack([np.ones(X.shape[0]), X])

    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optimize weights using gradient descent."""
        sigma = self._sigmoid(X @ self.coef_)
        m = X.shape[0]
        self.coef_ = self.coef_ + self.learning_rate * (1/m) * X.T @ (y - sigma)

    def _newtons_method(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optimize weights using Newton's method."""
        sigma = self._sigmoid(X @ self.coef_)
        S = np.diag(sigma * (1 - sigma))
        H = X.T @ S @ X + 1e-4 * np.eye(X.shape[1]) # Adding a small value (L2 Regularization) to ensure invertibility
        gradient = X.T @ (y - sigma)
        self.coef_ = self.coef_ + np.linalg.solve(H, gradient)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit the logistic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LogisticRegression
            Fitted estimator.
        """

        # Store unique classes
        self.classes_ = np.unique(y)
        
        # Add intercept
        if self.fit_intercept:
            X = self._add_intercept(X)

        # Initialize weights to zeros
        self.coef_ = np.zeros(X.shape[1])

        # Choose solver
        if self.solver == 'gradient_descent':
            update_fn = self._gradient_descent
        elif self.solver == 'newton':
            update_fn = self._newtons_method
        else:
            raise ValueError(f'Invalid Solver, Valid Solvers: gradient_descent, newton')

        # Optimization loop
        for i in range(self.max_iter):
            coef_old = self.coef_.copy()
            update_fn(X, y)
        
            if np.linalg.norm(self.coef_ - coef_old) < self.tol:
                self.n_iter_ = i + 1
                break 
        
        # If it doesn't converge yet then we have hit the maximum number of iterations
        else:
            self.n_iter_ = self.max_iter

        # Separate intercept from coefficients
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
        
        return self


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            Probability of each class (column 0: class 0, column 1: class 1).
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
            # Use full coefficient vector (intercept + weights)
            coef = np.concatenate([[self.intercept_], self.coef_])
        else:
            coef = self.coef_
        
        prob_class_1 = self._sigmoid(X @ coef)
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        threshold : float, default=0.5
            Decision threshold for classification.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the accuracy score on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Accuracy score.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
