import numpy as np


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample x is calculated as:
        z = (x - mean) / std

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std

        # Attributes set during fit
        self.mean_ = None  # Mean of each feature
        self.std_ = None  # Standard deviation of each feature
        self.n_features_ = None  # Number of features

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : StandardScaler
            Fitted scaler.
        """
        self.n_features_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0) if self.with_mean else np.zeros(self.n_features_)
        self.std_ = np.std(X, axis=0) if self.with_std else np.ones(self.n_features_)
        # Avoid division by zero for constant features
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_scaled : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_scaled : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Undo the scaling of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Scaled data.

        Returns
        -------
        X_original : array-like of shape (n_samples, n_features)
            Data in original scale.
        """
        return X * self.std_ + self.mean_
