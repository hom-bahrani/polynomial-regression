import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

class PolynomialRegression:
    """
    A polynomial regression implementation using gradient descent.
    """
    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000, normalize=True):
        """
        Initialize the polynomial regression model.
        
        Parameters:
        degree (int): Degree of the polynomial
        learning_rate (float): The step size for gradient descent
        num_iterations (int): Number of iterations for gradient descent
        normalize (bool): Whether to normalize the features
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.normalize = normalize
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.means = None
        self.stds = None
        
    def _generate_polynomial_features(self, X):
        """
        Generate polynomial features up to the specified degree.
        
        Parameters:
        X (numpy.ndarray): Features array of shape (m, n)
        
        Returns:
        numpy.ndarray: Polynomial features of shape (m, n_poly)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        num_samples, num_features = X.shape
        
        # List to store all the polynomial features
        poly_features = []
        
        # For each degree level
        for d in range(1, self.degree + 1):
            # Get all combinations of features (including repeats) for this degree
            combs = list(combinations_with_replacement(range(num_features), d))
            
            # For each combination, create a new feature
            for comb in combs:
                # Start with all ones
                new_feature = np.ones(num_samples)
                
                # Multiply by the appropriate features
                for idx in comb:
                    new_feature *= X[:, idx]
                    
                poly_features.append(new_feature)
        
        # Convert list of features to numpy array
        return np.column_stack(poly_features)
    
    def _normalize_features(self, X):
        """
        Normalize features by subtracting the mean and dividing by standard deviation.
        
        Parameters:
        X (numpy.ndarray): Features
        
        Returns:
        numpy.ndarray: Normalized features
        numpy.ndarray: Means
        numpy.ndarray: Standard deviations
        """
        num_samples, num_features = X.shape
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        
        # Replace zero standard deviations with 1 to avoid division by zero
        stds[stds == 0] = 1
        
        # Normalize
        X_norm = (X - means) / stds
        
        return X_norm, means, stds
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Parameters:
        X (numpy.ndarray): Training features
        y (numpy.ndarray): Target values
        """
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        num_samples, num_poly_features = X_poly.shape
        
        # Normalize features if specified
        if self.normalize:
            X_poly_norm, self.means, self.stds = self._normalize_features(X_poly)
        else:
            X_poly_norm = X_poly
            self.means = np.zeros(num_poly_features)
            self.stds = np.ones(num_poly_features)
        
        # Initialize weights and bias
        self.weights = np.zeros(num_poly_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Forward pass: compute predictions
            y_pred = self._predict_normalized(X_poly_norm)
            
            # Compute gradients
            dw = (1/num_samples) * np.dot(X_poly_norm.T, (y_pred - y))
            db = (1/num_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate cost for tracking progress
            cost = self._compute_cost(X_poly_norm, y)
            self.cost_history.append(cost)
            
            # Print cost every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
    
    def _predict_normalized(self, X_norm):
        """
        Make predictions using normalized features.
        
        Parameters:
        X_norm (numpy.ndarray): Normalized features
        
        Returns:
        numpy.ndarray: Predictions
        """
        return np.dot(X_norm, self.weights) + self.bias
    
    def predict(self, X):
        """
        Public method to make predictions.
        
        Parameters:
        X (numpy.ndarray): Features
        
        Returns:
        numpy.ndarray: Predictions
        """
        # Generate polynomial features for the input
        X_poly = self._generate_polynomial_features(X)
        
        if self.normalize:
            # Normalize the features using the stored means and stds
            X_poly_norm = (X_poly - self.means) / self.stds
            return self._predict_normalized(X_poly_norm)
        else:
            return self._predict_normalized(X_poly)
    
    def _compute_cost(self, X, y):
        """
        Compute the mean squared error cost.
        
        Parameters:
        X (numpy.ndarray): Features (normalized if self.normalize=True)
        y (numpy.ndarray): True values
        
        Returns:
        float: Mean squared error
        """
        num_samples = X.shape[0]
        y_pred = self._predict_normalized(X)
        cost = (1/(2*num_samples)) * np.sum((y_pred - y)**2)
        return cost
    
    def get_original_coefficients(self):
        """
        Convert the coefficients back to the original scale if normalization was used.
        
        Returns:
        tuple: (weights, bias) in the original scale
        """
        if self.normalize:
            # Convert weights back to original scale
            weights_original = self.weights / self.stds
            
            # Adjust bias to account for the mean shift
            bias_original = self.bias - np.sum(self.weights * self.means / self.stds)
            
            return weights_original, bias_original
        else:
            return self.weights, self.bias


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the mean absolute error.
    
    Parameters:
    y_true (numpy.ndarray): True values
    y_pred (numpy.ndarray): Predicted values
    
    Returns:
    float: Mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error.
    
    Parameters:
    y_true (numpy.ndarray): True values
    y_pred (numpy.ndarray): Predicted values
    
    Returns:
    float: Mean squared error
    """
    return np.mean((y_true - y_pred)**2)


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate the root mean squared error.
    
    Parameters:
    y_true (numpy.ndarray): True values
    y_pred (numpy.ndarray): Predicted values
    
    Returns:
    float: Root mean squared error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_residuals(y_true, y_pred):
    """
    Plot the residuals of the model.
    
    Parameters:
    y_true (numpy.ndarray): True values
    y_pred (numpy.ndarray): Predicted values
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=20)
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.show()


def plot_learning_curve(model):
    """
    Plot the learning curve (cost vs iterations) of the model.
    
    Parameters:
    model (PolynomialRegression): Trained polynomial regression model
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(model.num_iterations), model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Learning Curve')
    plt.show()


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the data into training and testing sets.
    
    Parameters:
    X (numpy.ndarray): Features
    y (numpy.ndarray): Target values
    test_size (float): Proportion of the dataset to include in the test split
    random_state (int): Seed for random number generator
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split_idx = int(num_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("Advertising.csv")
    
    # Prepare features and target
    X = df.drop('sales', axis=1).values
    y = df['sales'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    # Train the model with degree=2
    model = PolynomialRegression(degree=2, learning_rate=0.01, num_iterations=1000, normalize=True)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # Get original coefficients
    weights, bias = model.get_original_coefficients()
    
    # Print model coefficients
    print("\nModel Coefficients:")
    feature_names = df.drop('sales', axis=1).columns
    
    # Generate polynomial feature names
    poly_feature_names = []
    for d in range(1, model.degree + 1):
        combs = list(combinations_with_replacement(range(len(feature_names)), d))
        for comb in combs:
            name = " * ".join([feature_names[idx] for idx in comb])
            poly_feature_names.append(name)
    
    for i, name in enumerate(poly_feature_names):
        print(f"{name}: {weights[i]:.6f}")
    print(f"Bias: {bias:.6f}")
    
    # Plot residuals
    plot_residuals(y_test, y_pred)
    
    # Plot learning curve
    plot_learning_curve(model)