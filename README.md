# Polynomial Regression Implementation

A Python implementation of polynomial regression using gradient descent optimization.

## Features

- Custom polynomial feature generation
- Feature normalization
- Gradient descent optimization
- Model evaluation metrics (MAE, MSE, RMSE)
- Visualization tools for residuals and learning curves

## Usage

```python
from index import PolynomialRegression, train_test_split

# Load and prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train model
model = PolynomialRegression(degree=2, learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## Example

The repository includes example code using the Advertising dataset to predict sales based on advertising spending in different media channels.