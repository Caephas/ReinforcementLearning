import numpy as np
import matplotlib.pyplot as plt

"""
    1.	Generate a linearly separable dataset (e.g., two clusters of points).
	2.	Train a perceptron to classify the data using the perceptron learning rule.
	3.	Visualize performance using a decision boundary plot.
	4.	Keep it simple following the KISS (Keep It Simple, Stupid) principle.
"""
# Step 1: Generate a linearly separable dataset
np.random.seed(0)
num_samples = 100

# Generate two clusters (Class 1 and Class 2)
X1 = np.random.randn(num_samples, 2) + np.array([2, 2])  # Cluster around (2,2)
X2 = np.random.randn(num_samples, 2) + np.array([-2, -2])  # Cluster around (-2,-2)

# Combine the data
X = np.vstack((X1, X2))
y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))  # Class labels: 1 and -1

# Step 2: Implement a simple Perceptron
class Perceptron:
    def __init__(self, input_size:int, learning_rate:float=0.1, epochs:int=10):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else -1  # Step function

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                summation = np.dot(X[i], self.weights) + self.bias
                prediction = self.activation(summation)
                error = y[i] - prediction
                
                # Perceptron Learning Rule: Update weights and bias
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

    def predict(self, X):
        return np.array([self.activation(np.dot(x, self.weights) + self.bias) for x in X])

# Train the perceptron
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
perceptron.train(X, y)

# Step 3: Visualizing the Decision Boundary
def plot_decision_boundary(X, y, perceptron):
    plt.figure(figsize=(8, 6))

    # Scatter plot of points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Compute predictions for the grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid_points).reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")
    plt.title("Perceptron Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Step 4: Plot the decision boundary
plot_decision_boundary(X, y, perceptron)