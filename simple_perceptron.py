import numpy as np

class Perceptron:
    def __init__(self, input_size:int, learning_rate:float=0.1,epochs:int=10):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def summation_function(self,inputs):
        """Computes the weighted sum of the input plus bias"""
        return np.dot(inputs, self.weights) + self.bias
    
    def activation_function(self, summation):
        """Step function: Returns 1 if sum >=0 else returns 0"""
        return 1 if summation >= 0 else 0
    
    def train(self, X, y):
        """train the Perceptron suing the perceptron learning rule"""
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            for i in range(len(X)):
                inputs = X[i] #Input features
                actual_label = y[i] # the corresponding label

                summation = self.summation_function(inputs) # Computing the sum
                prediction = self.activation_function(summation) # Applying the activation

                error = actual_label - prediction

                # Update the weights and bias using the Perceptron learning rule
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

                print(f"Input: {inputs}, Prediction: {prediction}, Actual: {actual_label}, Error: {error}")
                print(f"Updated Weights: {self.weights}, Updated Bias: {self.bias}")
    
    def predict(self, X):
        """Predict the output for the new inputs"""
        predictions = []
        for inputs in X:
            summation = self.summation_function(inputs)
            predictions.append(self.activation_function(summation))

        return predictions
    

# Weights: Adjusted during training to improve predictions
#  Bias: Helps shift the decision boundary
# Summation Function: Computes weighted sum of inputs
#  Activation Function: Uses a step function for binary classification
#  Output: Final predicted class (0 or 1)

# Example Dataset (AND logic gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
y = np.array([0, 0, 0, 1])  # Expected output (AND logic gate)

# Initialize and train the perceptron
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
perceptron.train(X, y)

# Make predictions on new data
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = perceptron.predict(test_data)

print("\nFinal Predictions:")
for i in range(len(test_data)):
    print(f"Input: {test_data[i]}, Predicted Output: {predictions[i]}")