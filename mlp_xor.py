import numpy as np

#Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (for backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR training data
X = np.array([
    [0,0], #output 0
    [0,1], # output 1
    [1,0], # output 1
    [1,1] # output 0
])

# XOR target outputs
y = np.array([[0], [1], [1], [0]])

#Initialize weights randomly
np.random.seed(2526)
input_size = 2
hidden_size = 2
output_size = 1

# Weights and biases
W1 = np.random.uniform(-1, 1, (input_size, hidden_size)) # weights betweeen Input and hidden layer
b1 = np.zeros((1, hidden_size)) # Bias for hidden layer
W2 = np.random.uniform(-1, 1, (hidden_size, output_size)) #Weights between hidden and output layer
b2 = np.zeros((1, output_size))

#learning rate
lr = 0.5
epochs = 10000 # trianing iterations

# Training the MLP
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # Compute error
    error = y - final_output

    # BackPropagation
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_output)

    #Update Weights and Biases
    W2 += np.dot(hidden_output.T, d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr

    W1 += np.dot(X.T, d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    # Print error every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

# Testing the trained MLP
print("\nFinal Predictions:")
for i in range(len(X)):
    hidden_input = np.dot(X[i], W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    print(f"Input: {X[i]} -> Prediction: {final_output[0][0]:.4f}")