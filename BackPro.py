import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Initialize Weights and Biases
        self.weights_input_hidden = np.random.rand(self.input_nodes, self.hidden_nodes)
        self.weights_hidden_output = np.random.rand(self.hidden_nodes, self.output_nodes)
        self.bias_hidden = np.random.rand(self.hidden_nodes)
        self.bias_output = np.random.rand(self.output_nodes)

    def forward(self, X):
        # Forward Propagation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        # Calculate Error
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update Weights and Biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate

    def train(self, X, y, epochs):
        loss = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            loss.append(np.mean(np.square(y - output)))
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss[-1]}")
        return loss

# Data Preparation (XOR Dataset)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Hyperparameters
input_nodes = 2
hidden_nodes = 8
output_nodes = 1
learning_rate = 0.9
epochs = 2000

# Train Network
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
loss = nn.train(X, y, epochs)

# Plot Loss Curve
plt.plot(loss)
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()