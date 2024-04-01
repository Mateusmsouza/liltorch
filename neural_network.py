import math
import random


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_output = random.uniform(-1, 1)

        print(f"input {self.weights_input_hidden}")
        print(f"ouput {self.weights_hidden_output}")
        
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def forward(self, input_data):
        # Compute hidden layer activations
        hidden_activations = []
        for i in range(self.hidden_size):
            activation = self.bias_hidden[i]
            for j in range(self.input_size):
                activation += input_data[j] * self.weights_input_hidden[j][i]
            hidden_activations.append(self.sigmoid(activation))
        
        # Compute output layer activation
        output_activation = self.bias_output
        for i in range(self.hidden_size):
            output_activation += hidden_activations[i] * self.weights_hidden_output[i]
        
        return self.sigmoid(output_activation)
    
    def train(self, input_data, target, learning_rate):
        # Forward pass
        hidden_activations = []
        for i in range(self.hidden_size):
            activation = self.bias_hidden[i]
            for j in range(self.input_size):
                activation += input_data[j] * self.weights_input_hidden[j][i]
            hidden_activations.append(self.sigmoid(activation))
        
        output_activation = self.bias_output
        for i in range(self.hidden_size):
            output_activation += hidden_activations[i] * self.weights_hidden_output[i]
        predicted = self.sigmoid(output_activation)
        
        # Backpropagation
        output_error = target - predicted
        output_delta = output_error * predicted * (1 - predicted)
        
        hidden_errors = [0] * self.hidden_size
        for i in range(self.hidden_size):
            hidden_errors[i] = output_delta * self.weights_hidden_output[i] * hidden_activations[i] * (1 - hidden_activations[i])
        
        # Update weights and biases
        for i in range(self.hidden_size):
            self.weights_hidden_output[i] += learning_rate * output_delta * hidden_activations[i]
            for j in range(self.input_size):
                self.weights_input_hidden[j][i] += learning_rate * hidden_errors[i] * input_data[j]
            self.bias_hidden[i] += learning_rate * hidden_errors[i]
        
        self.bias_output += learning_rate * output_delta

# Example usage:
input_size = 1
hidden_size = 3
output_size = 1

# Create neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Training data
training_data = [(2,), (4,), (6,), (8,)]
targets = [4, 8, 12, 16]

# Train the network
for i in range(1000):
    for j in range(len(training_data)):
        nn.train(training_data[j], targets[j], 0.1)

# Test the trained network
test_data = [[10]]
for data in test_data:
    prediction = nn.forward(data)
    print("Input:", data, "Prediction:", prediction)

