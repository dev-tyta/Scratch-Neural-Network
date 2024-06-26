#### Libraries
# Standard library
import random
import numpy as np

class NeuralNetwork:
    """
    A class representing a neural network.

    Attributes:
        num_layers (int): The number of layers in the neural network.
        sizes (list): A list containing the number of neurons in each layer.
        biases (list): A list of bias vectors for each layer (except the input layer).
        weights (list): A list of weight matrices for each layer.

    Methods:
        feedforward(a): Perform a feedforward pass through the neural network.
        SGD(training_data, test_data, epochs, mini_batch_size, eta): Stochastic Gradient Descent (SGD) method for training the neural network.
        update_mini_batch(mini_batch, eta): Update the network's weights and biases by applying gradient descent using backpropagation on a mini-batch.
        backprop(x, y): Perform backpropagation to calculate the gradients of the cost function with respect to the biases and weights of the neural network.
        predict(x): Return the predicted output for the given input vector x.
        evaluate(test_data): Return the number of test examples for which the neural network outputs the correct result.
        cost_derivative(output_activations, y): Calculate the derivative of the cost function with respect to the output activations.

    """

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Perform a feedforward pass through the neural network.

        Args:
            a (ndarray): Input to the neural network.

        Returns:
            ndarray: Output of the neural network after the feedforward pass.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, test_data, epochs, mini_batch_size, eta):
        """
        Stochastic Gradient Descent (SGD) method for training the neural network.

        Parameters:
            training_data (list): The training data, a list of tuples (x, y) where x is the input and y is the expected output.
            test_data (list): The test data, a list of tuples (x, y) where x is the input and y is the expected output.
            epochs (int): The number of epochs to train the network.
            mini_batch_size (int): The size of each mini-batch used in training.
            eta (float): The learning rate.

        Returns:
            None
        """
        n_train = len(training_data)
        n_test = len(test_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print(f"Epoch {j}: {self.evaluate(test_data)}/ {n_test}")

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation on a mini-batch.

        Args:
            mini_batch (list): A list of tuples (x, y) representing the input and corresponding desired output.
            eta (float): The learning rate.

        Returns:
            None
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Perform backpropagation to calculate the gradients of the cost function
        with respect to the biases and weights of the neural network.

        Args:
            x (ndarray): Input data for a single training example.
            y (ndarray): Target output for the corresponding input.

        Returns:
            tuple: A tuple containing the gradients of the cost function with respect
            to the biases and weights of the neural network.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # (actual out - desired out)*(derivative of potential)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def predict(self, x):
        """Return the predicted output for the given input vector x."""
        # Apply feedforward to get the output vector
        output = self.feedforward(x)

        # Apply a threshold to the output
        threshold = 0.7  # Adjust this value based on your specific problem
        predicted_label = (output > threshold).astype(int)

        return predicted_label

    def evaluate(self, test_data):
        """Return the number of test examples for which the neural network
        outputs the correct result. The neural network's output is assumed to be
        the index of whichever neuron in the final layer has the highest activation."""

        # Make predictions for each test example
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        # Compare the predictions with the actual labels
        return sum(int(y_pred == y) for (y_pred, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Calculate the derivative of the cost function with respect to the output activations.

        Args:
            output_activations (ndarray): The output activations of the neural network.
            y (ndarray): The expected output.

        Returns:
            ndarray: The derivative of the cost function with respect to the output activations.
        """
        return (output_activations-y)
        

def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z))
