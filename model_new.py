import numpy as np
import random


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


class NeuralNetwork:
    
    def __init__(self, layers):
        self.h_biases = np.random.randn(layers[1],1)
        self.o_biases = np.random.randn(layers[2],1)
        
        self.h_weights = np.random.randn(layers[1],layers[0])
        self.o_weights = np.random.randn(layers[2],layers[1])
    
    def forward_propagation(self, x):
        a = sigmoid(np.dot(self.h_weights, x) + self.h_biases)
        
        output = sigmoid(np.dot(self.o_weights, a) + self.o_biases)
        
        return output
    
    def update_mini_batch(self, batch, l_rate):
        o_b = np.zeros(self.o_biases.shape)
        h_b = np.zeros(self.h_biases.shape)
        
        o_w = np.zeros(self.o_weights.shape)
        h_w = np.zeros(self.h_weights.shape)
        
        for x, y in batch:
            # Ensure x is a column vector. This may already be the case, but let's be sure.
            x = x.reshape(-1, 1)  # Reshaping x to have the proper shape
            
            # Ensure y is properly vectorized. Assuming y is a scalar representing the class, vectorized_result(j) is used.
            y = vectorized_result(y) if not isinstance(y, np.ndarray) else y  # Vectorizing y
            
            o_del_b, h_del_b, o_del_w, h_del_w = self.backprop(x, y)
            
            o_b = o_b + o_del_b
            h_b = h_b + h_del_b
            o_w = o_w + o_del_w
            h_w = h_w + h_del_w
        
        self.o_weights = self.o_weights - (l_rate/len(batch))*o_w
        self.h_weights = self.h_weights - (l_rate/len(batch))*h_w
        self.o_biases = self.o_biases - (l_rate/len(batch))*o_b
        self.h_biases = self.h_biases - (l_rate/len(batch))*h_b

    
    def backprop(self, x, y):
        z_h = np.dot(self.h_weights, x) + self.h_biases
        a_h = sigmoid(z_h)
        
        z_o = np.dot(self.o_weights, a_h) + self.o_biases
        predicted = sigmoid(z_o)
        
        delta = (predicted - y) * sigmoid_prime(z_o)
        
        o_del_b = delta
        o_del_w = np.dot(delta, a_h.transpose())
        
        delta = np.dot(self.o_weights.transpose(), delta) * sigmoid_prime(z_h)
        
        h_del_b = delta
        h_del_w = np.dot(delta, x.transpose())
        
        return (o_del_b, h_del_b, o_del_w, h_del_w)
        
    def fit(self, train_data, epochs, mini_batch_size, learning_rate):
        n = len(train_data)
        for i in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[j:j+mini_batch_size] for j in range(0,n, mini_batch_size)]
            for batch in batches:
                self.update_mini_batch(batch, learning_rate)
            print("epoch {} completed".format(i))
    
    def accuracy(self, test_data):
        test_results = [(np.argmax(self.forward_propagation(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


