import numpy as np

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = [
        [{'weights': np.random.rand(n_inputs + 1) - 0.5} for _ in range(n_hidden)],
        [{'weights': np.random.rand(n_hidden + 1) - 0.5} for _ in range(n_outputs)]
    ]
    return network

def activate(weights, inputs):
    return np.dot(weights[:-1], inputs) + weights[-1]

def transfer(activation):
    return 1 / (1 + np.exp(-activation))

def forward_propagate(network, row):
    inputs = np.array(row)
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = np.array(new_inputs)
    return inputs

def transfer_derivative(output):
    return output * (1 - output)

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network)-1:
            for j in range(len(layer)):
                error = sum([neuron['weights'][j] * neuron['delta'] for neuron in network[i + 1]])
                errors.append(error)
        else:
            for j, neuron in enumerate(layer):
                errors.append(expected[j] - neuron['output'])
        for j, neuron in enumerate(layer):
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
    for i, layer in enumerate(network):
        inputs = np.array(row[:-1] if i == 0 else [neuron['output'] for neuron in network[i - 1]])
        for neuron in layer:
            neuron['weights'][:-1] -= l_rate * neuron['delta'] * inputs
            neuron['weights'][-1] -= l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            row = np.array(row) if not isinstance(row, np.ndarray) else row
            outputs = forward_propagate(network, row)
            expected = np.zeros(n_outputs)
            expected[row[-1]] = 1
            sum_error += sum((expected-outputs)**2)
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print(f'>epoch={epoch}, lrate={l_rate}, error={sum_error}')

def predict(network, row):
    row = np.array(row) if not isinstance(row, np.ndarray) else row
    outputs = forward_propagate(network, row)
    return np.argmax(outputs)

def back_propagation(train_df, test_df, l_rate, n_epoch, n_hidden):
    train = train_df.values
    test = test_df.values
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = []
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions



def accuracy_metric(actual, predicted):
    correct = sum(1 for i, j in zip(actual, predicted) if i == j)
    return correct / len(actual) * 100


def evaluate_algorithm(train_df, test_df, algorithm, l_rate, n_epoch, n_hidden):
    predicted = algorithm(train_df, test_df, l_rate, n_epoch, n_hidden)
    actual = [row[-1] for row in test_df.values]
    accuracy = accuracy_metric(actual, predicted)
    return accuracy

