import numpy as np
import random
import matplotlib.pyplot as plt


class NeuralNetwork:
    '''
    Multilayer Perceptron Neural Network
    '''
    def __init__(self, num_inputs, hidden_layers, num_outputs):
        '''
        This case 
        num_inputs = 2
        hidden_layers = [3,3]
        num_outputs = 2
        '''
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        # Learning rate parameter
        #self.alpha = alpha

        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        # Save derivatives per layers
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # Save activations per layers
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):

        activations = inputs

        # Save the activations for backpropagation
        self.activations[0] = activations

        # Iter through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self.sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i+1] = activations

        # return the output layer activation
        return activations

    def back_propagate(self, error, verbose=False):

        for i in reversed(range(len(self.derivatives))):
            
            # Get activations for previous layer
            activations = self.activations[i + 1]

            # Apply sigmoid derivate function
            delta = error * self.sigmoid_derivative(activations)

            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            
            current_activations = self.activations[i]
            # Reshape
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            self.derivatives[i] = np.dot(current_activations, delta_reshaped)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print('Derivatives for W_{} : {}'.format(i, self.derivatives[i]))
        
        return error

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return x*(1-x)

    def gradient_descent(self, lr):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += lr*derivatives

    def mean_squared_error(self, target, output):
        '''
        Mean Squared Error Loss Function
        '''
        return np.average((target - output) ** 2)


if __name__ == '__main__':

    mlp = NeuralNetwork(2, [3,3], 2)

    # Create numpy array
    inputs = np.array([0.3, 0.2])
    target = np.array([0.5])

    # Create forward propagation
    output = mlp.forward_propagate(inputs)

    # Cal error
    error = target - output

    # Create backpropagation
    mlp.back_propagate(error, verbose=True)