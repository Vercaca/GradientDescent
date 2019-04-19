import random

import sys
sys.path.append('../../ml')

from utils import draw_progess_bar
from activation import ActivationFunction

STR_REPORT_BROADER = '+'+'-' * 60 + '+'

class GradientDescent:
    def __init__(self, activ_func = 'Sigmoid', layers=[], learning_rate='0.01', debug=True):
        self.__activation = activ_func
        self.layers = layers
        self.learning_rate = learning_rate
        self.debug = debug

    def update(self, targets):
        self.feed_backward(targets)
        self.update_weights()

    def neuron_calculate_delta(self, neuron, error):
        neuron.delta = error * self.__activation.dfunc(neuron.output)


    def feed_backward(self, targets): # backpropagating
        if len(targets) != len(self.layers[-1].neurons):
            raise Exception('wrong target numbers')

        # calculate deltas of output layer
        # ''' Delta weight_ji = - (target_j - output_j) * deactivate_func(h_j) * input_i
        for j, neuron_j in enumerate(self.layers[-1].neurons):
            error = - (targets[j] - neuron_j.output)
            self.neuron_calculate_delta(neuron_j, error)


        # calculate the hidden layers
        n_hidden_layers = len(self.layers[:-1])
        l = n_hidden_layers - 1

        while l >= 0:
            curr_layer, last_layer = self.layers[l], self.layers[l+1]

            for i, neuron_i in enumerate(curr_layer.neurons):
                # sum up the errors sent from the last layer
                total_error = 0
                for j, neuron_j in enumerate(last_layer.neurons):
                    total_error += neuron_j.delta * neuron_j.weights[i] # total_error += delta_j * input_i_to_j

                self.neuron_calculate_delta(neuron_i, total_error)

            if self.debug:
                print("Layer {}: deltas: {}".format(l+1, curr_layer.deltas))

            l -= 1

        return None

    def layer_update_weights(self, layer, learning_rate):
        for neuron in layer.neurons:
            self.neuron_update_weights(neuron, learning_rate)
        return None

    def neuron_update_weights(self, neuron, learning_rate):
        for i in range(neuron.n_weights):
            neuron.weights[i] -= learning_rate * neuron.delta * neuron.inputs[i]
        neuron.bias -= learning_rate * neuron.delta

        # update output
        neuron.calculate_output(neuron.inputs)

        return None


    def update_weights(self):
        learning_rate = self.learning_rate
        for l in self.layers:
            self.layer_update_weights(l, learning_rate)

        return None
