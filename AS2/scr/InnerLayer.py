
import numpy as np

class HiddenLayer:

	def __init__(self, input_size, output_size):
		
		self.output_size = output_size
		self.input_size = input_size

		self.weights = 2 * np.random.rand(self.output_size, self.input_size) - 1
		self.weight_gradient = np.zeros((self.output_size, self.input_size))

		self.output = np.zeros((self.output_size,1))

	def sigmoid_activation(self, x, derivation = False):
		if derivation:
			return (x * (1 - x) * np.eye(len(x)))
			
		else:
			return 1/(1+np.exp(-x))

	def forward(self, input):
		self.output = self.sigmoid_activation(np.inner(self.weights, input)) 
		return self.output

	def backward(self, delta, input, next_layer_weight):

		##Delta stores the propagated error until the current layer.
		delta = np.dot(self.sigmoid_activation(self.output, derivation = True), delta)
		self.weight_gradient = np.outer(delta, input)
		return np.dot(self.weights.T, delta)


