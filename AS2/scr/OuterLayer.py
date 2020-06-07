
import numpy as np

class OutputLayer:

	def __init__(self, input_size, output_size):
		
		self.output_size = output_size
		self.input_size = input_size

		self.weights = 2 * np.random.rand(self.output_size, self.input_size) - 1
		self.weight_gradient = np.zeros((self.output_size, self.input_size))

		self.output = np.zeros((self.output_size,1))

	def softmax_activation(self, x, axis = -1, derivation = False):
		if derivation:
			return (np.eye(len(x)) * x - np.outer(x, x))
		else:
			y = np.exp(x)  + 1E-9
			return y / np.sum(y, axis, keepdims=True)
	
	def forward(self, input):

		self.output = self.softmax_activation(np.inner(self.weights, input))
		return self.output

	def backward(self, input, loss):

		delta = np.dot(self.softmax_activation(self.output, derivation = True), loss)
		self.weight_gradient = np.outer(delta, input)
		return np.dot(self.weights.T, delta)
