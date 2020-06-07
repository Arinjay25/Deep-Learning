import numpy as np
from InnerLayer import HiddenLayer
from OuterLayer import OutputLayer
from collections import defaultdict

class Model:
	def __init__(self):
		self.input_size = 0
		self.output_size = 0
		self.hidden_layers = []
		self.output_layer = 0
		self.metrics = defaultdict(list)

	def build_model(self, input_size, output_size, hidden_layer_info):
		np.random.seed(42)
		self.input_size = input_size
		self.output_size = output_size

		prev_layer_size = input_size
		
		for hidden_layer_size in hidden_layer_info:
			self.hidden_layers.append(HiddenLayer(prev_layer_size, hidden_layer_size))
			prev_layer_size = hidden_layer_size

		self.output_layer = OutputLayer(prev_layer_size, self.output_size)

	def feedforward(self, input):##Feedforward function.

		current_layer_input = input
		for layer in self.hidden_layers:
			current_layer_input = layer.forward(current_layer_input)
		yhat = self.output_layer.forward(current_layer_input)

		return yhat

	def backward_propagation(self, y_pred, y_true, x_train):##Backpropagation function.

		loss_derivative = self.categorical_loss(y_pred, y_true, derivation = True)
		prev_layer_output = self.hidden_layers[-1].output
		delta = self.output_layer.backward(prev_layer_output, loss_derivative)
		next_layer_weight = self.output_layer.weights

		for layer_no in range(len(self.hidden_layers)-1,0,-1):
			
			prev_layer_output = self.hidden_layers[layer_no-1].output
			delta =  self.hidden_layers[layer_no].backward(delta, prev_layer_output, next_layer_weight)
			next_layer_weight = self.hidden_layers[layer_no].weights

		delta =  self.hidden_layers[0].backward(delta, x_train, next_layer_weight)

	def update_parameters(self, learning_rate):

		for layer in self.hidden_layers:
			layer.weights = layer.weights - (learning_rate * layer.weight_gradient)
		
		self.output_layer.weights = self.output_layer.weights - (learning_rate * self.output_layer.weight_gradient)


	def categorical_loss(self, y_pred, y_true, derivation = False):##Categorical Cross Entropy Function. 
	#Derivation is set to true in case of backward propagation, to get the derivation of the loss function.
		
		if derivation:
			return - (y_true/y_pred)
		else:
			return -np.sum(np.log(y_pred[y_true == 1]))  / len(y_pred)


	def accuracy(self, y_pred, y_train):

		return np.round(np.float(sum(np.argmax(y_train, axis = -1) == np.argmax(y_pred, axis = -1)))/len(y_pred) * 100, 2)


	def shuffle_dataset(self, x, y):

		assert len(x) == len(y)
		p = np.random.permutation(len(y))
		return x[p], y[p]

	def evaluate_model(self, x, y):#loss and accuracy with respect to x and y.
		y_pred = np.zeros(y.shape)
		loss = 0
		for i in range(len(y)):
			y_pred[i] = self.feedforward(x[i])
			loss = loss + self.categorical_loss(y_pred[i], y[i])	
		loss = loss / len(y)
		accuracy = self.accuracy(y_pred, y)
		return (loss, accuracy)


	def fit(self, x_train, y_train, x_val = 0, y_val = 0, learning_rate = 0.01, epochs = 5):

		for epoch in range(epochs):
			print("Epoch: " + str(epoch+1))
			for i in range(len(x_train)):
				y_pred = self.feedforward(x_train[i])
				loss = self.categorical_loss(y_pred, y_train[i])
				self.backward_propagation(y_pred, y_train[i], x_train[i])
				self.update_parameters(learning_rate)
			
			(train_loss, train_accuracy) = self.evaluate_model(x_train, y_train)
			(val_loss, val_accuracy) = self.evaluate_model(x_val, y_val)
			self.metrics['acc'].append(train_accuracy)
			self.metrics['loss'].append(train_loss)
			self.metrics['val_acc'].append(val_accuracy)
			self.metrics['val_loss'].append(val_loss)
			x_train, y_train = self.shuffle_dataset(x_train, y_train)
		return self.metrics


	
			


