import numpy as np
from OuterLayer import OutputLayer
from InnerLayer import HiddenLayer
from Model import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import urllib
from urllib.request import urlopen
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(data):

	##Helper function to one-hot-encode the IRIS dataset
	
	values = np.array(data)
	
	# integer encode
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(values)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

	return onehot_encoded

def load_iris_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    raw_data = urllib.request.urlopen(url)
    dataset = np.genfromtxt(raw_data, dtype = 'str', missing_values='?', delimiter=',')
    y_train = one_hot_encode(dataset[:,-1])
    x_train = dataset[:,:-1].astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state = 42)
    return (x_train, y_train), (x_test, y_test)


def load_mnist_data():
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images = train_images.reshape(60000, (28*28))
	train_images = train_images/ 255.0

	test_images = test_images.reshape(10000, (28*28))
	test_images = test_images/ 255.0
	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)

	return (train_images, train_labels), (test_images, test_labels)

def plot_metrics(metrics, epochs, file_name):
    
	accuracy = metrics['acc']
	val_accuracy = metrics['val_acc']
	loss = metrics['loss']
	val_loss = metrics['val_loss']
	plt.figure(figsize=(18,8))
	plt.subplot(1,2,1)
	plt.plot(range(epochs), accuracy, 'r', label = 'Train Accuracy')
	plt.plot(range(epochs), val_accuracy, 'b', label = 'Validation Accuracy')
	plt.legend()
	  
	plt.subplot(1,2,2)
	plt.plot(range(epochs), loss, 'r', label = 'Train Loss')
	plt.plot(range(epochs), val_loss, 'b', label = 'Validation Loss')

	plt.legend()
	plt.savefig(file_name)

def train(x_train, y_train, x_val = 0, y_val = 0):
	model = Model()
	model.build_model(input_size = x_train.shape[-1], output_size = y_train.shape[-1], hidden_layer_info = hidden_layers)
	metrics = model.fit(x_train = x_train, y_train = y_train, x_val = x_test, y_val = y_test, learning_rate = learning_rate, epochs = epochs)
	return metrics


if __name__ == '__main__':

    learning_rate = 0.01
    epochs = 25
    hidden_layers = [256]
	
	##Train on IRIS Dataset, with a subset of 3 classes

    (x_train, y_train), (x_test, y_test) = load_iris_data()
    print("IRIS Data Set")

    metrics = train(x_train, y_train, x_test, y_test)
    plot_metrics(metrics, epochs, 'IRIS Accuracy.png')
    print("IRIS Data Set Result: ", metrics)
	
	##Train on MNIST Dataset
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
	
    x_train =x_train[1:10000]
    y_train = y_train[1:10000]
    print("MNIST Data Set")

    metrics = train(x_train, y_train, x_test, y_test)
    plot_metrics(metrics, epochs, 'MNIST Accuracy.png')
    print("MNIST Data Set Result: ", metrics)
