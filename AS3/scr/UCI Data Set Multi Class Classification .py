#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Programming Question
#Regression Dataset - Crimes
##Dataset URL - http://archive.ics.uci.edu/ml/datasets/communities+and+crime

import numpy as np
import urllib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import models, optimizers
from keras import layers, losses
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import BatchNormalization
import keras
import tensorflow as tf
import pandas as pd
import time
get_ipython().run_line_magic('matplotlib', 'inline')


#Setting Random seed
from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(2)


# In[10]:


def loaddata():
  url = "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"

  rawdata = urllib.request.urlopen(url)

  dataset = np.genfromtxt(rawdata, dtype = 'float32', missing_values='?', delimiter=',')

  #Delete column for Community Name - String column.
  dataset = np.delete(dataset,3,1)
  
  #Convert column for State to Categorical
  col0Cat = to_categorical(dataset[:,0])
  #Drop column 0 - will be replaced by categorical values
  dataset = np.delete(dataset,0,1)

  #Drop column 1,2 - Too many missing values
  dataset = np.delete(dataset,[0,1],1)

  #Separate predictors and target variable
  x_train, y_train = dataset[:,:-1], dataset[:,-1]

  #Append categorical tensor for state to x_train
  x_train = np.append(x_train, col0Cat, axis = 1)
  
  #Column 26 has 1 missing value - Replace it with mean of that column
  x_train[np.where(np.isnan(x_train[:,26])), 26] = np.nanmean(x_train[:,26])
  
  #Delete Multiple columns with 1675 missing values
  del_cols = [ 97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 117, 118, 119, 120, 122]
  x_train = np.delete(x_train, del_cols, 1)
  

  #Create Train and Test split  
  x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state = 42)
  

  return (x_train, y_train), (x_test, y_test)


# In[11]:


def print_graphs(history, epochs, batch_size, smooth = 0, plot_title = "L1 Loss"):
  history_dict = history.history
  mae = history_dict['mae']
  val_mae = history_dict['val_mae']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']
    
  fig = plt.figure(figsize=(15,7))
  fig.suptitle(plot_title, fontsize = 14, weight = 'bold')
  plt.subplot(1,2,1)
  plt.plot(range(epochs), mae, 'r', label = 'Train MAE')
  plt.plot(range(epochs), val_mae, 'b', label = 'Validation MAE')
  plt.legend()
  
  plt.subplot(1,2,2)
  plt.plot(range(epochs), loss, 'r', label = 'Train Loss')
  plt.plot(range(epochs), val_loss, 'b', label = 'Validation Loss')
  
  plt.legend()
  plt.show()
  
  
def define_model(num_layers = 1, num_neurons = [16], input_shape = (0,), 
                 loss = "L1", optimizer = "RMSprop", optimizer_lr = 0, 
                 dropout = 0, regularizer = "L1", reg_rate = 0, batch_norm = False):
  
  assert input_shape[0] != 0
  assert num_layers == len(num_neurons)
  
  model = models.Sequential()
  
  for i in range(num_layers):
    if reg_rate:
      if regularizer == "l1":
        model.add(layers.Dense(num_neurons[i], kernel_regularizer = keras.regularizers.l1(reg_rate), activation = 'relu', input_shape = input_shape))
      elif regularizer == "l2":
        model.add(layers.Dense(num_neurons[i], kernel_regularizer = keras.regularizers.l2(reg_rate), activation = 'relu', input_shape = input_shape))
      elif regularizer == "l1_l2":
        model.add(layers.Dense(num_neurons[i], kernel_regularizer = keras.regularizers.l1_l2(reg_rate), activation = 'relu', input_shape = input_shape))
      else:
        print("WARNING: Invalid regularizer given. Using L1 regularization with 0.01 Regularization Rate.")
        model.add(layers.Dense(num_neurons[i], kernel_regularizer = regularizers.l1(0.01), activation = 'relu', input_shape = input_shape))
    else:
      model.add(layers.Dense(num_neurons[i],  activation = 'relu', input_shape = input_shape))
      
    #Add dropout to all but the penultimate layer.
    if dropout:
      model.add(layers.Dropout(dropout))
    if batch_norm:
      model.add(BatchNormalization())
  
  model.add(layers.Dense(1))
  
  if optimizer_lr == 0:
    optimzer_lr = 0.01
    

  if optimizer == "sgd":
    optimizer = optimizers.sgd(lr = optimizer_lr)
  elif optimizer == "RMSprop":
    optimizer = optimizers.RMSprop(lr = optimizer_lr)
  elif optimizer == "Adagrad":
    optimizer = optimizers.Adagrad(lr = optimizer_lr)
  else:
    print("!!WARNING: Incompatible Optimizer provided. Using RMSprop!!")
    optimizer = optimizers.RMSprop()

  #model.summary()
  
  loss_fn = ""
  if loss == "L1":
    loss_fn = keras.losses.mean_absolute_error
  elif loss == "L2":
    loss_fn = keras.losses.mean_squared_error
  elif loss == "logcosh":
    loss_fn = keras.losses.logcosh
  elif loss == "huber":
    loss_fn = tf.losses.huber_loss
  else:
    print("!!WARNING: Incompatible loss function given. Accepted values are {L1, L2, huber, logcosh}. Using L2 loss!!")
    loss_fn = losses.mean_squared_error
  model.compile(optimizer = optimizer,
             loss = loss_fn,
             metrics = ["mae"])
  
  return model


# In[12]:


(x_train, y_train), (x_test, y_test) = loaddata()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state = 42)
print("x_train: " + str(x_train.shape))
print("y_train: " + str(y_train.shape))
print("x_val: " + str(x_val.shape))
print("y_val: " + str(y_val.shape))
print("x_test: " + str(x_test.shape))
print("y_test: " + str(y_test.shape))


# In[13]:


# Tuning Loss Function

batch_size = 512
epochs = 100

losses = ['L1', 'L2', 'huber', 'logcosh']
hidden_layers = [[8], [16], [8,8], [8,16]]
metrics = {}
best_model = ""
min_so_far = np.inf
for i, loss in enumerate(losses):
  print("Building Model with Loss: " + loss)
  for j, hidden_layer in enumerate(hidden_layers):
    print("Hidden Layers: " + str(hidden_layer))
    model = define_model(num_layers = len(hidden_layer), num_neurons = hidden_layer, 
                         input_shape = (x_train.shape[1],), loss = loss)
    history = model.fit(x_train, y_train,
                     epochs = epochs,
                     batch_size = batch_size,
                     validation_data = (x_val, y_val),
                     verbose = 0)
    test_output = model.evaluate(x_test, y_test)
    if test_output[1] < min_so_far:
      min_so_far = test_output[1]
      best_model = "Loss: " + loss + ", Hidden Layers: " + str(hidden_layer)
    if loss in metrics.keys():
      if len(metrics['L1']) < 1:
        metrics[loss] = {str(hidden_layer) : {"Test Loss" : test_output[0], "Test MAE" : test_output[1]}}
      else:
        metrics[loss].update({str(hidden_layer) : {"Test Loss" : test_output[0], "Test MAE" : test_output[1]}})
    else:
      metrics[loss] = {str(hidden_layer) : {"Test Loss" : test_output[0], "Test MAE" : test_output[1]}}
    print("Test Loss: %f, Test MAE: %f" % (test_output[0], test_output[1]))
    print_graphs(history=history, epochs = epochs, batch_size = batch_size, 
                 plot_title = loss + " Loss: " + str(hidden_layer))



print("All Combinations: ", metrics)

print("Best Model for loss: ",best_model)


# #Best Model for loss:  Loss: L1, Hidden Layers: [8, 16]

# In[14]:


#Optimizers

batch_size = 512
epochs = 100

opts = ['sgd', 'RMSprop', 'Adagrad']
lrs = [0.001, 0.005, 0.01, 0.05]

col_names =  ['Optimizer', '0.001', '0.005', '0.01', '0.05']
metrics = pd.DataFrame({}, index=np.arange(len(opts)), columns=col_names)
best_model = ""
min_so_far = np.inf
for i, opt in enumerate(opts):
  print("Building Model with Optimizer: " + opt)
  metrics.loc[i][0] = opt
  for j, lr in enumerate(lrs):
    print("Learning Rate: " + str(lr))
    start = time.clock()
    model = define_model(num_layers = 2, num_neurons = [8, 16], 
                         input_shape = (x_train.shape[1],), loss = "L1",
                        optimizer = opt, optimizer_lr = lr)
    history = model.fit(x_train, y_train,
                     epochs = epochs,
                     batch_size = batch_size,
                     validation_data = (x_val, y_val),
                     verbose = 0)
    time_taken = time.clock() - start
    test_output = model.evaluate(x_test, y_test)
    metrics.loc[i][j+1]  = {'MAE' : round(test_output[1], 4), 
                          'Time': round(time_taken, 4)}
    if test_output[1] < min_so_far:
      min_so_far = test_output[1]
      best_model = "Optimizer: " + opt + ", Learning Rate: " + str(lr)
    print("Test Loss: %f, Test MAE: %f" % (test_output[0], test_output[1]))
    print_graphs(history=history, epochs = epochs, batch_size = batch_size, 
                 plot_title = "Optimizer: " + opt + ", LR: " + str(lr))


print(metrics)


# In[15]:


#Weight Decay

batch_size = 512
epochs = 100

regs = ['l1', 'l2', 'l1_l2']
reg_rates = [0.001, 0.005, 0.01, 0.05]
indexes =  ['L1 Weight Decay', 'L2 Weight Decay', 'L1_L2 Weight Decay']
metrics = pd.DataFrame({}, index=indexes, columns=reg_rates)
best_model = ""
min_so_far = np.inf
for i, regularizer in enumerate(regs):
  print("Building Model with " + regularizer + " weight decay")
  #metrics.loc[i][0] = opt
  for j, reg_rate in enumerate(reg_rates):
    print("Regularization Rate: " + str(reg_rate))
    start = time.clock()
    model = define_model(num_layers = 2, num_neurons = [8, 16], 
                         input_shape = (x_train.shape[1],), loss = "L1",
                        optimizer = opt, optimizer_lr = lr,
                        regularizer = regularizer, reg_rate = reg_rate)
    history = model.fit(x_train, y_train,
                     epochs = epochs,
                     batch_size = batch_size,
                     validation_data = (x_val, y_val),
                     verbose = 0)
    time_taken = time.clock() - start
    test_output = model.evaluate(x_test, y_test)
    metrics.loc[indexes[i]][reg_rates[j]]  = {'MAE' : round(test_output[1], 4)}
    
    if test_output[1] < min_so_far:
      min_so_far = test_output[1]
      best_model = "Regularizer: " + regularizer + ", Regularization Rate: " + str(reg_rate)
    print("Test Loss: %f, Test MAE: %f" % (test_output[0], test_output[1]))
    print_graphs(history=history, epochs = epochs, batch_size = batch_size, 
                 plot_title = "Regularizer: " + regularizer + ", Regularization Rate: " + str(reg_rate))


print("All Combinations: ", metrics)

print("Best Model for Weight Decay: ",best_model)


# Best Model for Weight Decay:  Regularizer: l2, Regularization Rate: 0.01

# In[17]:


#Dropout


batch_size = 512
epochs = 100

dropouts = np.arange(0.2, 0.55, 0.05)
indexes =  ['L1 Weight Decay', 'L2 Weight Decay', 'L1_L2 Weight Decay']
metrics = pd.DataFrame(0.0, index=dropouts, columns=['MAE'])
best_model = ""
min_so_far = np.inf
for i, dropout in enumerate(dropouts):
  print("Building Model with Dropout level: " + str(dropout))
  #metrics.loc[i][0] = opt
  start = time.clock()
  model = define_model(num_layers = 2, num_neurons = [8, 16], 
                       input_shape = (x_train.shape[1],), loss = "logcosh",
                       optimizer = opt, optimizer_lr = lr,
                       regularizer = 'l2', reg_rate = 0.005,
                       dropout = dropout)
  history = model.fit(x_train, y_train,
                     epochs = epochs,
                     batch_size = batch_size,
                     validation_data = (x_val, y_val),
                     verbose = 0)
  time_taken = time.clock() - start
  test_output = model.evaluate(x_val, y_val)
  metrics.loc[dropouts[i]]['MAE']  = round(test_output[1], 4)
    
  if test_output[1] < min_so_far:
    min_so_far = test_output[1]
    best_model = "Dropout: " + str(dropout)
  print("Test Loss: %f, Test MAE: %f" % (test_output[0], test_output[1]))
  print_graphs(history=history, epochs = epochs, batch_size = batch_size, 
                 plot_title = "Dropout Level: " + str(dropout))


print("All Combinations: ", metrics)

print("Best Model for Dropouts: ",best_model)


# In[18]:


# final model
model = define_model(num_layers = 2, num_neurons = [8, 16], 
                       input_shape = (x_train.shape[1],), loss = "logcosh",
                       optimizer = opt, optimizer_lr = lr,
                       regularizer = 'l2', reg_rate = 0.005,
                       dropout = dropout, batch_norm = True)
history = model.fit(x_train, y_train,
                     epochs = epochs,
                     batch_size = batch_size,
                     validation_data = (x_val, y_val),
                     verbose = 0)
test_output = model.evaluate(x_val, y_val)
print("Test Loss: %f, Test MAE: %f" % (test_output[0], test_output[1]))
print_graphs(history=history, epochs = epochs, batch_size = batch_size, 
                 plot_title = "Adding Batch Normalization")


# In[ ]:




