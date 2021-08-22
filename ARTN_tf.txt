import pandas as pd
import numpy as np
import glob
#import statistics as stat
from tqdm import tqdm
from keras.layers import LeakyReLU
import math

# Libs
# Recurrent Neural Network
import tensorflow as tf
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import add
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate

"""# Custom Training

"""

import os
import matplotlib.pyplot as plt

import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return tf.keras.losses.MSE(y_true=y, y_pred=y_)

#l = loss(model, features, labels, training=False)
#print("Loss test: {}".format(l))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

## Note: Rerunning this cell uses the same model variables

# Keep results for plotting

anim_len = 60

path =r'/content/drive/MyDrive/data_CMU/MartialArtsBvh_valid'

filenames = glob.glob(path + "/135*_pos.csv")

train_loss_results = []

num_epochs = 401

epoch_loss_avg = tf.keras.metrics.Mean()

for epoch in range(num_epochs):
  obs_loss = []
  epoch_loss = []  

  # Training loop
  for filename in filenames:
    
    df_norm, df_split = preprocess(filename,anim_len)

    num_obs=int(df_norm.iloc[0:df_norm.shape[0]-1].shape[0]/anim_len)

    #for j in range(1,len(df_split)):    
    #print("idx: " + str(j))         
    
    #for i in range(1,j):
    #print("j: " + str(j) + " i: " + str(i))    
    
    state_frame = np.reshape(np.array(df_norm.iloc[0:num_obs*anim_len,1:67]),(num_obs,anim_len,66))
    #state_frame = np.reshape(np.array(df_norm.iloc[0:len(df_norm)-1,1:67]),(1,1,66))

    state_resp = np.reshape(np.array(df_norm.iloc[1:num_obs*anim_len+1,1:67]),(num_obs,anim_len,66))
    #state_resp = np.reshape(np.array(df_norm.iloc[i+1][1:67]),(1,1,66))

    target_frame = np.reshape(state_resp[:,state_resp.shape[1]-1,:],(num_obs,1,66))*np.ones(state_frame.shape)
    
    offset_frame = target_frame - state_frame

    # Optimize the model
    loss_value, grads = grad(model_RTN, [target_frame, offset_frame, state_frame], state_resp)

    optimizer.apply_gradients(zip(grads, model_RTN.trainable_variables))  
    # loss average per epoch
    epoch_loss_avg.update_state(loss_value)

  train_loss_results.append(epoch_loss_avg.result())      
  # End epoch 
  if epoch % 1 == 0:
    print("Epoch {:03d}: Loss: {:.3f}".format(epoch,epoch_loss_avg.result()))
  
  if epoch % 50 == 0:
    print("Epoch {:03d} model saved".format(epoch))    
    # serialize weights to HDF5
    model_RTN.save_weights("model_weights_indian.h5")
    print("Saved model to disk")
    
    #model_RTN.save(ModelPath, overwrite=True)

state_resp.shape

