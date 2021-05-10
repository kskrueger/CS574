# Tutorial from: https://towardsdatascience.com/3-steps-to-forecast-time-series-lstm-with-tensorflow-keras-ba88c6f05237

# import packages
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import time

import os
from load import read_csv, process_daily_items

# CONSTANTS
timestep_size = 7

# DATASET PROCESSING
raw_dataset = read_csv("dataset/Groceries_dataset.csv")  # read in the dataset
daily_item_counts, date_labels, food_labels = process_daily_items(raw_dataset)

num_foods = len(food_labels)
num_timesteps = daily_item_counts.shape[0] - timestep_size - 1
X_chunks = np.zeros((num_timesteps, timestep_size, num_foods))
Y_chunks = np.zeros((num_timesteps, 1, num_foods))
for i in range(0, num_timesteps):
    X_chunks[i, :, :] = daily_item_counts[i:i+7, :]
    Y_chunks[i, :, :] = daily_item_counts[i+7+1, :]

np.random.shuffle(X_chunks)
np.random.shuffle(Y_chunks)

split = 512  # int(num_timesteps*.7)
x_train = X_chunks[:split]
x_test = X_chunks[split:]
y_train = Y_chunks[:split]
y_test = Y_chunks[split:]

# Create the Keras model.
# Use hyperparameter optimization if you have the time.

ts_inputs = tf.keras.Input(shape=(timestep_size, 1))

# units=10 -> The cell and hidden states will be of dimension 10.
#             The number of parameters that need to be trained = 4*units*(units+2)
x = layers.LSTM(units=50, return_sequences=True)(ts_inputs)
x = layers.LSTM(units=50, return_sequences=True)(x)
x = layers.LSTM(units=50)(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)

# Specify the training configuration.
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])

model.summary()

# train in batch sizes of 128.
BATCH_SIZE = 128
NUM_EPOCHS = 50
NUM_CHUNKS = split

X, y = np.sum(x_train, axis=2), np.sum(y_train, axis=2)
Xv, yv = np.sum(x_test, axis=2), np.sum(y_test, axis=2)
model.fit(X, y, validation_data=(Xv, yv), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

for epoch in range(NUM_EPOCHS):
    print('epoch #{}'.format(epoch))
    for i in range(NUM_CHUNKS):
        # X, y = x_train[i, :, 60:61].T, y_train[i, :, 60]
        X, y = np.array([np.sum(x_train[i], axis=1)]), np.sum(y_train[i], axis=1)
        Xv, yv = np.array([np.sum(x_train[i], axis=1)]), np.sum(y_train[i], axis=1)

        # model.fit does train the model incrementally. ie. Can call multiple times in batches.
        # https://github.com/keras-team/keras/issues/4446
        model.fit(x=np.array(X), y=np.array(y), validation_data=(X, y), batch_size=BATCH_SIZE)

    # shuffle the chunks so they're not in the same order next time around.
    np.random.shuffle(X_chunks)
    np.random.shuffle(Y_chunks)


# # VALIDATION
#
# # evaluate the model on the validation set.
# #
# # Create the validation CSV like we did before with the training.
# global_active_power_val = df_val['Global_active_power'].values
# global_active_power_val_scaled = scaler.transform(global_active_power_val.reshape(-1, 1)).reshape(-1, )
#
# # The history length in minutes.
# history_length = 7*24*60
# # The sampling rate of the history. Eg. If step_size = 1, then values from every minute will be in the history.
# step_size = 10
# # If step size = 10 then values every 10 minutes will be in the history.
# # The time step in the future to predict. Eg. If target_step = 0, then predict the next timestep after the end of the history period.
# # If target_step = 10 then predict 10 timesteps the next timestep (11 minutes after the end of history).
# target_step = 10
#
# # The csv creation returns the number of rows and number of features. We need these values below.
# num_timesteps = create_ts_files(global_active_power_val_scaled,
#                                 start_index=0,
#                                 end_index=None,
#                                 history_length=history_length,
#                                 step_size=step_size,
#                                 target_step=target_step,
#                                 num_rows_per_file=128*100,
#                                 data_folder='ts_val_data')
