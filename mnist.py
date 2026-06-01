
# importing the dependencies

# %% step 1: load the datasets
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist #type: ignore
from tensorflow.math import confusion_matrix

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# %%
print(type(x_train))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# %%
print(x_train[10])
# %%
plt.imshow(x_train[50])
plt.show()
print(y_train[50])
# %%
print(np.unique(y_train))
print(np.unique(y_test))
# %%
x_train = x_train/255
x_test = x_test/255
# %%
print(x_train[10])
# %%
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
# %%
