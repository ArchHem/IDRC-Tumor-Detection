import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.metrics import Precision

"""DELETE THE images subdirectory to run, otherwise it will recognize a non-valid directory!"""

import pathlib
"""Link to pre-split dataset"""
imgs = pathlib.Path('Images1')

"""Image count - might be useful later on"""
image_count = len(list(imgs.glob('*/*.png')))

"""Presplit dataset into LIST object, if need be, good for displaying
use img1 = PIL.Image.open(str(positives[i])) to display the i-th member of the positives, 
together with img1.show()"""

positives = list(imgs.glob('pos/*'))

negatives = list(imgs.glob('non/*'))


"""We need to think about this, we need to look into how it resizes
images, might introduce noise - Bendeguz"""

"""Batch size, I need to read about this more, but since I had to work with little amount of images,
I had just taken the value in the tutorial and did a linear 
extrapolation in the way I assume it to be good...

20% split is good I guess?"""

batch_size = 50
height = 400
width = 400
seed = 101

train_ds = tf.keras.utils.image_dataset_from_directory(
  imgs,
  validation_split=0.2,
  subset="training",
  seed=seed,
  image_size=(height, width),
  batch_size=batch_size,

)

val_ds = tf.keras.utils.image_dataset_from_directory(
  imgs,
  validation_split=0.2,
  subset="validation",
  seed=seed,
  image_size=(height, width),
  batch_size=batch_size,

)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

"""normalize RGB"""
normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

num_classes = len(class_names)

"""shape cehcking, it is RGB color space, and correctly normalized? Why does it push greyspace error?"""
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

"""Dark magic form here on. Will need to read on this. I include the resizing on the model."""

def hsv_conversion(x):
    return tf.image.rgb_to_hsv(x)

"""transform to rgb first or nah? layers.Lambda(hsv_conversion, input_shape = (height, width, 3))"""
"""Exclude random rotation (?)"""

dimensions = 3
kernel_s = 3
neg, pos = len(negatives), len(positives)


"""single neuron output - binary model. 2 additional layers, higher quality in general"""
model = Sequential([
  layers.Resizing(height,width),
  layers.Rescaling(1./255, input_shape=(height, width, dimensions)),
  layers.RandomZoom(0.05),
  layers.Conv2D(16, kernel_s, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, kernel_s, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, kernel_s, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128,kernel_s,padding = 'same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128,kernel_s,padding = 'same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(1, activation = 'sigmoid')
])
"""class weights - use inverse multiplicants"""
cl_wght = {0: (1 / neg) * ((neg + pos) / 1), 1:(1 / pos) * ((neg + pos) / 1)}
print(cl_wght)
"""From_logits = True crashes it, pls investigate why

tried, did not work, returning to weighted example: tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, alpha = 0.25, gamma = 2)"""
"""try adding precision label: https://github.com/huggingface/transformers/issues/10075"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])



epochs = 20

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  class_weight = cl_wght
)

print(history.history)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

model.save('models/higher_q_model')

model.summary()

plt.show()