import tensorflow as tf
import keras
import matplotlib as plt
import numpy as np

from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras import backend as K

iv3 = InceptionV3()
print(iv3.summary())

from keras.preprocessing import image
x = image.img_to_array(image.load_img('tw.jpg', target_size=(299, 299)))

#cambio de rango de 0 - 255 a -1 - 1
x /= 255
x -= 0.5
x *= 2 

x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
y = iv3.predict(x)
decode_predictions(y)