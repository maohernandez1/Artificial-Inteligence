import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np

from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras import backend as K

iv3 = InceptionV3()

from keras.preprocessing import image
x = image.img_to_array(image.load_img('hacked.png', target_size=(299, 299)))

#cambio de rango de 0 - 255 a -1 - 1
x /= 255
x -= 0.5
x *= 2 

x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
y = iv3.predict(x)
y.shape

print(decode_predictions(y))

#ataque adversario
inp_layer = iv3.layers[0].input
out_layer = iv3.layers[-1].output

target_class = 951
loss = out_layer[0, target_class]
grad = K.gradients(loss, inp_layer)[0]
optimize_gradient = K.function([inp_layer, K.learning_phase()], [grad, loss])
adv = np.copy(x)
pert = 0.01
max_pert = x + 0.01
min_pert = x - 0.01
cost = 0.0
while cost < 0.95:
    gr, cost = optimize_gradient([adv, 0])
    adv += gr
    adv = np.clip(adv, min_pert, max_pert)
    adv = np.clip(adv, -1, 1)
    print("Target cost:", cost)

hacked = np.copy(adv)

adv /= 2
adv += 0.5
adv *= 255

plt.imshow(adv[0].astype(np.uint8))
plt.show()