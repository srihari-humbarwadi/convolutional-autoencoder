
# coding: utf-8

# In[1]:


from tensorflow.python.keras.layers import Conv2D, ReLU, BatchNormalization, Dense, Input, Conv2DTranspose, UpSampling2D, Flatten, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.optimizers import Nadam
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets import mnist


# In[2]:


img_height, img_width = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = np.expand_dims(x_train, axis=-1)
print(X.shape)


# In[3]:


input_image = Input(shape=(img_height, img_width, 1), name='image_imput')
x = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', name ='encoder_conv1', activation='relu')(input_image)
x = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', name ='encoder_conv2', activation='relu')(x)
x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='valid', name ='encoder_conv3', activation='relu')(x)
x = Flatten()(x)
encoded = Dense(units=10)(x)

y = Dense(units=1152, activation='relu')(encoded)
y = Reshape((3, 3, 128))(y)
y = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', name ='decoder_deconv1', activation='relu')(y)
y = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', name ='decoder_deconv2', activation='relu')(y)
decoded_image = Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(2, 2), padding='same', name ='decoder_deconv3', activation='relu')(y)


CAE = Model(inputs = input_image, outputs = decoded_image, name = 'CAE')


# In[4]:


tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(filepath='models/top_weights.h5', monitor='acc', save_best_only='True', save_weights_only='True', verbose=1)
es = EarlyStopping(monitor='loss', patience=15, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss')
callbacks = [tb, mc, es, rlr]
CAE.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# In[ ]:

# CAE.load_weights('models/top_weights.h5')
# CAE.save('CAE.h5')
CAE.fit(X, X, epochs=1000, batch_size=256, callbacks=callbacks)

