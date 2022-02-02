#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from keras.datasets import fashion_mnist


# In[27]:


## Seeding
np.random.seed(42)
tf.random.set_seed(42)


# In[28]:


(train_input, train_target), (test_input, test_target) = fashion_mnist.load_data()
train_input,test_input =train_input.astype('float32')/255.0, test_input.astype('float32')/255.0


# In[29]:


## Flattening the images.
train_input = np.reshape(train_input, (-1, 28 * 28 * 1))
test_input = np.reshape(test_input, (-1, 28 * 28 * 1))
print(train_input.shape, test_input.shape)


# In[30]:


input_img= keras.Input(shape=(784,))


# In[31]:


def encoder(input_img):
    encoder_layer1 = Dense(units=128, activation='relu')(input_img)
    encoder_layer2 = Dense(units=64, activation='relu')(encoder_layer1)
    encoder_layer3 = Dense(units=32, activation='relu')(encoder_layer2)
    return encoder_layer3


# In[32]:


def decoder(encoder_layer3):
    decoder_layer1 = Dense(units=64, activation='relu')(encoder_layer3)
    decoder_layer2 = Dense(units=128, activation='relu')(decoder_layer1)
    decoder_layer3 = Dense(units=784, activation='relu')(decoder_layer2)
    return decoder_layer3
    


# In[33]:


encoder = tf.keras.Model(input_img, encoder(input_img))
autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy','mse'])


# In[34]:


autoencoder.summary()


# In[35]:


autoencoder_train = autoencoder.fit(train_input, train_input , epochs=30,batch_size=256,shuffle=True,validation_data = (test_input, test_input))


# In[36]:


# summarize history for loss
plt.plot(autoencoder_train.history['loss'])
plt.plot(autoencoder_train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[37]:


# summarize history for accuracy
plt.plot(autoencoder_train.history['accuracy'])
plt.plot(autoencoder_train.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[38]:


encoder_test = encoder.predict(test_input)
autoencoder_test = autoencoder.predict(test_input)

print(autoencoder_test.shape)


# In[39]:


import random
c=[random.randint(0,10000),random.randint(0,10000),random.randint(0,10000),random.randint(0,10000)]
plt.figure(figsize=(40, 4))
for i in range(len(c)):
  # display original images
  ax = plt.subplot(3, 20, i + 1)
  plt.imshow(test_input[c[i]].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  #display reconstructed images
  ax = plt.subplot(3, 20, 2*20 +i+ 1)
  plt.imshow(autoencoder_test[c[i]].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  
    
plt.show()


# In[40]:


#Defining, training, and ploting the results of the second model with higher depth 
def encoder2(input_img):
    encoder_layer1 = Dense(units=392, activation='relu')(input_img)
    encoder_layer2 = Dense(units=196, activation='relu')(encoder_layer1)
    encoder_layer3 = Dense(units=128, activation='relu')(encoder_layer2)
    encoder_layer4 = Dense(units=64, activation='relu')(encoder_layer3)
    encoder_layer5 = Dense(units=32, activation='relu')(encoder_layer4)
    return encoder_layer5


# In[41]:


def decoder2(encoder_layer3):
    decoder_layer1 = Dense(units=64, activation='relu')(encoder_layer3)
    decoder_layer2 = Dense(units=128, activation='relu')(decoder_layer1)
    decoder_layer3 = Dense(units=196, activation='relu')(decoder_layer2)
    decoder_layer4 = Dense(units=392, activation='relu')(decoder_layer3)
    decoder_layer5 = Dense(units=784, activation='relu')(decoder_layer4)
    return decoder_layer5


# In[42]:


encoder_2 = tf.keras.Model(input_img, encoder2(input_img))
autoencoder_2 = Model(input_img, decoder2(encoder2(input_img)))
autoencoder_2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy','mse'])


# In[43]:


autoencoder_2.summary()


# In[44]:


autoencoder_2.compile(optimizer='adam', loss='MSE', metrics=['accuracy'])


# In[45]:


autoencoder2_train = autoencoder.fit(train_input, train_input , epochs=30,batch_size=256,shuffle=True,validation_data = (test_input, test_input))


# In[46]:


# summarize history for loss
plt.plot(autoencoder2_train.history['loss'])
plt.plot(autoencoder2_train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[47]:


# summarize history for accuracy
plt.plot(autoencoder2_train.history['accuracy'])
plt.plot(autoencoder2_train.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[48]:


encoder2_test = encoder.predict(test_input)
autoencoder2_test = autoencoder.predict(test_input)

print(autoencoder2_test.shape)


# In[49]:


c=[random.randint(0,10000),random.randint(0,10000),random.randint(0,10000),random.randint(0,10000)]
plt.figure(figsize=(40, 4))
for i in range(len(c)):
  # display original images
  ax = plt.subplot(3, 20, i + 1)
  plt.imshow(test_input[c[i]].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  #display reconstructed images
  ax = plt.subplot(3, 20, 2*20 +i+ 1)
  plt.imshow(autoencoder_test[c[i]].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  
    
plt.show()


# In[ ]:





# In[ ]:




