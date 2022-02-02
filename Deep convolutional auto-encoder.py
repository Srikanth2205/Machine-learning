#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0


# In[2]:


import keras
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import  *
from tensorflow.keras.losses import MSE
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras.models import Model


# In[3]:


(train_input, train_target), (test_input, test_target) = fashion_mnist.load_data()
train_input,test_input =train_input.astype('float32')/255.0, test_input.astype('float32')/255.0


# In[4]:


training_data = train_input.reshape(-1, 28,28, 1)
testing_data = test_input.reshape(-1, 28,28, 1)


# In[5]:


print("shape of training data:-",training_data.shape)
print("shape of testing data",testing_data.shape)
print(training_data.dtype)
print(testing_data.dtype)
print(np.max(training_data), np.max(testing_data))


# In[6]:


inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))


# In[7]:


def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv_layer1 = Conv2D(24, (3, 3), activation='relu', padding='same')(input_img)
    conv_layer1 = BatchNormalization()(conv_layer1)
    pooling_1 = MaxPooling2D(pool_size=(2, 2),padding='same')(conv_layer1) 
    conv_layer2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pooling_1) 
    conv_layer2 = BatchNormalization()(conv_layer2)
    pooling_2 = MaxPooling2D(pool_size=(2, 2),padding='same')(conv_layer2)
    conv_layer3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pooling_2) 
    conv_layer3 = BatchNormalization()(conv_layer3)
    return conv_layer3


# In[8]:


def decoder(conv_layer3):
    #decoder
    conv_layer4 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_layer3)
    conv_layer4 = BatchNormalization()(conv_layer4)
    up1 = UpSampling2D((2,2))(conv_layer4)
    conv_layer6 = Conv2D(24, (3, 3), activation='relu', padding='same')(up1) 
    conv_layer6 = BatchNormalization()(conv_layer6)
    up2 = UpSampling2D((2,2))(conv_layer6)
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(up2) 
    return decoded


# In[9]:


autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = Adam(), metrics=['accuracy','mse'])


# In[10]:


print(autoencoder.summary())
autoencoder_train = autoencoder.fit(training_data, training_data , epochs=15,batch_size=256,shuffle=True,validation_data = (testing_data, testing_data))


# In[11]:


# summarize history for loss
plt.plot(autoencoder_train.history['loss'])
plt.plot(autoencoder_train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[12]:


# summarize history for accuracy
plt.plot(autoencoder_train.history['accuracy'])
plt.plot(autoencoder_train.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[27]:


autoencoder_test = autoencoder.predict(testing_data)

print(autoencoder_test.shape)


# In[29]:


import random
ind=[random.randint(0,10000),random.randint(0,10000),random.randint(0,10000),random.randint(0,10000)]
plt.figure(figsize=(40, 4))
for i in range(len(ind)):
  # Display original
  ax = plt.subplot(3, 20, i + 1)
  plt.imshow(testing_data[ind[i]].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # Display reconstructed images
  ax = plt.subplot(3, 20, i + 1 + 20)
  plt.imshow(predicted_test[ind[i]].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:





# In[ ]:




