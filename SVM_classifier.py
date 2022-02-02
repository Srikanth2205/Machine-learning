#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.layers import  Flatten, Dense
from tensorflow.keras.losses import MSE
from keras.datasets import fashion_mnist 
from tensorflow.keras.models import Model
import sklearn
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# In[2]:


## Seeding
np.random.seed(42)
tf.random.set_seed(42)


# In[3]:


(train_input, train_target), (test_input, test_target) = fashion_mnist.load_data()
train_input,test_input =train_input.astype('float32')/255.0, test_input.astype('float32')/255.0


# In[4]:


## Flattening the images.
train_input = np.reshape(train_input, (-1, 28 * 28 * 1))
test_input = np.reshape(test_input, (-1, 28 * 28 * 1))
print(train_input.shape, test_input.shape)


# In[5]:


input_img= keras.Input(shape=(784,))


# In[6]:


def encoder(input_img):
    encoder_layer1 = Dense(units=128, activation='relu')(input_img)
    encoder_layer2 = Dense(units=64, activation='relu')(encoder_layer1)
    encoder_layer3 = Dense(units=32, activation='relu')(encoder_layer2)
    return encoder_layer3


# In[7]:


def decoder(encoder_layer3):
    decoder_layer1 = Dense(units=64, activation='relu')(encoder_layer3)
    decoder_layer2 = Dense(units=128, activation='relu')(decoder_layer1)
    decoder_layer3 = Dense(units=784, activation='relu')(decoder_layer2)
    return decoder_layer3
    


# In[8]:


encoder = tf.keras.Model(input_img, encoder(input_img))
autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy','mse'])


# In[9]:


autoencoder.summary()


# In[10]:


autoencoder_train = autoencoder.fit(train_input, train_input , epochs=30,batch_size=256,shuffle=True,validation_data = (test_input, test_input))


# In[11]:


#Defining the train set and labels for SVM classifier
svm_training_input = encoder.predict(train_input)
svm_testing_input = encoder.predict(test_input)
svm_training_target = train_target
svm_testing_target = test_target


# In[12]:


svm_training_input.shape


# In[13]:


svc_linear = LinearSVC()
svc_linear.fit(svm_training_input, svm_training_target)


# In[14]:


target_pred_svc_autoencoder = svc_linear.predict(svm_testing_input)


# In[15]:


class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

confusion_matrix_SVC = metrics.confusion_matrix(test_target, target_pred_svc_autoencoder)
print(confusion_matrix_SVC)

plt.figure()
plt.title("Confusion Matrix")
plt.imshow(confusion_matrix_SVC, interpolation = 'nearest', cmap=plt.cm.Reds)
marker = np.arange(len(class_labels))
plt.xticks(marker, class_labels, rotation=90)
plt.yticks(marker, class_labels)
plt.show()

accuracy = metrics.accuracy_score(test_target, target_pred_svc_autoencoder)
print(f"SVM Model's testing accuracy: {accuracy}")  

print(metrics.classification_report(test_target, target_pred_svc_autoencoder, target_names=class_labels))


# In[53]:


target_pred_svc_autoencoder_1 = svc_linear.predict(svm_training_input)


# In[54]:


accuracy4 = metrics.accuracy_score(train_target, target_pred_svc_autoencoder_1)
print(f"SVM Model's training accuracy: {accuracy4}")  


# In[16]:


from sklearn.model_selection import GridSearchCV
  
# defining parameter range
param_grid = {'C': [0.1, 1], 
              'gamma': [0.1, 1],
              'kernel': ['linear']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(svm_training_input, svm_training_target)


# In[17]:


grid_predictions = grid.predict(svm_testing_input)


# In[18]:


Class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

confusion_matrix_grid = metrics.confusion_matrix(test_target, target_pred_svc_autoencoder)
print(confusion_matrix_grid)

plt.figure()
plt.title("Confusion Matrix")
plt.imshow(confusion_matrix_grid, interpolation = 'nearest', cmap=plt.cm.Blues)
marker2 = np.arange(len(class_labels))
plt.xticks(marker, class_labels, rotation=90)
plt.yticks(marker, class_labels)
plt.show()

accuracy2 = metrics.accuracy_score(test_target, grid_predictions)
print(f"SVM Model's accuracy_Grid_search: {accuracy2}")  

print(metrics.classification_report(test_target, target_pred_svc_autoencoder, target_names=class_labels))


# In[19]:


svc_rbf = SVC(C=1, kernel='rbf', gamma="auto")
svc_rbf.fit(svm_training_input, svm_training_target)


# In[20]:


target_pred_svc_rbf_autoencoder = svc_rbf.predict(svm_testing_input)


# In[21]:


class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

confusion_matrix_SVC_rbf = metrics.confusion_matrix(test_target, target_pred_svc_autoencoder)
print(confusion_matrix_SVC)

plt.figure()
plt.title("Confusion Matrix")
plt.imshow(confusion_matrix_SVC_rbf, interpolation = 'nearest', cmap=plt.cm.Oranges)
marker3 = np.arange(len(class_labels))
plt.xticks(marker, class_labels, rotation=90)
plt.yticks(marker, class_labels)
plt.show()

accuracy3 = metrics.accuracy_score(test_target, target_pred_svc_rbf_autoencoder)
print(f"Kernel based  SVM Model's testing accuracy: {accuracy3}")  

print(metrics.classification_report(test_target, target_pred_svc_autoencoder, target_names=class_labels))


# In[55]:


target_pred_svc_autoencoder_2 = svc_rbf.predict(svm_training_input)


# In[56]:


accuracy5 = metrics.accuracy_score(train_target, target_pred_svc_autoencoder_2)
print(f"Kernel based SVM Model's training accuracy: {accuracy5}")


# In[52]:


from matplotlib.pyplot import figure

fig = plt.figure(figsize=(8, 5), dpi=80)
ax = fig.add_subplot(111)
# x axis values
x = ['linear SVM Model accuracy ', 'grid search based SVM Model accuracy', 'Kernel based SVM Model accuracy']
# corresponding y axis values
y = [accuracy*100,accuracy2*100,accuracy3*100]

#plt.barh(x, y)

#for index, value in enumerate(y):
 #   plt.text(value, index, str(value))
plt.plot(range(len(x)), y, 'go-') # Plotting data
plt.xticks(range(len(x)), x) # Redefining x-axis labels

for i, v in enumerate(y):
    ax.text(i, v+3, "%f" %v, ha="center")
plt.ylim(50, 107)
# naming the x axis
plt.xlabel('SVM Model ')
# naming the y axis
plt.ylabel(' Accuracy %')
  
# giving a title to my graph
plt.title('Accuracy comparison of SVM Models')

plt.show(block=True)


# In[ ]:





# In[ ]:




