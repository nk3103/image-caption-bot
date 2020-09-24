#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import keras
import nltk
from nltk.corpus import stopwords
import string
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input,Dense,Dropout,Embedding,LSTM
from keras.layers.merge import add


# In[2]:


model = load_model('./model_weights/model_9.h5')
#model._make_predict_function()

# In[3]:


encoding_model =ResNet50(weights='imagenet',input_shape=(224,224,3))
model_res = Model(encoding_model.input,encoding_model.layers[-2].output)
#model_res._make_predict_function()

# In[4]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    return img


# In[12]:


def encode_img(img):
    img = preprocess_img(img)
    feature_vec = model_res.predict(img)
    feature_vec = feature_vec.reshape((1,feature_vec.shape[1]))
    #print(feature_vec.shape)
    return feature_vec


# In[13]:


#enc= encode_img('photo.jfif')


# In[14]:


#enc.shape
with open('./w2i.pkl','rb') as f:
    word_to_idx = pickle.load(f)
with open('./i2w.pkl','rb') as f:
    idx_to_word = pickle.load(f)

# In[16]:


def predict_caption(photo):

    in_text = "startseq"
    max_len=35
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')

        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)

        if word == "endseq":
            break

    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# In[18]:





# In[19]:

def caption(img):
    enc = encode_img(img)
    caption = predict_caption(enc)
    return caption


# In[ ]:
