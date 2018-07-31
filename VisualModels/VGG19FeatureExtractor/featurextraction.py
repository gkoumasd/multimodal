#Feature extraction from VGG19 pre-trained network

import os

#use a single GPU in Keras with TensorFlow
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(2017) 
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
import pandas as pd

dir_csv = '../../SplitedData.csv'
dir_img = '../../Flickr/'
global model


def main(Type):
  # load pre-trained model
  # Top is true, since we use the FC layer. 
  #If include_top=False, the model extracts a 7x7x512 matrix as output. Then I should flatten that maxtrix.
  model = VGG19(weights='imagenet', include_top=True)
  # display model layers
  model.summary()
  
  image_preprocess(Type)


def datasets(Type):
  #load csv data to dataframe
  df = pd.read_csv(dir_csv, sep='\t')

  #select data
  selected  = df[df['Type']==Type]
  selected  = selected.reset_index(drop=True)
  return selected


def image_preprocess(Type):
 img_list = []
 img_names = []
 df = datasets(Type)
 #counter = 0
 for i in range (0,len(df)):
  img_names.append(df.at[i,'Images'])
  print ('Processing ' , i , ' out of ' ,  len(df), ' images') 
  #image processing
  path = os.path.join(dir_img,df.at[i,'Images'].replace('\\', '/'))
  img = image.load_img(path, target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = preprocess_input(img)

  img_list.append(img)
  
  #counter +=1
  #if counter == 20:
   # break

 filename = open('img_test.txt', 'w')
 for img in img_names:
  filename.write("%s\n" % img)
 
 feature_extraction(img_list)
 
def feature_extraction(img_list):
  
  feature_list = []
  # load pre-trained model
  base_model = VGG19(weights='imagenet')
  # define model from base model for feature extraction from fc2 layer
  model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
  
  
  for i in range (0, len(img_list)):
   fc2_features = model.predict(img_list[i])
   print('Extracting', i , ' out of ', len(img_list))
   feature_list.append(fc2_features)
   
  
  pickle.dump(feature_list, open( "test_features.p", "wb" ) )
  b = pickle.load( open( "test_features.p", "rb" ) )
  print('DONE!')
  print('Total features:', len(b)) 
  print('Feature 0:', b[0])
  print('Lentgh of 0:', len(b[0][0]))


if __name__== "__main__":
 main('Test')

#select test dataset

#load pre-trained model
#base_model = VGG19(weights='imagenet')

# pre-process the image
#img = image.load_img('../../website/gkoumasd.github.io/me.jpg', target_size=(224, 224))
#img = image.img_to_array(img)
#img = np.expand_dims(img, axis=0)
#img = preprocess_input(img)

# define model from base model for feature extraction from fc2 layer
#model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

# obtain the outpur of fc2 layer
#fc2_features = model.predict(img)
#print ("Feature vector dimensions: ",fc2_features.shape)
#print(fc2_features)
