import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

'''
Main Function. Initilizes the feature vector. Reshapes training and testing. Stacks the feature vector with input vector.
'''
def Main():
  Features = np.array([[0]*3]*60000) #matrix to hold the values of the custom features
  Testing = np.array([[0]*3]*10000) #matrix to hold the values of the custom features

  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  train_images = train_images.reshape(60000, 28*28) #reshape train images
  test_images = test_images.reshape(10000, 28*28) #reshape test images
  test_images = np.concatenate((test_images,Testing), axis = 1) # make the testing matrix the same size as training (to prevent error: expected dense_23_input to have shape (787,) but got array with shape (784,) ) 

  train_images = train_images / 255.0 #normalize between 0-1
  test_images = test_images / 255.0 # normalize between 0-1
 
  for X in range(60000):
    Features[X][0] = calcHeight(train_images[X]) # calculate height of current image, and place it inside the appropriate index of the features matrix
    Features[X][1] = calcWidth(train_images[X]) # calculate width of current image, and place it inside the appropriate index of the features matrix
    #Features[X][2] = calcWhiteSpaces(train_images[X]) 

  train_images = np.concatenate((train_images,Features), axis = 1) #stack our feature vector onto our training vector 
  MachineLearning(train_images, train_labels, test_images, test_labels)



'''
Function that contains all the machine learning components 
'''
def MachineLearning(train_images, train_labels, test_images, test_labels):
  baseModel = keras.Sequential(keras.layers.Dense(10, activation=tf.nn.softmax))   # set up the softmax activation layer
  baseModel.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])  # compile the model 

  # train the model
  epochs = 4
  history = baseModel.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
  test_loss, test_acc = baseModel.evaluate(test_images, test_labels)

  print('Test accuracy:', test_acc)




'''
Function to calulate the normalized height of number contained within the passed image 
'''
def calcHeight(input_image): # The image after being flattened into a 1d vector

  topIndex = np.nonzero(input_image) # get indices of all non-zero elements
  topIndex = topIndex[0][0] # get the first one

  reversedImage = input_image[::-1] # reverse the image array 
 
  bottomIndex = np.nonzero(reversedImage) #get indices of all non-zero elements of the reversed array 
  bottomIndex = 783 - bottomIndex[0][0] # get the first one
  return (bottomIndex - topIndex)/784 # return the normalized value 




'''
Function to calulate the normalized width of number contained within the passed image 
'''
def calcWidth(input_image):
  reversedImage = input_image[::-1] # reverse the image array 
  break1 = False
  break2 = False
# for finding the lefmost index
  for i in range(0,28): # sudo rows 
    for X in range(0,784): # all pixels 
      if (X % 28 == i): # only check the pixels in the sudo row 
        if (input_image[X] != 0):
          leftIndex = X
          break1 = True
          break
      if(break1 == True):
        break
# for finding the rightmost index
    for X in range(783,0,-1):
      if (X % 28 == i):
        if (reversedImage[X] != 0):
          rightIndex = X
          break2 = True
          break
      if(break2 == True):
        break
  width = 28 - ((leftIndex%28) + (28-(784-rightIndex)%28))
  return width

'''
The base model had an accuracy of ~92%, while this model has an accuracy of ~87%.
The drop in accuracy means that the chosen features were not good ones. 
I beleive this is because many of the digits had very simliar width and height values. This did not give the 
system any more useful information to work with, and increased the amount of useless information it had to sort through.
'''
Main()

