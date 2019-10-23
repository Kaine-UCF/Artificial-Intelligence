'''
Brian Kaine Margretta
10/22/19
CAP4630 Artificial Intelligence
Homework #1 Part C
'''

from keras.datasets import mnist
import numpy as np
import keras as K
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = mnist.load_data() #load data
num_images = len(train_images)  # get total number of images
num_digits = 10 #0-9

width = train_images[0].shape[0] # get width, should always be 28 for minst
height = train_images[0].shape[1] # get height, should always be 28 for minst

zerod_image = np.zeros([width,height],dtype = int) #initilize 28x28 array to 0

average_image = [zerod_image]*10 # create an array of 10 new "blank" images where each index holds its respective #'s average 
o_count = 0

for n in range(num_images): #iterate through all images
  for i in range(num_digits): #iterate through all digits
    if(train_labels[n] == i): # only add images together if theyre the same digit
      average_image[i] = np.add(average_image[i],train_images[n])


for x in range(num_digits): #display the average of all digits
  plt.imshow(average_image[x])
  plt.show()
