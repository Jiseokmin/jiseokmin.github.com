<img width="575" alt="캡처" src="https://user-images.githubusercontent.com/28971360/54863560-8a728800-4d8d-11e9-8f9b-fc1b41945163.PNG">


import matplotlib.pyplot as plt
import numpy as np

file_data		= "mnist_train.csv"
handle_file	= open(file_data, "r")
data        		= handle_file.readlines()
handle_file.close()

size_row	= 28    # height of the image
size_col  	= 28    # width of the image

num_image	= len(data)
count       	= 0     # count for the number of images

#
# normalize the values of the input data to be [0, 1]
#
def normalize(data):

    data_normalized = (data - min(data)) / (max(data) - min(data))

    return(data_normalized)


#
# make a matrix each column of which represents an images in a vector form 
#
list_image  = np.empty((size_row * size_col, num_image), dtype=float)
list_label  = np.empty(num_image, dtype=int)

for line in data:

    line_data   = line.split(',')
    label       = line_data[0]
    im_vector   = np.asfarray(line_data[1:])
    im_vector   = normalize(im_vector)

    list_label[count]       = label
    list_image[:, count]    = im_vector    

    count += 1

# 
# plot first 100 images out of 10,000 with their labels
# 
f1 = plt.figure(1)

for i in range(100):

    label       = list_label[i]
    im_vector   = list_image[:, i]
    im_matrix   = im_vector.reshape((size_row, size_col))

    plt.subplot(10, 10, i+1)
    plt.title(label)
    plt.imshow(im_matrix, cmap='Greys', interpolation='None')

    frame   = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

plt.show()


<img width="291" alt="캡처" src="https://user-images.githubusercontent.com/28971360/54863567-afff9180-4d8d-11e9-8426-9c631a5b48b0.PNG">



<img width="229" alt="캡처" src="https://user-images.githubusercontent.com/28971360/54863576-c279cb00-4d8d-11e9-8525-9aeeb31fe87c.PNG">


import matplotlib.pyplot as plt
import numpy as np
import math

file_data		= "mnist_train.csv"
handle_file	= open(file_data, "r")
data        		= handle_file.readlines()
handle_file.close()

size_row	= 28    # height of the image
size_col  	= 28    # width of the image

num_image	= len(data)
count       	= 0     # count for the number of images

d = 0  #for storing distance

#
# using count numbers of 'number' and calculate average of each number
#
count_zero  = []
count_one  = []
count_two  = []
count_three  = []
count_four  = []
count_five  = []
count_six  = []
count_seven  = []
count_eight  = []
count_nine  = []


avg_zero  = 0
avg_one  = 0
avg_two  = 0
avg_three  = 0
avg_four  = 0
avg_five  = 0
avg_six  = 0
avg_seven  = 0
avg_eight  = 0
avg_nine  = 0

#######################




#
# normalize the values of the input data to be [0, 1]
#
def normalize(data):

    data_normalized = (data - min(data)) / (max(data) - min(data))

    return(data_normalized)

#
# Compute Euclidean distance
#
def distance(x, y):
 
    
    d = (x - y) ** 2
    s = np.sum(d)
    r = np.sqrt(s)
    
    
    return r

#
# make a matrix each column of which represents an images in a vector form 
#
list_image  = np.empty((size_row * size_col, num_image), dtype=float)
list_label  = np.empty(num_image, dtype=int)

for line in data:

    line_data   = line.split(',')
    label       = line_data[0]
    im_vector   = np.asfarray(line_data[1:])
    im_vector   = normalize(im_vector)
    

    list_label[count]       = label
    list_image[:, count]    = im_vector    

    count += 1



# 
# plot first 100 images out of 10,000 with their labels
# 
f1 = plt.figure(1)

for i in range(100):

    label       = list_label[i]
    im_vector   = list_image[:, i]
   
    im_matrix   = im_vector.reshape((size_row, size_col))
    
    d = distance(label,im_vector)    # store distance

    if list_label[i] == 0:
        count_zero.append(d)
    elif list_label[i] == 1:
         count_one.append(d)
    elif list_label[i] == 2:
         count_two.append(d)
    elif list_label[i] == 3:
         count_three.append(d)
    elif list_label[i] == 4:
         count_four.append(d)
    elif list_label[i] == 5:
         count_five.append(d)
    elif list_label[i] == 6:  
         count_six.append(d)
    elif list_label[i] == 7:  
         count_seven.append(d)
    elif list_label[i] == 8:
         count_eight.append(d)
    elif list_label[i] == 9:
         count_nine.append(d)
    
#
# calculate average
#

avg_zero  =  sum(count_zero, 0.0)/len(count_zero)
avg_one  = sum(count_one, 0.0)/len(count_one)
avg_two  = sum(count_two, 0.0)/len(count_two)
avg_three  = sum(count_three, 0.0)/len(count_three)
avg_four  = sum(count_four, 0.0)/len(count_four)
avg_five  = sum(count_five, 0.0)/len(count_five)
avg_six  = sum(count_six, 0.0)/len(count_six)
avg_seven  = sum(count_seven, 0.0)/len(count_seven)
avg_eight  = sum(count_eight, 0.0)/len(count_eight)
avg_nine  = sum(count_nine, 0.0)/len(count_nine)
    
print(" ")    
print("Zero_Average :",avg_zero)
print("One_Average :" ,avg_one)
print("Two_Average :" ,avg_two)
print("Three_Average :", avg_three)
print("Four_Average :" ,avg_four)
print("Five_Average :", avg_five)
print("Six_Average :" ,avg_six)
print("Seven_Average :", avg_seven)
print("Eight_Average :" ,avg_eight)
print("Nine_Average :" ,avg_nine)



<img width="212" alt="캡처" src="https://user-images.githubusercontent.com/28971360/54863584-dc1b1280-4d8d-11e9-8bf8-d5acab7876e1.PNG">


<img width="399" alt="캡처" src="https://user-images.githubusercontent.com/28971360/54863590-ea692e80-4d8d-11e9-949a-397a5440726b.PNG">


import matplotlib.pyplot as plt
import numpy as np

file_data		= "mnist_train.csv"
handle_file	= open(file_data, "r")
data        		= handle_file.readlines()
handle_file.close()

size_row	= 28    # height of the image
size_col  	= 28    # width of the image

num_image	= len(data)
count       	= 0     # count for the number of images


zero  = 1
one  = 2
two  = 3
three  = 4
four  = 5
five  = 6
six  = 7
seven  = 8
eight  = 9
nine  = 10
nothing = 11      #'nothing' is Uncategorized number.

#
# normalize the values of the input data to be [0, 1]
#
def normalize(data):

    data_normalized = (data - min(data)) / (max(data) - min(data))

    return(data_normalized)

#
# Compute Euclidean distance
#
def distance(x, y):
 
    
    d = (x - y) ** 2
    s = np.sum(d)
    r = np.sqrt(s)
    
    
    return r
#
# make a matrix each column of which represents an images in a vector form 
#
list_image  = np.empty((size_row * size_col, num_image), dtype=float)
list_label  = np.empty(num_image, dtype=int)

for line in data:

    line_data   = line.split(',')
    label       = line_data[0]
    im_vector   = np.asfarray(line_data[1:])
    im_vector   = normalize(im_vector)

    list_label[count]       = label
    list_image[:, count]    = im_vector    

    count += 1

# 
# plot first 100 images out of 10,000 with their labels
# 
f1 = plt.figure(1)    

for i in range(100):

    label       = list_label[i]
    im_vector   = list_image[:, i]
    im_matrix   = im_vector.reshape((size_row, size_col))

    d = distance(label,im_vector)
        
    
#using the euclidean distance and the average of each number and classify numvers

    if d > 6 and d < 16:
        plt.subplot(14,11,zero)
        plt.imshow(im_matrix, interpolation='None')
        
      
        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        zero += 11
        
    elif d > 21 and d < 31:
        plt.subplot(14,11,one)
        plt.imshow(im_matrix, interpolation='None')
        
      
        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        one += 11
        
    elif d > 47 and d < 57:
        plt.subplot(14,11,two)
        plt.imshow(im_matrix, interpolation='None')
        
      
        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        two += 11 
    
    elif d > 73 and d < 85:
        plt.subplot(14,11,three)
        plt.imshow(im_matrix, interpolation='None')
        
      
        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        three += 11
        
    elif d > 104 and d < 114:
        plt.subplot(14,11,four)
        plt.imshow(im_matrix, interpolation='None')
        
      
        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        four += 11  
    
    elif d > 132 and d < 142:
        plt.subplot(14,11,five)
        plt.imshow(im_matrix, interpolation='None')
        
      
        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        five += 11
        
    elif d > 159 and d < 169:
        plt.subplot(14,11,six)
        plt.imshow(im_matrix, interpolation='None')
        
      
        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        six += 11 
    
    elif d > 188 and d < 198:
        plt.subplot(14,11,seven)
        plt.imshow(im_matrix, interpolation='None')
        
      
        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        seven += 11
        
    elif d > 215 and d < 225:
        plt.subplot(14,11,eight)
        plt.imshow(im_matrix, interpolation='None')
        
      
        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        eight += 11  
        
    elif d > 244 and d < 254:
        plt.subplot(14,11,nine)
        plt.imshow(im_matrix, interpolation='None')
        
      
        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        nine += 11
        
    else :
        plt.subplot(20,11,nothing)
        plt.imshow(im_matrix, interpolation='None')
        
      
        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        nothing += 11
        
    
plt.show()



<img width="274" alt="캡처" src="https://user-images.githubusercontent.com/28971360/54863595-01a81c00-4d8e-11e9-8c02-47b2270246f3.PNG">



<img width="461" alt="캡처" src="https://user-images.githubusercontent.com/28971360/54863598-108ece80-4d8e-11e9-9808-32cac6a54dbf.PNG">
