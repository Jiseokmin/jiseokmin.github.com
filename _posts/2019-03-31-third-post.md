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
# example of distance function between two vectors x and y
#
def distance(x, y):

    d = (x - y) ** 2
    s = np.sum(d)
    # r = np.sqrt(s)

    return(s)

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





import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


train_start=0
train_end=500

train_file= "mnist_train.csv"
temp_train=np.array(pd.read_csv(train_file).ix[train_start-1:train_end-1].as_matrix(),dtype='uint8')


# display 'average' digits
fig=plt.figure(figsize=(10,3))
for i in range(10):
   
    train_ave=np.mean(temp_train[:,1:][temp_train[:,0]== i],axis=0)
    ax=fig.add_subplot(2,5,i+1)
    ax.set_axis_off()
    a=np.copy(train_ave)
    #a=np.reshape(a,(28,28))
    ax.imshow(a.reshape((28,28)), cmap='gray', interpolation='nearest', clim=(0,300))

plt.show()



K = 7   #First K = 7

import random
%matplotlib inline
from matplotlib import pyplot as plt

centroid = random.sample(range(0,10), K)  ## first,we should get random number for initial value of centeroid
centroid.sort()
centroid2 = copy.deepcopy(centroid)

fig=plt.figure(figsize=(9,3))
for i in range(K):
   
    train_ave=np.mean(temp_train[:,1:][temp_train[:,0]== centroid [i]],axis=0)
    ax=fig.add_subplot(1,K,i+1)
    ax.set_axis_off()
    a=np.copy(train_ave)
   
    ax.imshow(a.reshape((28,28)), cmap='gray', interpolation='nearest', clim=(0,300))

plt.show()



import copy


cnt = [0*K for i in range(K)]     #for count each label's
itera = 0
total = 0
temp = [0*K for i in range(K)]
ssum = [0*K for i in range(K)]
ssum2 = [[0 for cols in range(20)]for rows in range(20)]

cluster = [[0 for cols in range(500)]for rows in range(K)]
accuracy =[]
accu_cnt = [0,0,0,0,0,0,0,0,0,0]



while True:
 
    accu_cnt = [0,0,0,0,0,0,0,0,0,0]
    
    cluster = [[0 for cols in range(500)]for rows in range(K)]
    cnt = [0*K for i in range(K)]
    temp = [0*K for i in range(K)]
    ssum = [0*K for i in range(K)]
    
    for i in range(500):
        for j in range(K):
       
            temp[j] = abs(list_label[i]-centroid [j])
        
       
        cluster[temp.index(min(temp))][cnt[temp.index(min(temp))]] = list_label[i]
        cnt[temp.index(min(temp))] += 1
    
    
    for j in range(K):
        for k in range(500):
            total = total + cluster[j][k]
                
        ssum[j] = total
        total = 0
        
    
    for k in range(K):
        ssum2[itera][k] = ssum[k]
                
   
    
    for j in range(K):

        centroid[j] = round(ssum[j]/cnt[j])    #compute next centroid
        
    centroid.sort()
    centroid = list(set(centroid))
    K = len(centroid)
   
    fig=plt.figure(figsize=(9,3))
    
    for i in range(K):
   
        train_ave=np.mean(temp_train[:,1:][temp_train[:,0]== centroid [i]],axis=0)
        ax=fig.add_subplot(1,K,i+1)
        ax.set_axis_off()
        a=np.copy(train_ave)
      
        ax.imshow(a.reshape((28,28)), cmap='gray', interpolation='nearest', clim=(0,300))

    for i in range(K):
        print(centroid[i])
        
        
    for i in range(K):
        for j in range(cnt[i]):
            if(cluster[i][j] == i):
                accu_cnt[i] += 1
           
                
   
    accuracy.append(sum(accu_cnt)/sum(cnt))    #compute accuracy
    
    
    if(centroid == centroid2):
        itera += 1
        break
    else:
        centroid2 = copy.deepcopy(centroid)
        itera += 1
    
        
   
for i in range(1,20):
    for j in range(20):
        plt.scatter(i, ssum2[i][j])
    
    
plt.show()


for i in range(0,itera):
        plt.scatter(i+1, accuracy[i])
    
    
plt.show()
    
    
    
  
