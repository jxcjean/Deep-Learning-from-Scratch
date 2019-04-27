# Intel MKL FATAL ERROR: Cannot load mkl_intel_thread.dll.
#### https://blog.csdn.net/zhangpeterx/article/details/84872125

def helloWorld(object):
    print('*********Start ' + object + '*********')

helloWorld("Deep Learning")

#import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

np_a=np.array([[1,2],[3,4]])
np_b=np.array([[1,3],[2,4]])
print(np_a+np_b)
print(np_a[1])
np_a=np_a.flatten()
print(np_a)

x=np.arange(0,6,0.1) #以0.1位单位，生成0-6的数据
y=np.sin(x)

print(y)
plt.plot(x,y)
plt.show()

img=imread('dataset/x1hm.png') # read a image
plt.imshow(img)
plt.show()
