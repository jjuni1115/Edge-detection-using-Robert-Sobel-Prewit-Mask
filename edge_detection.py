from PIL import Image
import numpy as np
from matplotlib import pyplot as plot
import random
import math

SNR=8.0
#mask value
sobel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
roberts_x=np.array([[1,0,0],[0,-1,0],[0,0,0]])
roberts_y=np.array([[0,-1,0],[1,0,0],[0,0,0]])
prewit_x=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewit_y=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
stochastic_mask_x=np.array([[0.267,0.364,0,-0.364,-0.267],[0.373,0.562,0,-0.562,-0.373],[0.463,1.000,0,-1.000,-0.463],[0.373,0.562,0,-0.562,-0.373],[0.267,0.364,0,-0.364,-0.267]])
stochastic_mask_y=np.array([[0.267,0.373,0.463,0.373,0.267],[0.364,0.562,1.000,0.562,0.364],[0,0,0,0,0],[-0.364,-0.562,-1.000,-0.562,-0.364],[-0.267,-0.373,-0.463,-0.373,-0.267]])
#num of edge
roberts_edge=0
roberts_edge2=0
sobel_edge=0
sobel_edge2=0
prewit_edge=0
prewit_edge2=0
Pe=[]


def variance(input_image,n):                  #variance function
    MN=len(input_image)*len(input_image[0])
    sum_x=0.0
    sum_x2=0.0
    for i in range(len(input_image)):
        for k in input_image[i]:
            sum_x2+=k**2
            sum_x+=k
    sigma_2=(sum_x2/float(MN))-((sum_x/float(MN))**2)
    return sigma_2

def AddGaussianNoise (input_img, noise_img, sigma):  #add noise function
    for i in range(0,512):
        for j in range(0,512):
            s=input_img[i][j]+Gaussian(sigma)
            if s > 255:
                s = 255
            elif s < 0:
                s = 0
            noise_img[i][j]=s
def Gaussian(sd):           #noise function
    ready=0
    gstore=0.0
    v1=0.0
    v2=0.0
    r=0.0
    fac=0.0
    gaus=0.0
    r1=0
    r2=0
    flag=0
    if ready==0:
        while 1:
            if flag==0:
                r1=random.randint(0,32767)
                r2=random.randint(0,32767)
                v1=2*(float(r1)/float(32767)-0.5)
                v2=2*(float(r2)/float(32767)-0.5)
                r=v1*v1+v2*v2
                flag=1
            elif flag==1 and r>1.0:
                r1 = random.randint(0, 32767)
                r2 = random.randint(0, 32767)
                v1 = 2 * (float(r1)/float(32767)-0.5)
                v2 = 2 * (float(r2)/float(32767)-0.5)
                r = v1 * v1 + v2 * v2
            else:
                break
        fac=float(math.sqrt(float(-2*math.log(r)/r)))
        gstore=v1*fac
        gaus=v2*fac
        ready=1
    else:
        ready=0
        gaus=gstore

    return gaus*sd


def threshold(pixel):      #threshold and count edge funtion
    edge_num=0
    for i in range(0,512):
        for j in range(0,512):
            if pixel[i][j]>=150:
                pixel[i][j]=255
                edge_num+=1
            else:
                pixel[i][j]=0
    return (pixel,edge_num)

def edge_detection(pixel,mask_x,mask_y): #edge dectection function (parameter: mask x, mask y, image pixel)
    Gx = 0
    Gy = 0
    new_pixel = np.ones((512, 512), dtype=int)
    for i in range(0, 512):
        for j in range(0, 512):
            if i == 0:
                new_pixel[i][j] = 0
            elif j == 0:
                new_pixel[i][j] = 0
            elif i == 511:
                new_pixel[i][j] = 0
            elif j == 511:
                new_pixel[i][j] = 0
            else:
                for k in range(0, 3):
                    for m in range(0, 3):
                        Gx = Gx + mask_x[k][m] * pixel[i + k - 1][j + m - 1]
                        Gy = Gy + mask_y[k][m] * pixel[i + k - 1][j + m - 1]
                new_pixel[i][j] = math.sqrt(Gx * Gx + Gy * Gy)
                Gx = 0
                Gy = 0
    return new_pixel

def stochastic_operation(pixel):     #stochastic edge dectection function
    Gx = 0
    Gy = 0
    new_pixel = np.ones((512, 512), dtype=int)
    for i in range(3, 509):
        for j in range(3, 509):
            if i == 0:
                new_pixel[i][j] = 0
            elif j == 0:
                new_pixel[i][j] = 0
            elif i == 511:
                new_pixel[i][j] = 0
            elif j == 511:
                new_pixel[i][j] = 0
            else:
                for k in range(0, 5):
                    for m in range(0, 5):
                        Gx = Gx + stochastic_mask_x[k][m] * pixel[i + k - 1][j + m - 1]
                        Gy = Gy + stochastic_mask_y[k][m] * pixel[i + k - 1][j + m - 1]
                new_pixel[i][j] = math.sqrt(Gx * Gx + Gy * Gy)
                Gx = 0
                Gy = 0
    return new_pixel

def Error_rate(pixel1,pixel2):    #error rate function
    x=0
    for i in range(0,512):
        for j in range(0,512):
            if pixel1[i][j] != pixel2[i][j]:
                x += 1
    return x

#load image
im=Image.open('lena_bmp_512X512_new.bmp')
pixel=np.array(im)
noise_pixel = np.ones((512, 512), dtype=int)

#Add noise on the image
var=variance(pixel,8.0)
stddev_noise=float(math.sqrt(var/math.pow(10.0,SNR/10)))
AddGaussianNoise(pixel,noise_pixel,stddev_noise)

#setting figure
fig=plot.figure(figsize=(5,5))
rows=1
cols=2

#print image and noise image
im1=fig.add_subplot(rows,cols,1)
im1.imshow(pixel,cmap='gray')
im1.set_title('Original Image')
im1.axis('off')
im2=fig.add_subplot(rows,cols,2)
im2.imshow(noise_pixel,cmap='gray')
im2.set_title('Noise Image')
im2.axis('off')
plot.tight_layout()
plot.show()

#roberts edge dection
roberts_pixel = np.ones((512, 512), dtype=int)
roberts_pixel=edge_detection(pixel,roberts_x,roberts_y)
(roberts_pixel,roberts_edge)=threshold(roberts_pixel)
print('Roberts edge count(normal image):',roberts_edge)

#noise image roberts edge dection
roberts_pixel2 = np.ones((512, 512), dtype=int)
roberts_pixel2=edge_detection(noise_pixel,roberts_x,roberts_y)
(roberts_pixel2,roberts_edge2)=threshold(roberts_pixel2)
e=Error_rate(roberts_pixel,roberts_pixel2)
print('Roberts edge number(noise image):',roberts_edge2)
print('number of missed or new edge pixels after adding noise :',e)
print('roberts Pe:',e/roberts_edge)
Pe.append(e/roberts_edge)

#checking edge detection(roberts)
fig=plot.figure(figsize=(5,5))
rows=1
cols=2
im1=fig.add_subplot(rows,cols,1)
im1.imshow(roberts_pixel,cmap='gray')
im1.set_title('Original Image(Roberts)')
im1.axis('off')
im2=fig.add_subplot(rows,cols,2)
im2.imshow(roberts_pixel2,cmap='gray')
im2.set_title('Noise Image(Roberts)')
im2.axis('off')
plot.tight_layout()
plot.show()

#sobel edge detection
sobel_pixel=np.ones((512,512),dtype=int)
sobel_pixel=edge_detection(pixel,sobel_x,sobel_y)
(sobel_pixel,sobel_edge)=threshold(sobel_pixel)
print('Sobel edge count(normal image):',sobel_edge)

#noise image sobel edge dection
sobel_pixel2=np.ones((512,512),dtype=int)
sobel_pixel2=edge_detection(noise_pixel,sobel_x,sobel_y)
(sobel_pixel2,sobel_edge2)=threshold(sobel_pixel2)
e=Error_rate(sobel_pixel,sobel_pixel2)
print('Sobel edge count(noise image):',sobel_edge2)
print('number of missed or new edge pixels after adding noise:',e)
print('sobel Pe:',e/sobel_edge)
Pe.append(e/sobel_edge)

#checking edge detection(sobel)
fig=plot.figure(figsize=(5,5))
rows=1
cols=2
im1=fig.add_subplot(rows,cols,1)
im1.imshow(sobel_pixel,cmap='gray')
im1.set_title('Original Image(Sobel)')
im1.axis('off')
im2=fig.add_subplot(rows,cols,2)
im2.imshow(sobel_pixel2,cmap='gray')
im2.set_title('Noise Image(Sobel)')
im2.axis('off')
plot.tight_layout()
plot.show()

#prewit edge detection
prewit_pixel=np.ones((512,512),dtype=int)
prewit_pixel=edge_detection(pixel,prewit_x,prewit_y)
(prewit_pixel,prewit_edge)=threshold(prewit_pixel)
print('prewit edge count(normal image):',prewit_edge)

#noise image prewit edge dection
prewit_pixel2=np.ones((512,512),dtype=int)
prewit_pixel2=edge_detection(noise_pixel,prewit_x,prewit_y)
(prewit_pixel2,prewit_edge2)=threshold(prewit_pixel2)
e=Error_rate(prewit_pixel,prewit_pixel2)
print('prewit edge count(noise image):',prewit_edge2)
print('number of missed or new edge pixels after adding noise:',e)
print('prewit Pe:',e/prewit_edge)
Pe.append(e/prewit_edge)

#checking edge detection(prewit)
fig=plot.figure(figsize=(5,5))
rows=1
cols=2
im1=fig.add_subplot(rows,cols,1)
im1.imshow(sobel_pixel,cmap='gray')
im1.set_title('Original Image(prewit)')
im1.axis('off')
im2=fig.add_subplot(rows,cols,2)
im2.imshow(sobel_pixel2,cmap='gray')
im2.set_title('Noise Image(prewit)')
im2.axis('off')
plot.tight_layout()
plot.show()

#stochastic edge detection
stochastic_pixel=np.ones((512,512),dtype=int)
stochastic_pixel=stochastic_operation(pixel)
(stochastic_pixel,st_edge)=threshold(stochastic_pixel)
print('stochastic edge count(normal image)',st_edge)

#noise image stochastic edge dection
stochastic_pixel2=np.ones((512,512),dtype=int)
stochastic_pixel2=stochastic_operation(noise_pixel)
(stochastic_pixel2,st_edge2)=threshold(stochastic_pixel2)
e=Error_rate(stochastic_pixel,stochastic_pixel2)
print('stochastic edge count(noise image):',st_edge2)
print('number of missed or new edge pixels after adding noise:',e)
print('stochastic Pe:',e/st_edge)
Pe.append(e/st_edge)

#checking edge detection(stochastic)
fig=plot.figure(figsize=(5,5))
rows=1
cols=2
im1=fig.add_subplot(rows,cols,1)
im1.imshow(stochastic_pixel,cmap='gray')
im1.set_title('Original Image(stochastic)')
im1.axis('off')
im2=fig.add_subplot(rows,cols,2)
im2.imshow(stochastic_pixel2,cmap='gray')
im2.set_title('Noise Image(stochastic)')
im2.axis('off')
plot.tight_layout()
plot.show()

#print pe each mask
x=['roberts','Sobel','prewit','Stochastic']
plot.ylabel('Pe')
plot.title('Edge dectection error rate')
plot.bar(x,Pe)
plot.show()







