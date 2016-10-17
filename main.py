#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import ndimage

def fourier_image(filename):
    image = Image.open(filename)
    #image = Image.open("lena-cropped-bw.jpg")
    image_gray = image.convert('L') # convert to grayscale
    image_npa = np.asarray(image_gray)

    image_freq = np.fft.fftshift(np.fft.fft2(image_npa))
    
    valsx = np.linspace(0,image_npa.shape[1],image_npa.shape[1]) - image_npa.shape[1]/2
    valsy = np.linspace(0,image_npa.shape[0],image_npa.shape[0]) - image_npa.shape[0]/2
    xx, yy = np.meshgrid(valsx,valsy)
    mask = np.sqrt(xx**2 + yy**2)
    image_freq[mask>10] = 0

    image_restore = np.fft.ifft2(image_freq)
    
    plt.figure("original"),plt.imshow(image_gray, cmap='gray', interpolation='none')
    plt.figure("frequency"),plt.imshow(np.log(np.abs(image_freq)), cmap='gray', interpolation='none'), plt.colorbar()
    plt.figure("restore"),plt.imshow(np.abs(image_restore), cmap='gray', interpolation='none')
    plt.show()


def clickable_fourier():
    size = 64
    image_freq = np.zeros((size, size), dtype='complex')

    image_restore = abs(np.fft.ifft2(image_freq))

    f = plt.figure("frequency"),plt.imshow(np.log(image_freq.real), cmap='gray', interpolation='none'), plt.colorbar()
    plt.axvline(x=size/2, color='white')
    plt.axhline(y=size/2, color='white')
    plt.figure("restored"),plt.imshow(image_restore, cmap='gray', interpolation='none'), plt.colorbar()

    def onclick(event):
        print event
        ix, iy = round(event.xdata), round(event.ydata)
        if event.button==1:
            image_freq[iy,ix] = (1-image_freq[iy,ix].real + image_freq[iy,ix].imag)  
        else:
            image_freq[iy,ix] = (image_freq[iy,ix].real + 1j*(1-image_freq[iy,ix].imag))

        image_restore = abs(np.fft.ifft2(image_freq))
        plt.figure("frequency"),plt.imshow(np.abs(image_freq), cmap='gray', interpolation='none')
        plt.draw()
        plt.figure("restored"),plt.imshow(image_restore, cmap='gray', interpolation='none')
        plt.draw()

    f[0].canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def noise2d(size=512, exponent=-2):
    """
    exponent: 2 brownian noise, 1 pink noise 
    """
    freq = np.random.rand(size,size)-0.5 + 1j*(np.random.rand(size,size)-0.5)

    vals = np.linspace(0,1,size)-.5
    xx, yy = np.meshgrid(vals,vals)
    filter = np.sqrt(xx**2 + yy**2)**exponent

    freq *= filter
    result = abs(np.fft.ifft2(freq))

    #plt.figure("frequency"),plt.imshow(np.log(abs(freq)), cmap='gray'), plt.colorbar()
    #plt.figure("filter"),plt.imshow(np.log(filter), cmap='gray'), plt.colorbar()
    #plt.figure("result"),plt.imshow(result, cmap='gray'), plt.colorbar()
    #plt.show()
    result -= np.min(result)
    result /= np.max(result)
    return result


def normalize(m):
    m -= np.min(m)
    m /= np.max(m)
    return m

def mutate(size=128, exponent=-2, freq=None, amplify_edges=False):
    if freq is None:
        freq = np.random.rand(size,size)-0.5 + 1j*(np.random.rand(size,size)-0.5)
        vals = np.linspace(0,1,size)-.5
        xx, yy = np.meshgrid(vals,vals)
        filter = np.sqrt(xx**2 + yy**2)**exponent
        freq *= filter
    else:
        freq_new = np.random.rand(size,size)-0.5 + 1j*(np.random.rand(size,size)-0.5)
        vals = np.linspace(0,1,size)-.5
        xx, yy = np.meshgrid(vals,vals)
        filter = np.sqrt(xx**2 + yy**2)**exponent
        freq_new *= filter
        freq = freq*0.98 + 0.02 * freq_new

    result = abs(np.fft.ifft2(freq))

    if amplify_edges:
        sx = ndimage.sobel(result, axis=0, mode='wrap')
        sy = ndimage.sobel(result, axis=1, mode='wrap')
        sob = np.hypot(sx, sy)
        sob = normalize(sob)
        result += sob

    result = normalize(result)

    return result, freq


if __name__ == "__main__":

    if True:
        fourier_image("lena-cropped-bw.jpg")

    if False:
        clickable_fourier()

    if False:
        img = noise2d()
        plt.imshow(img, cmap='gray')
        plt.show()

    if False:
        a,b,c = noise2d(),  noise2d(), noise2d()
        img = np.dstack((a,b,c))
        plt.imshow(img)
        plt.show()

    if False:
        result1, freq1 = mutate()
        result2, freq2 = mutate()
        result3, freq3 = mutate()
        for i in range(100):
            result1, freq1 = mutate(freq=freq1, amplify_edges=True)
            result2, freq2 = mutate(freq=freq2, amplify_edges=True)
            result3, freq3 = mutate(freq=freq3, amplify_edges=True)
            img = np.dstack((result1,result2,result3))
            plt.figure("result"+str(i)),plt.imshow(img, cmap='gray'), plt.colorbar()
            plt.show()
