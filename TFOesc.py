# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:55:30 2015

@author: ER
"""
"""
###### Optical Fourier Transform: scaled fft2 #####

This is only to test the program

W0:  beam waist radius
A0:  Beam maximum amplitud at waist
z:   distance from the waist
Gb: Gaussian beam complex amplitude
info: not working
ch: Optical vortex charge

dx: sample size
N: matrix size for scaled fft2
f: OFT lens focal distance
wl:  wavelength
NN: view window size

Inplane: Inplane complex amplitude for a SPP and a Gaussian beam
U1: OFT of the Inplane
FU1: phse of U1

------
Explanation for the scaling of the matrix

if the sampling is dx, the max freq is fmax = 1/(2*dx)
If at the exit plane (OFT plane) the max cood with respect to the optical axis
is Umax = wl*F*fmax, where F is the focal length, 2Umax = Ndx =wl*F/(dx)^2
N = wl*F/(dx)^2. Recalling that the sampling has to be the same at the inplan
and exit plane
------

"""

import Functions as fun
import pylab as plt
import numpy as np


NN = 512  #view window in pixels
f = 2.0 # focal distance
dx = 1.0e-3 #sample size in mm
ch = 5 #SPP charge
W0 = 0.1 # beam waist in mm
A0 = 1.0 # beam max real amplitude
wl = 532.0e-6 # wavelength in mm
z = 0.0 # Gaussian bean distance from its waist in mm
info = 0 #Gaussian beam information

N =wl*f/(dx)**2 #matrix size need it in pixels

#%% Creates the inplane SPP and Gauss beam
VO = fun.SPP(N,dx,ch)
Gb = fun.Gbeam(N,dx,W0,A0,wl,z,info)
Inplane = VO*Gb # inplane field

FInplane = fun.FaseIma(N,Inplane)

plt.figure(1)
Inplane2 = Inplane[(N-NN)/2:(N-NN)/2+NN,(N-NN)/2:(N-NN)/2+NN]
im = plt.imshow(abs(Inplane2), cmap='gray',interpolation='nearest')
plt.title('Inplane real amplitude, wiew window')
plt.colorbar(im, orientation='horizontal')
plt.axis('off')
#plt.savefig("test-FT.png",bbox_inches='tight')
#plt.show()
del Inplane2

plt.figure(2)
FInplane2 = FInplane[(N-NN)/2:(N-NN)/2+NN,(N-NN)/2:(N-NN)/2+NN]
im = plt.imshow(FInplane2, cmap='gray',interpolation='nearest', vmin=-np.pi , vmax=np.pi)
plt.title('Inplane phase, view window')
plt.colorbar(im, orientation='horizontal')
plt.axis('off')
#plt.savefig("test-FT.png",bbox_inches='tight')
#plt.show()
del FInplane2

#%% OFT by fft2

U1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Inplane)))
FU1 = fun.FaseIma(N,U1)


plt.figure(3)
U12 = U1[(N-NN)/2:(N-NN)/2+NN,(N-NN)/2:(N-NN)/2+NN]
im = plt.imshow(abs(U12), cmap='gray',interpolation='nearest')
plt.title('FFT real amplitude')
plt.colorbar(im, orientation='horizontal')
plt.axis('off')
#plt.savefig("test-FT.png",bbox_inches='tight')
#plt.show()
del U12

plt.figure(4)
FU12 = FU1[(N-NN)/2:(N-NN)/2+NN,(N-NN)/2:(N-NN)/2+NN]
im = plt.imshow(FU1, cmap='gray',interpolation='nearest', vmin=-np.pi , vmax=np.pi)
plt.title('FFT phase')
plt.colorbar(im, orientation='horizontal')
plt.axis('off')
#plt.savefig("test-FT.png",bbox_inches='tight')
#plt.show()
del FU12