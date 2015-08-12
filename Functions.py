# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:52:06 2015

@author: elgaral
"""

import numpy as np
import cmath as cm
import scipy.misc

s=[]


"""
###### Gaussian beam (Saleh, 3. 1-7) #####

N: matrix size
dx: sample size
W0:  beam waist radius
A0:  Beam maximum amplitud at waist
wl:  wavelength
z:   distance from the waist
info: 1 if you want the main beam information printed

G: Gaussian beam complex amplitude
"""
def Gbeam(N,dx,W0,A0,wl,z,info):
    if z == 0:
        z = 1.0e-20
        
    z0 = np.pi*W0**2/wl  # Rayleigh range
    W = W0*np.sqrt(1 + (z/z0)**2)  # Beam width at z
    k = 2*np.pi/wl # wave number
    R = z*(1 + (z0/z)**2) # beam radius of curvature
    Xi = np.arctan(z/z0) # Guoy phase
    
    x = np.arange(-N*dx/2,N*dx/2,dx)  # -N*dx/2 <= x <= N*dx/2 - dx
    X , Y = np.meshgrid(x,x)   #
    Amp = np.exp(-(X**2 + Y**2)/W**2)
    Phase = np.exp(-1j*k*z - 1j*k*(X**2 + Y**2)/(2*R) + 1j*Xi)
    
    G = A0*(W0/W)*Amp*Phase
    return G
    
"""
###### Thin Lens (Goodman, 5. 5-10) #####

N: matrix size
dx: sample size
wl:  wavelength
f:   focal distance

Lens: Lens complex amplitude
""" 
def Lente(N,dx,f,wl):
    k = 2*np.pi/wl
    
    x = np.arange(-N*dx/2,N*dx/2,dx)  # -N*dx/2 <= x <= N*dx/2 - dx
    X , Y = np.meshgrid(x,x)   #
    Lens = np.exp(-1j*k/(2*f)*(X**2 + Y**2))
    return Lens

"""
###### Phase of a complex Image #####

N: matrix size
Ima:  Complex Image

Fase: Phase of the complex image
""" 
def FaseIma(N,Ima): 
    Fase = np.zeros((N,N))
    for cont1 in np.arange(0,N):
        for cont2 in np.arange(0,N):
            Fase[cont1,cont2] = cm.phase(Ima[cont1,cont2])
    return Fase
    
    
"""
###### SPP  #####

N: matrix size
dx: sample size
ch: SPP charge

SPP: Spiral phase plate complex value
"""
def SPP(N,dx,ch):    
    x = np.arange(-N*dx/2,N*dx/2,dx)  # -N*dx/2 <= x <= N*dx/2 - dx
    X , Y = np.meshgrid(x,x)   #
    
    angle = np.arctan2(Y,X)
    SPP = np.exp(1j*ch*angle)
    return SPP
    
"""    
###### Scale Image #####

N: matrix size
Esc: scaling factor
Type: type of image, real amplitude (amp) or phase (fase)
G: Image of real values

G2: scaled image
"""
def scalaIma(N,Esc,Type,G):
    if Esc != 1:
        if Type == 'amp':
            maxi = np.amax(G)
            G = 255*G/maxi
            G = scipy.misc.imresize((G),Esc)
            G = G*maxi/255.
            print maxi
        elif Type == 'fase':
            G = 255*(G + np.pi)/(2*np.pi)
            G = scipy.misc.imresize(G,Esc)
            G = 2*np.pi*G/255. - np.pi
    
    
        NN = np.round(N*Esc)
        if NN % 2 != 0: #odd
            NN = NN  -1
        
        if NN > N:
            G2 = G[(NN-N)/2:(NN-N)/2+N,(NN-N)/2:(NN-N)/2+N]
            
        if NN < N:
            G2 = np.zeros((N,N))
            G2[(N-NN)/2:(N-NN)/2+NN,(N-NN)/2:(N-NN)/2+NN] = G
    else:
        G2 = G
    
    return G2
