# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:22:01 2022

@author: elgaral
"""

import numpy as np
import pylab as plt
import Clase_propagacion as ptf
import Clase_objetosEntrada as oE
import Clase_Graficadores as gr
# Se crea un objeto de la clase propa y se definen los parámetros de entrada
prueba = ptf.propa()
prueba.z = 1000.0
prueba.L = 6.0
prueba.N = 1024
prueba.wl = 0.5e-3

# Se decide trabajar con una resolución particular por lo que se redefine
# el tamaño físico del espacio de trabajo
dx = 8e-3
prueba.L = prueba.N*dx

# Se le pide a la clase mostrar los parámetros y verificar la condición 
# de resolución del objeto "prueba"
prueba.parameters()
prueba.resolution()

# Se crea el objeto "mask" de la clase
mask = oE.objPlanoEntrada(L=prueba.L,N=prueba.N)

#Algunos posibles plano de entrada
u1 = mask.centerCircle(radi=1.0)
#u1 = mask.centerSquare(l=2.0)
#u1 = mask.centerGauss(w0=2.5)*mask.centerSPP(ell=2)

# Graficación plano de entrada
# Se crea objeto de la clase graficadores.

grafica = gr.FieldPlotter(dx = dx, units = 'um' )
# grafica.plot_amplitude(u1)

# Propagación usando TF e IR
H = prueba.TFunc()
h = prueba.IRfunc()
u2 = prueba.propaTF(u1)
u3 = prueba.propaIR(u1)
u4 = mask.centerSPP(ell = 4)

# Graficación plano de salida


# grafica.plot_amplitude(u2, title = 'Amplitud: con TF')
# grafica.plot_phase(u2, title = 'Fase: con TF')
# plt.show()

grafica.plot_phase(u4, title = 'vórtice')
grafica.plot_compare([u2, u3, u4], labels = ['propTF', 'propIR', 'vórtice'])
plt.show()
# plt.figure()
# plt.title('Amplitud: con IR')
# plt.imshow(abs(u3))
# plt.colorbar()
# plt.figure()
# plt.title('Fase: con IR')
# plt.imshow(np.angle(u3), cmap = 'twilight')
# plt.colorbar()
# grafica.plot_amplitude(u3, title = 'Amplitud: con TF')
# grafica.plot_phase(u3, title = 'Fase: con TF')
# plt.show()




