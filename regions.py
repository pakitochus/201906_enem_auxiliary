#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:55:23 2019

@author: pakitochus

Script para visualizar un perceptrón básico de con una capa oculta de 2 
neuronas. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# generamos el grid
xx, yy = np.meshgrid(np.linspace(-10,10, 201), np.linspace(-5, 5, 101))

# concatenamos un tercer canal de 1 para el bias, para que sea inner product
xfin = np.stack((xx,yy, np.ones_like(xx)), axis=2)

w1 = np.array(np.random.randn(3))*2 #weights de neurona 0 (capa 1) [-1.5,  0.6, -0.3]
w2 = np.array(np.random.randn(3))*2 #weights de neurona 1 (capa 1) [0.78, -1.8,  0.81]

x1 = np.tanh(np.inner(w1, xfin)) # out de neurona 0 (capa 1)
x2 = np.tanh(np.inner(w2, xfin)) # out de neurona 1 (capa 1)

xresult = np.stack((x1, x2, np.ones_like(x1)), axis=2) # mismo que 20

w3 = np.array(np.random.randn(3))*2 # weights de neurona salida (capa 2)
x3 = np.tanh(np.inner(w3, xresult)) # out de neurona salida (capa 2)

plt.close('all')# para que no se acumulen los plots

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(2,2) # generamos un grid de 2x2
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, :]) # el de abajo lo concatenamos

# saturamos el valor maximo para distinguir mejor la separacion de regiones
maxval = np.abs(x1).max()
ax1.imshow(x1, vmin=-maxval, vmax=maxval, cmap='coolwarm', alpha=.8)
ax1.set_title(f'y0 = tanh({w1[0]:.1f} x0 + {w1[1]:.1f} x1 + {w1[2]:.1f})')
ax1.set_ylabel('x1')
ax1.set_xlabel('x0')
ax1.set_yticks(np.linspace(0,100,5))
ax1.set_yticklabels(np.linspace(-5, 5, 5))
ax1.set_xticks(np.linspace(0,200,5))
ax1.set_xticklabels(np.linspace(-10, 10, 5))


maxval = np.abs(x2).max()
ax2.imshow(x2, vmin=-maxval, vmax=maxval, cmap='coolwarm', alpha=.8)
ax2.set_title(f'y1 = tanh({w2[0]:.2f} x0 + {w2[1]:.2f} x1 + {w2[2]:.2f})')
ax2.set_ylabel('x1')
ax2.set_xlabel('x0')
ax2.set_yticks(np.linspace(0,100,5))
ax2.set_yticklabels(np.linspace(-5, 5, 5))
ax2.set_xticks(np.linspace(0,200,5))
ax2.set_xticklabels(np.linspace(-10, 10, 5))

maxval = np.abs(x3).max()
ax3.imshow(x3, vmin=-maxval, vmax=maxval, cmap='coolwarm', alpha=.8)
ax3.set_title(f'tahn({w3[0]:.2f} y0 + {w3[1]:.2f} y1 + {w3[2]:.2f})')
ax3.set_ylabel('y1')
ax3.set_xlabel('y0')
ax3.set_yticks(np.linspace(0,100,5))
ax3.set_yticklabels(np.linspace(-5, 5, 5))
ax3.set_xticks(np.linspace(0,200,5))
ax3.set_xticklabels(np.linspace(-10, 10, 5))

plt.tight_layout()

#%% PARA MÁS NEURONAS. 

plt.close('all') # para que no se acumulen los plots

# generamos el grid
xx, yy = np.meshgrid(np.linspace(-10,10, 201), np.linspace(-5, 5, 101))

N1 = 20 #number of neurons in 1st layer

# concatenamos un tercer canal de 1 para el bias, para que sea inner product
xfin = np.stack((xx,yy, np.ones_like(xx)), axis=2)

W1 = np.array(np.random.randn(3, N1)) #weights de neurona 0 (capa 1)

Xout = np.tanh(np.inner(W1.T, xfin)) # out de neurona 0 (capa 1)

xresult = np.vstack((Xout, np.ones_like(Xout[0])[np.newaxis])) # mismo que 20

w3 = np.array(np.random.randn(1,N1+1)) # weights de neurona salida (capa 2)
x3 = np.tanh(np.inner(w3, xresult.transpose([2, 1, 0]))) # out de neurona salida (capa 2)

fig, ax = plt.subplots(figsize=[6.7 , 3.64])

maxval = np.abs(x3).max()
ax.imshow(x3[0].T, vmin=-maxval, vmax=maxval, cmap='coolwarm', alpha=.8)
ax.set_title(f'tahn({w3[0,0]:.2f} y0 + {w3[0,1]:.2f} y1 + ... + {w3[0,-1]:.2f})')
ax.set_ylabel('y1')
ax.set_xlabel('y0')
ax.set_yticks(np.linspace(0,100,5))
ax.set_yticklabels(np.linspace(-5, 5, 5))
ax.set_xticks(np.linspace(0,200,5))
ax.set_xticklabels(np.linspace(-10, 10, 5))
plt.tight_layout()