#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:40:59 2019

@author: pakitochus
"""

# pintar cosas
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

global points 
points = np.array([[0,0], [0,2], [1, 3], [1,1]])

def draw_patch(ax, position, color):
    patch = plt.Polygon(points + position, color=color)
    ax.add_patch(patch)
    return patch

def draw_matrix(ax, offset, shape, colorlist = None):
    #vertical 2, #horizontal 1
    N = np.product(list(shape))
    if colorlist==None:
        colorlist = ['#'+str(hex(np.random.randint(16)))[2:]*6 for i in range(N)]
    else:
        assert len(colorlist)>=N
        
    matrix = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            curr_ix = np.ravel_multi_index((r, c), shape)
            matrix.append(draw_patch(ax, offset + np.array([r,r+c*2]), colorlist[curr_ix]))
            
    return matrix


def draw_action_field(ax, offset_in, offset_out, shape_in, shape_out, color='red'):
    
    ax.plot([offset_in[0], offset_in[0]], [offset_in[1], offset_in[1]+shape_in[1]*2], color=color)
    ax.plot([offset_in[0], offset_in[0]+shape_in[0]], [offset_in[1], offset_in[1]+shape_in[0]], color=color)
    ax.plot([offset_in[0], offset_in[0]+shape_in[0]], [offset_in[1]+shape_in[1]*2, offset_in[1]+shape_in[0]+shape_in[1]*2], color=color)
    ax.plot([offset_in[0]+shape_in[0], offset_in[0]+shape_in[0]], 
            [offset_in[1]+shape_in[0], offset_in[1]+shape_in[0]+shape_in[1]*2], color=color)
    
    ax.plot([offset_in[0], offset_out[0]], [offset_in[1], offset_out[1]], color=color)
    ax.plot([offset_in[0], offset_out[0]], 
            [offset_in[1]+shape_in[1]*2, offset_out[1]+shape_out[1]*2], color=color)
    ax.plot([offset_in[0]+shape_in[0], offset_out[0]+shape_out[0]], 
            [offset_in[1]+shape_in[1]*3, offset_out[1]+shape_out[1]*3], color=color)
    ax.plot([offset_in[0]+shape_in[0], offset_out[0]+shape_out[0]], 
            [offset_in[1]+shape_in[0], offset_out[1]+shape_out[0]], color=color)
    


load_image = 'cat.gif' # set to None if random image. 
# se pueden usar las imagenes "cat, clock, face o tree .gif". 

if load_image == None:
    msize = (8,8) # tamaño matriz inicial: numero par. 
    matrix = np.random.random(msize)
else:
    matrix = plt.imread(load_image)
    if matrix.ndim>2:
        matrix = matrix.mean(axis=2)
    msize = matrix.shape
    
    
filtro = 'sobel' # either 'sobel' or None
if filtro==None:
    fsize = 3 # tamaño de filtro (impar)
    filtro = np.random.random((fsize, fsize))
elif filtro=='sobel':
    filtro = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])/8
    fsize = 3
    
padding = 0
stride = 1
finalwidth = (msize[0]-fsize+2*padding)/stride + 1
finalsize = (int(finalwidth), int(finalwidth))

distance = 10 #distancia entre las matrices


mcolors = ['#'+hex(int(el*15))[2]*6 for el in matrix.flatten()]
colors_filt = ['#'+hex(int(el*15))[2]*6 for el in (filtro-filtro.min()).flatten()]

patch_list = {'offset':[], 'color': []}

plt.close('all')


fig, ax = plt.subplots(figsize=(12,8))

def init(ax, msize, finalsize, distance):
    """ Inicializa los ejes"""
    ax.clear()
    ax.set_ylim([-2, msize[1]*3])
    ax.set_xlim([-distance-2, distance+finalsize[0]+2])
    ax.set_axis_off()
    ax.set_aspect('equal')
    
def update(ix, ax, msize, fsize, finalsize, distance):
    """ Actualiza el plot y repinta el frame completo """
    init(ax, msize, finalsize, distance)
    r,c = np.unravel_index(ix, finalsize)
    
    pos_foa = np.array([-distance+c, c+2*r])
    pos_filter = np.array([c, c+2*r]) #[c, c+2*r]
    pos_patch = np.array([distance+1+c,  c+2*r+fsize])
    
    draw_matrix(ax, np.array([-distance, 0]), msize, mcolors)
               
    draw_action_field(ax, pos_foa, pos_filter, 
                      (fsize,fsize), (fsize,fsize))
    
    draw_matrix(ax, pos_filter, (fsize,fsize), colors_filt)
    
    draw_action_field(ax, pos_filter, pos_patch, 
                      (fsize,fsize), (1,1))
    
    var = np.sum(matrix[r:r+fsize, c:c+fsize]*filtro)
    newcolor = '#'+hex(int(var*16/2))[-1]*6
    patch_list['offset'].append(pos_patch)
    patch_list['color'].append(newcolor)
    
    for i in range(len(patch_list['color'])):
        draw_patch(ax, patch_list['offset'][i], patch_list['color'][i])   
        
    
    return ax



anim = FuncAnimation(fig, update, frames = np.prod(finalsize), 
                     fargs = (ax, msize, fsize, finalsize, distance), 
                     interval = 500) 

anim.save('conv_animation.gif', dpi=80, writer='imagemagick')
    



#%%














