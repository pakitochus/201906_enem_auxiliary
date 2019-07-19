#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:40:59 2019

@author: pakitochus
"""

# pintar cosas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib import animation, rc

import copy

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
    
#draw_romb(np.array([0,0]), colors_filt)
#

finalsize = (4,4)

             
distance = 10
fsize = 3
msize = (6,6)

matrix = np.random.random(msize)
filtro = np.random.random((fsize, fsize))

mcolors = ['#'+hex(int(el*15))[2]*6 for el in matrix.flatten()]
colors_filt = ['#'+hex(int(el*15))[2]*6 for el in filtro.flatten()]

patch_list = {'offset':[], 'color': []}

plt.close('all')


for ix in range(np.prod(finalsize)):
    fig, ax = plt.subplots(figsize=(12,8))
    r,c = np.unravel_index(ix, finalsize)
    
    pos_foa = np.array([-distance+c, c+2*r])
    pos_filter = np.array([c, c+2*r]) #[c, c+2*r]
    pos_patch = np.array([distance+1+c,  c+2*r+3])
    
    draw_matrix(ax, np.array([-distance, 0]), msize, mcolors)
               
#    draw_action_field(ax, pos_foa, pos_filter, 
#                      (fsize,fsize), (fsize,fsize))
#    
    draw_matrix(ax, pos_filter, (fsize,fsize), colors_filt)
    
#    draw_action_field(ax, pos_filter, pos_patch, 
#                      (fsize,fsize), (1,1))
    
    var = np.sum(matrix[r:r+3, c:c+3]*filtro)
    newcolor = '#'+hex(int(var*16/2))[-1]*6
    patch_list['offset'].append(pos_patch)
    patch_list['color'].append(newcolor)
    
    for i in range(len(patch_list['color'])):
        draw_patch(ax, patch_list['offset'][i], patch_list['color'][i])   
        
    ax.set_ylim([-1, 18])
    ax.set_xlim([-12, 15])
    plt.axis('off')
    plt.show()
    fig.savefig(f'frame{ix}.png')
    plt.close(fig)

               
    



#%%














