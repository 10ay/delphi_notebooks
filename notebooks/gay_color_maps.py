#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:50:33 2020

@author: cailingallinger
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from scipy.interpolate import interp2d

x,y,c = zip(*np.random.rand(300,3)*4-2)

#x0 = np.arange(-5.01, 5.01, 0.25)
#x1 = np.arange(-5.01, 5.01, 0.25)
#xx, yy = np.meshgrid(x0, x1)
#z = np.sin(xx**2+yy**2)

norm=plt.Normalize(0,1) 

def pride():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#760089","#0044ff","#00811f","#ffef00","#ff8c00","#e70000","#784e18","#000000"])

def pride_r():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#000000","#784e18","#e70000","#ff8c00","#ffef00","#00811f","#0044ff","#760089"])

def trans():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#55CDFC","#55CDFC","#F7A8B8","#F7A8B8","#FFFFFF","#FFFFFF"]) 

def trans_r():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FFFFFF","#FFFFFF","#F7A8B8","#F7A8B8","#55CDFC","#55CDFC"]) 

def bi():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#0000ff","#0000ff","#a349a4","#ff0080","#ff0080"]) #reversed

def bi_r():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ff0080","#ff0080","#a349a4","#0000ff","#0000ff"]) 

def lesbian():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#D62900","#FF9B55","#FFFFFF","#D461A6","#A50062"])

def lesbian_r():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#A50062","#D461A6","#FFFFFF","#FF9B55","#D62900"])

def gq():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#48821E","#48821E","#FFFFFF","#FFFFFF","#B77FDD","#B77FDD"]) 

def gq_r():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#B77FDD","#B77FDD","#FFFFFF","#FFFFFF","#48821E","#48821E"])

def nb():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#000000","#9C59D1","#FFFFFF","#FFF430"])

def nb_r():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FFF430","#FFFFFF","#9C59D1","#000000"])

def leather():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#000000","#26297e","#26297e","#000000","#26297e","#26297e","#FFFFFF","#26297e","#26297e","#000000","#e50d3c","#26297e","#000000"])

def leather_r():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#000000","#26297e","#e50d3c","#000000","#26297e","#26297e","#FFFFFF","#26297e","#26297e","#000000","#26297e","#26297e","#000000"])

def pan():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FF218C","#FF218C","#FFD800","#FFD800","#21B1FF","#21B1FF"])

def pan_r():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#21B1FF","#21B1FF","#FFD800","#FFD800","#FF218C","#FF218C"])
