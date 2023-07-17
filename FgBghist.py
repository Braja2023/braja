# -*- coding: utf-8 -*-
"""
Created on Sat May 27 22:18:57 2023

@author: Joe
"""

from PIL import Image,ImageOps
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle


def get_hist(fname):
    

    image1 = Image.open(fname+'m.png')
    imageGrey = ImageOps.grayscale(Image.open(fname+'.png'))
    imageGrey=np.asarray(imageGrey)
    image1 = np.asarray(image1)
    
    # image1 = cv2.imread(fname+'m.png')
    # imageGrey = cv2.imread(fname+'.png',0)
    
    R=image1[:,:,0]
    G=image1[:,:,1]
    B=image1[:,:,2]
    
    pixelsFg=[]
    pixelsBg=[]
    
    
    [r,c]=np.shape(B)
    
    for i in range(r):
        for j in range(c):
            if R[i][j]==255 and G[i][j]==0 and B[i][j]==0:
                pixelsFg.append([i,j])
            if R[i][j]==0 and G[i][j]==255 and B[i][j]==0:
                pixelsBg.append([i,j])
                
    bg=np.zeros([r,c])
    fg=np.zeros([r,c])    
    
    
        
    histBg=[0 for i in range(255)]
    histFg=[0 for i in range(255)]
    
    for i in pixelsBg:
        bg[i[0]][i[1]]=255
        histBg[imageGrey[i[0]][i[1]]]+=1
    for i in pixelsFg:
        fg[i[0]][i[1]]=255
        histFg[imageGrey[i[0]][i[1]]]+=1
    histBgNorm=histBg#[(i/len(histBg)) for i in histBg]
    histFgNorm=histFg#[(i/len(histFg)) for i in histFg]
    # plt.plot([i/len(histBg) for i in histBg])
    # plt.plot([i/len(histFg) for i in histFg])
    # plt.show()
    return [histBgNorm,histFgNorm,pixelsBg,pixelsFg]
