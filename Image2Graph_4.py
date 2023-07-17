# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:16:20 2023

@author: JoeP
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pickle
from FgBghist import get_hist
# In[]


def position(C,i,j):
    return (C*i)+j

# def index(R,C,pos):
#     return [int(pos/(R-1))-1,pos-int(pos/C)*C]

# In[]

fname='coins10'
# image1=np.ones([4,4])
# image1[1][0]=220
# image1[0][1]=190
image1 = cv2.imread(fname+'.png',0)
image1 = np.array(image1,dtype=float)
[r,c]=np.shape(image1)
no_of_pixel=r*c
n=2
Adj=np.zeros([no_of_pixel+n,no_of_pixel+n])
w=np.zeros([no_of_pixel+n,no_of_pixel+n])


neighbour=1
for i in range(r):
    for j in range(c):
        # print(i,j)
        x=position(c, i, j)
        z=position(c, i, j+1)
        if i+1<r:
            y=position(c, i+1, j)
            Adj[x][y]=Adj[y][x]=1
            w[x][y]=w[y][x]=np.exp(-abs(image1[i][j]-image1[i+1][j]))
        if j+1<c:
            z=position(c, i, j+1)
            Adj[x][z]=Adj[z][x]=1
            w[x][z]=w[z][x]=np.exp(-abs(image1[i][j]-image1[i][j+1]))
            
[Bg,Fg,Pbg,Pfg]=get_hist(fname)


s_nodes=np.zeros([r,c])
t_nodes=np.zeros([r,c])

allpos=[]
for i in range(r):
    for j in range(c):
        k=position(c, i, j)
        pb=Bg[int(image1[i][j]-1)]
        pf=Fg[int(image1[i][j]-1)]
        
        bmax=max(Bg)
        fmax=max(Fg)
        
        Adj[no_of_pixel][k]=1#Adj[k][no_of_pixel]=.2
        Adj[k][no_of_pixel+1]=1#Adj[no_of_pixel+1][k]=.8
            
        w[no_of_pixel][k]=np.exp(-((pb)+.001)*(255/bmax))#Adj[k][no_of_pixel]=.2
        w[k][no_of_pixel+1]=np.exp(-((pf)+.001)*(255/fmax))#Adj[no_of_pixel+1][k]=.8
        
        s_nodes[i][j]=w[no_of_pixel][k]
        t_nodes[i][j]=w[k][no_of_pixel+1]
    
        allpos.append([k,w[no_of_pixel][k],pf])   
        # w[no_of_pixel][k]=1+np.exp(pf-pb)#Adj[k][no_of_pixel]=.2
        # w[k][no_of_pixel+1]=1+np.exp(pb-pf)#Adj[no_of_pixel+1][k]=.8
        
        # if pb==0 and pf==0:
        #     w[k][no_of_pixel+1]=1+3*math.log(0.01)

for i in Pfg:
    k=position(c, i[0], i[1])
    Adj[no_of_pixel][k]=1#Adj[k][no_of_pixel]=.2
    Adj[k][no_of_pixel+1]=1#Adj[no_of_pixel+1][k]=.8
    
    w[no_of_pixel][k]=10#Adj[k][no_of_pixel]=.2
    w[k][no_of_pixel+1]=0#Adj[no_of_pixel+1][k]=.8
    s_nodes[i[0]][i[1]]=w[no_of_pixel][k]
    t_nodes[i][0][i[1]]=w[k][no_of_pixel+1]
    
for i in Pbg:
    k=position(c, i[0], i[1])
    Adj[no_of_pixel][k]=1#Adj[k][no_of_pixel]=.2
    Adj[k][no_of_pixel+1]=1#Adj[no_of_pixel+1][k]=.8
    
    print(i)
    w[no_of_pixel][k]=0#Adj[k][no_of_pixel]=.2
    w[k][no_of_pixel+1]=10#Adj[no_of_pixel+1][k]=.8
    s_nodes[i[0]][i[1]]=w[no_of_pixel][k]
    t_nodes[i[0]][i[1]]=w[k][no_of_pixel+1]


print(1)          
            
data=[w,Adj,[r,c]]
pickle.dump([w,Adj,[r,c]], open("op_data.p", "wb"))   
#np.savez('op', w,Adj,r,c)   
print(2)   
plt.imshow(image1, cmap=plt.get_cmap('gray'))


plt.show()


f = open("SA.txt", "a")
f.write("\n\n"+str(r)+"\n")
f.close()
