# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 16:43:55 2023

@author: Joe
"""

import collections
import numpy as np 
import dimod
import pandas as pd
import matplotlib.pyplot as plt
from dimod import BinaryQuadraticModel,ConstrainedQuadraticModel, Integer, Binary, ExactSolver, ExactCQMSolver
from dwave.samplers import SimulatedAnnealingSampler
import pickle
import time
# In[] prepare weight matrx
data = pickle.load(open("op_data.p","rb"))
# W=[[0,2,0,0,0,2,0],
#    [2,0,2,2,0,0,0],
#    [0,2,0,2,0,0,0],
#    [0,2,2,0,1,0,0],
#    [0,0,0,1,0,1,0],
#    [2,0,0,0,1,0,1],
#    [0,0,0,0,0,1,0]]

# W=np.array(W)

W_all=data[0]
W_all=np.array(W_all*10000000,dtype=int)

arr_size=np.size(W_all[0])
Wf=np.copy(np.transpose(W_all[arr_size-2,:]))
Wb=np.copy(np.transpose(W_all[:,arr_size-1]))
W=np.copy(W_all)
W[arr_size-2,:]=W[:,arr_size-1]=0

# In[]

x1 = np.array([dimod.Binary(f'x_{j}') for j in range(np.size(W,1))])

#prepare decision variables
x=np.reshape(x1,(np.size(x1),1))
xt=np.transpose(x)

#ones vector
one=np.ones(np.shape(x))
oneT=np.ones(np.shape(xt))

theta=np.zeros(np.shape(xt))
size=np.size(x)
theta[0][size-2]=theta[0][size-1]=1


#objective function
z1=np.matmul(x,oneT-xt)+np.matmul(one-x,xt)
z2=np.multiply(z1,W)
z3=np.matmul(z2,one)
z4=np.matmul(oneT,z3)

z5=np.matmul(np.transpose(x),Wf)+np.matmul(np.transpose(one-x),Wb)


z=z4[0][0]+z5[0]

#constraint
c1=np.matmul(theta,x)
c2=c1[0][0]-1
c=pow(c2,2)

bqm=BinaryQuadraticModel(z+100*c)

starttime = time.time()
sampleset = SimulatedAnnealingSampler().sample(bqm)
timetaken = time.time()-starttime
samples=sampleset.samples()
print(timetaken)

f = open("SA.txt", "a")
f.write(str(timetaken)+"\t")
f.close()


# In[]