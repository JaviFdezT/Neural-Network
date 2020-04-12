#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:46:38 2020

@author: javi
"""



import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd 
import sys
from sklearn.utils import shuffle
import random
import warnings
warnings.filterwarnings("ignore")

file="data/training.dat"
filetest="data/testing.dat"
nhlayers=1
nodes=[2]
doshuffle=True
savefile=True
lrate=2e-1
epoch=1000     #iteraction over the dataset
batchpercent=1 # porcetage of input rows used every iteraction (0-1)
tolerance=1e-1

def fo(x):
    return x
    return 1/(1 + np.exp(-x)) 
def derfo(x):
    return np.ones(x.size)
    return fo(x)*(1-fo(x))
def fh(x): 
    return np.tanh(x)
    return 1/(1 + np.exp(-x))
def derfh(x):
    return 1/(np.cosh(x)**2)
    return fh(x)*(1-fh(x))


try:
    fline=open(file).readline().rstrip().split(";")
    inputs=int(fline[0])
    targets=int(fline[-1])
    data=pd.read_csv(file,sep=(";"),comment='#',header=None,skiprows=1).astype("float")
    if doshuffle:data = shuffle(data)
    print("##################\nReading input file\n##################")
except pd.errors.EmptyDataError:
    print("################\nWrong input file\n################")
    sys.exit(0)
    
indata=np.array(data)[:, :inputs]
target=np.array(data)[:, inputs:]

w=[]
w.append(np.ones((nodes[0],inputs+1),))
for i in range(nhlayers-1):
    w.append(np.random.random((nodes[i+1],nodes[i]+1),))
w.append(np.random.random((inputs,nodes[-1]+1),))
weights=np.array(w)
miner=1e5

def forward(traininput):
    global hlayers
    global outlay
    hlayers=[]
    hlayers.append([1]+list(traininput))
    for layer in list(range(nhlayers+1)):
        if layer==nhlayers:
            outlay=list(fo(np.dot(weights[layer],hlayers[-1])))
        else:
            hlayers.append([1]+list(fh(np.dot(weights[layer],hlayers[-1]))))
    return outlay
    
for run in range(epoch):
  for row in np.random.choice(len(indata), int(np.ceil(len(indata)*batchpercent))):
    forward(indata[row])
    for layer in list(reversed(range(nhlayers+1))):
        if layer==nhlayers:            
            #print(delta,np.transpose(np.array(hlayers[layer][1:])))
            delta=np.multiply(outlay-target[row],derfo(np.dot(weights[layer],hlayers[layer])))
            change=np.dot(delta,np.array(hlayers[layer]).reshape(1,len(hlayers[layer])))
            weights[layer]-=lrate*change
        else:      
            if layer!=nhlayers-1:   
                delta=delta[1:]
            deltp=np.multiply(np.dot(np.transpose(weights[layer+1]),delta),[1]+list(derfh(np.dot(weights[layer],hlayers[layer]))))
            delta=np.multiply(np.dot(np.transpose(weights[layer+1]),delta),[1]+list(derfh(np.dot(weights[layer],hlayers[layer]))))
            change=np.dot(delta[1:].reshape(nodes[layer],1),np.array(hlayers[layer]).reshape(1,len(hlayers[layer])))
            weights[layer]-=lrate*change
    error=np.sum((outlay-target[row])**2)/len(outlay)
    if error<miner:
        miner=error
    if tolerance<miner:
        pass#print(miner)
 

try:
    ers=[]
    test=pd.read_csv(filetest,sep=(";"),comment='#',header=None,skiprows=1).astype("float")
    testdata=np.array(test)[:, :inputs]
    targetdata=np.array(test)[:, inputs:]
    outputdata=[forward([i]) for i in testdata]
    print("INPUT   -   TARGET    -   OUTPUT    -   100(t-o)/t [%]")
    for el in range(len(testdata)):
        print(testdata[el],"-",targetdata[el],"-",outputdata[el],"-",100*abs(outputdata[el]-targetdata[el])/targetdata[el])
        ers.append(100*abs(outputdata[el]-targetdata[el])/targetdata[el])
    ers=np.array(ers)
    ers[ers >= 1E308] = 0
    print("###################\nTesting results ...")
    print("Final error =",np.round(ers.mean(),decimals=3),"%")
    print("###################")
    if len(testdata[0])==1 and len(outputdata[0])==1:
        plt.figure(figsize=(7,5))
        plt.plot(testdata,outputdata,"o-",markersize=9,label="MODEL",color="blue")
        plt.plot(testdata,targetdata,"*-",label="Test",color="red")
        plt.xlabel("input",fontsize=16)
        plt.ylabel("output",fontsize=16)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=15,loc="best")
        if savefile:plt.savefig("2dfit.png",bbox_inches='tight')
except pd.errors.EmptyDataError:pass 
