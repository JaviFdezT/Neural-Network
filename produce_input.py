#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:51:02 2020

@author: javi
"""



import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


file=open("data/training.dat","w")
file.write("1;1\n")
indata=np.linspace(0,1,num=21)
outdata=[]
for i in indata:
    if i>0.5:
        outdata.append(0.1)
    else:
        outdata.append(0.9)
outdata=[]
for i in indata:
    outdata.append(i)
outdata=[]
for i in indata:
        outdata.append(i**3)
for i in range(len(indata)):
    file.write(str(indata[i])+";"+str(outdata[i])+"\n")
file.close()

file=open("data/testing.dat","w")
file.write("1;1\n")
indata=np.linspace(0,1,num=7)
outdata=[]
for i in indata:
    if i>0.5:
        outdata.append(0.1)
    else:
        outdata.append(0.9)
outdata=[]
for i in indata:
    outdata.append(i)
outdata=[]
for i in indata:
        outdata.append(i**3)
for i in range(len(indata)):
    file.write(str(indata[i])+";"+str(outdata[i])+"\n")
file.close()



