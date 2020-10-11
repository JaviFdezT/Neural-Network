#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 18:53:14 2020

@author: javi
"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import sys
from sklearn.utils import shuffle
import random
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
import tqdm



class NeuralNetwork:
        
    def __init__(self):
        """
        Initialises variables to None.

        Returns
        -------
        None.

        """
        self.normalization=None
        self.nhlayers=None
        self.nodes=None
        self.doshuffle=None
        self.savefile=None
        self.lrate=None
        self.epoch=None
        self.batchpercent=None 
        self.tolerance=None
        self.plot=None
        self.indata, self.testdata, self.target, self.targetdata = None,None,None,None
        self.features=None
        self.weights=None
        self.miner=None
        
    def loadData(self,indata,target,nhlayers,nodes,
                 test_size=0.15,normalize=True,epoch=1000,batchpercent=1,lrate=2e-1,
                 tolerance=1e-1,doshuffle=True,savefile=False,plot=False):
        """
        Loads data and initialises variables

        Parameters
        ----------
        indata : (mxn) matrix, where m is the number of points and n is the number of
                     input features.
            Input data.
        target : (mxp) matrix, where m is the number of points and p is the number of
                     output features.
            Output data.
        nhlayers : integer
            Number of hidden layers.
        nodes : array. Dim=nhlayers.
            Array which contains the number of nodes per hidden layer.
        test_size : float, optional. Range=[0-1)
            Percentage of points used as test set. The default is 0.15.
        normalize : boolean, optional
            If True, input and output data are normalized in the interval. The 
                    default is True.
        epoch : integer, optional
            Number of iteractions over the dataset. The default is 1000.
        batchpercent : float, optional. Range=(0-1]
            Percentage of input rows used every iteraction. The default is 1.
        lrate : float, optional
            Learning rate: tuning parameter that determines the step size at each
                    iteration. The default is 2e-1.
        tolerance : float, optional. Range=(0-1)
            Error tolerance. Once the relative error is minimized and reaches this value,
                    the algorithm stops. The default is 1e-1.
        doshuffle : boolean, optional
            If True, all points are shuffled. The default is True.
        savefile : boolean, optional
            If True, weights are saved. The default is False.
        plot : boolean, optional
            If True, plots are generated. The default is False.

        Returns
        -------
        None.
        """        
        self.nhlayers=nhlayers
        self.nodes=nodes
        self.doshuffle=doshuffle
        self.savefile=savefile
        self.lrate=lrate
        self.epoch=epoch     
        self.batchpercent=batchpercent 
        self.tolerance=tolerance
        self.plot=plot
        self.features=indata.shape[-1]
        self.outputs=target.shape[-1]
        
        self.normalization=[]
        for i in range(self.features):
            if normalize and np.max(indata[:,i])-np.min(indata[:,i])>0:
                self.normalization.append(((np.max(indata[:,i])-np.min(indata[:,i]))/2,np.min(indata[:,i])))
            else:
                self.normalization.append((1.0,0.0))
        for i in range(self.outputs):
            if normalize and np.max(target[:,i])-np.min(target[:,i])>0:
                self.normalization.append(((np.max(target[:,i])-np.min(target[:,i]))/2,np.min(target[:,i])))
            else:
                self.normalization.append((1.0,0.0))
        self.normalization=np.array(self.normalization)
        
        for i in range(self.features):
            indata[:,i]=(np.array(indata[:,i])-self.normalization[i][1])/self.normalization[i][0]-1
        for i in range(self.outputs):
            target[:,i]=(np.array(target[:,i])-self.normalization[i+self.features][1])/self.normalization[i+self.features][0]-1
        
        self.indata, self.testdata, self.target, self.targetdata = train_test_split(
            indata,target,test_size=test_size, random_state=42,shuffle=self.doshuffle)
     
        
        w=[]
        w.append(np.ones((nodes[0],self.features+1),))
        for i in range(nhlayers-1):
            w.append(np.random.random((nodes[i+1],nodes[i]+1),))
        w.append(np.random.random((self.outputs,nodes[-1]+1),))
        self.weights=np.array(w)
        self.miner=1e5
        
    @staticmethod
    def fo(x):
        """
        Sigmoid  function

        Parameters
        ----------
        x : float
            input value.

        Returns
        -------
        float
            output value.

        """
        return x
        return 1/(1 + np.exp(-x)) 
    def derfo(self,x):
        """
        Derivative of the sigmoid  function

        Parameters
        ----------
        x : float
            input value.

        Returns
        -------
        float
            output value.

        """
        return np.ones(x.size)
        return self.fo(x)*(1-self.fo(x))
    @staticmethod
    def fh(x): 
        """
        Sigmoid  function. This function is applied at the hidden layers.

        Parameters
        ----------
        x : float
            input value.

        Returns
        -------
        float
            output value.

        """
        return np.tanh(x)
        return 1/(1 + np.exp(-x))
    def derfh(self,x):
        """
        Derivative of the sigmoid  function. This function is applied at the hidden layers.

        Parameters
        ----------
        x : float
            input value.

        Returns
        -------
        float
            output value.

        """
        return 1/(np.cosh(x)**2)
        return self.fh(x)*(1-self.fh(x))
    
     
    def predict(self,inputvalue):
        """
        Given an input value, returns the output predicted by the NN.

        Parameters
        ----------
        
        inputvalue : (1xn) matrix, where 1 is the number of points and n is the number of
                     input features.
            Input data.

        Returns
        -------
            (mxp) matrix, where m is the number of points and p is the number of
                     output features.

        """
        for i in range(self.features):
            inputvalue[:,i]=(np.array(inputvalue[:,i])-self.normalization[i][1])/self.normalization[i][0]-1
               
        output=np.array([self.forward(np.array(list(row))) for row in inputvalue])
        for i in range(self.outputs):
            output[:,i]=(output[:,i]+1)*self.normalization[i+self.features][0]+self.normalization[i+self.features][1]
        return output 
         
    def forward(self,traininput):
        """
        Forward propagation given an input value.
        
        Parameters
        ----------
        
        traininput : (1xn) matrix, where 1 is the number of points and n is the number of
                     input features.
            Input data.

        Returns
        -------
            (mxp) matrix, where m is the number of points and p is the number of
                     output features.

        """   
        
        self.hlayers=[]
        self.hlayers.append([1]+list(traininput))
        for layer in list(range(self.nhlayers+1)):
            if layer==self.nhlayers:
                self.outlay=np.array(self.fo(np.dot(self.weights[layer],self.hlayers[-1])))
            else:
                self.hlayers.append([1]+list(self.fh(np.dot(self.weights[layer],self.hlayers[-1]))))
        self.hlayers=np.array(self.hlayers)
        return self.outlay
        
    def train(self):
        """
        Train the NN.        

        Returns
        -------
        None.

        """
        
        start_time = datetime.now()
        for run in tqdm.tqdm(range(self.epoch)):
          """if (run+1)/self.epoch in [1,0.9,0.8,0.7,0.6,0.6]:
              print("Epoch:",run+1,"(",str(100*(run+1)/self.epoch),"% )")
          """
          for row in np.random.choice(len(self.indata), int(np.ceil(len(self.indata)*self.batchpercent))):           
            self.forward(self.indata[row])
            for layer in list(reversed(range(self.nhlayers+1))):
                if layer==self.nhlayers:            
                    #print(delta,np.transpose(np.array(hlayers[layer][1:])))
                    delta=np.multiply(self.outlay-self.target[row],self.derfo(np.dot(self.weights[layer],self.hlayers[layer])))
                    change=np.dot(delta.reshape(self.outputs,1),np.array(self.hlayers[layer]).reshape(1,len(self.hlayers[layer])))
                    self.weights[layer]-=self.lrate*change
                else:      
                    if layer!=self.nhlayers-1:   
                        delta=delta[1:]
                    delta=np.multiply(np.dot(np.transpose(self.weights[layer+1]),delta),np.array([1]+list(self.derfh(np.dot(self.weights[layer],self.hlayers[layer])))))
                    change=np.dot(delta[1:].reshape(self.nodes[layer],1),np.array(self.hlayers[layer]).reshape(1,len(self.hlayers[layer])))
                    self.weights[layer]-=self.lrate*change
            error=np.sum((self.outlay-self.target[row])**2)/len(self.outlay)
            if error<self.miner:
                self.miner=error
            if self.tolerance<self.miner:
                pass
        time_elapsed = datetime.now() - start_time
        print("###################")

        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


    def test(self):
        """
        Tests the performance of the NN.        

        Returns
        -------
        None.

        """
        try:
            ers=[]
            outputdata=[self.forward(i) for i in self.testdata]
            if self.plot:print("INPUT   -   TARGET    -   OUTPUT    -   100(t-o)/t [%]")
            for el in range(len(self.testdata)):
                if self.plot:print(self.testdata[el],"-",self.targetdata[el],"-",outputdata[el],"-",100*abs(outputdata[el]-self.targetdata[el])/self.targetdata[el])
                if outputdata[el]-self.targetdata[el]<0.01:
                    ers.append(0)
                else:
                    ers.append(100*abs(outputdata[el]-self.targetdata[el])/abs(self.targetdata[el]))
            ers=np.array(ers)
            ers[ers >= 1E308] = 0
            print("###################\nTesting results ...")
            print("Final error =",np.round(ers.mean(),decimals=3),"%")
            print("###################")
        except pd.errors.EmptyDataError:pass 

    def loadModel(self,filename):
        """
        Loads a NN from a file.

        Parameters
        ----------
        filename : string
            File name where the NN is saved.

        Returns
        -------
        None.

        """
        file=open(filename,"r")
        for line in file:
            if "features" in line:
                data=line.replace("\n","").split("=")
                self.features=int(data[1])
            elif "outputs" in line:
                data=line.replace("\n","").split("=")
                self.outputs=int(data[1])
            elif "nhlayers" in line:
                data=line.replace("\n","").split("=")
                self.nhlayers=int(data[1])
            elif "nodes" in line:
                data=line.replace("\n","").split("=")                
                data=np.array([i for i in data[1].split("  ") if len(i)>0],dtype=int)
                if len(data)!=self.nhlayers:
                    sys.exit("Data couldn't be imported correctly. Dim(nodes)!=nhlayers.")
                self.nodes=data
            elif "normalization" in line:
                data=line.replace("\n","").split("=")
                data=np.array([i for i in data[1].split("  ") if len(i)>0],dtype=float)
                if len(data)!=2*(self.features+self.outputs):
                    sys.exit("Data couldn't be imported correctly. Dim(vector)!=4.")
                data=data.reshape(self.features+self.outputs,2)
                for i in range(len(data)):
                    data[i]=tuple(data[i])
                self.normalization=data
            elif "weights" in line:
                w=[]
                for i in range(self.nhlayers+1):
                    data=file.readline()
                    data=[i for i in data.replace("\n","").split(" ") if len(i)>0]
                    if len(data)==2:
                        curw=[]
                        wx,wy=int(data[0]),int(data[1])
                        for i in range(wx*wy):    
                            data=file.readline()
                            data=[i for i in data.replace("\n","").split(" ") if len(i)>0]
                            curw.append(float(data[0]))
                        curw=np.reshape(np.array(curw,dtype=float),(wx,wy))
                        w.append(curw)
                self.weights=np.array(w)
        file.close()
            
    def saveModel(self,filename):
        """
        Saves a NN into a data file        

        Parameters
        ----------
        filename : string
            File name where the NN is saved.

        Returns
        -------
        None.

        """
        file=open(filename,"w")
        file.write("features="+str(self.features)+"\n")
        file.write("outputs="+str(self.outputs)+"\n")
        file.write("nhlayers="+str(self.nhlayers)+"\n")
        file.write("nodes="+"  ".join([str(nodes) for nodes in self.nodes])+"\n")
        norm=list()
        for item in self.normalization:
            norm.append(item[0])
            norm.append(item[1])
        file.write("normalization="+"  ".join([str(item) for item in norm])+"\n")
        file.write("weights=\n")
        for i in range(self.nhlayers+1):
            file.write((str(self.weights[i].shape[0])+"  "+str(self.weights[i].shape[1])+"\n"))
            for weight in self.weights[i].reshape(1,self.weights[i].shape[0]*self.weights[i].shape[1])[0]:
                file.write(str(weight)+"\n")
        file.close()
            