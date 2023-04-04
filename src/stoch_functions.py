# Functions with sum of terms structure
import torch
import numpy as np

def LevyN13_i(i,x):
    w = 1.0+(x-1.0)/4.0
    if i==0:
        return  torch.sin(torch.pi*w[0])**2
    else:
        return  ( (w[i-1]-1)**2*(1+10*torch.sin(torch.pi*w[i-1]+1)**2)+ (w[-1]-1)**2*(1+torch.sin(2*torch.pi*w[-1])**2)         )

def PermDBeta_i(i,x):
    beta=0.5  # Added default value for beta
    v = 0
    d = x.size(dim=0)
    # j = np.arange(1, d+1)
    # j = range(1,d+1)
    # v = torch.sum(((j**(i+1) + beta) *((x/j)**(i+1) - 1))**2 ) 
    for j in range(d):
        # v+=(1+j+beta)*(x[j]**(i+1) -1. / (j+1)**(i+1))   
        # v+=(1+j+beta)*(x[j]/(j+1)**(i+1))  
        v+= ((((j+1)**(i+1) + beta) *((x[j]/(j+1))**(i+1) - 1))**2)/d 
    return v
    
def Rastrigin_i(i,x):
    v = 10+x[i]**2-10*torch.cos(2*torch.pi*x[i])
    return v
    
def RosenBrock_i(i,x):
    b = 100
    a = 1
    return b * torch.pow(x[i] - torch.pow(x[i - 1], 2) , 2) + torch.pow(a - x[i], 2)