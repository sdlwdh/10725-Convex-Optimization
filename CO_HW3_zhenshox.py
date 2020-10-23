#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp


# # Question 3

# **3.1**

# In[2]:


df=pd.read_csv('samples.csv',header=None)


# In[24]:


d=100
N=1000
X=np.array(df.iloc[:,:d])
y=np.array(df.iloc[:,-1])


# In[35]:


#(3.1.1)
w=cp.Variable(d)
#prob=cp.Problem(cp.Minimize(cp.sum(cp.log(1+cp.exp(cp.multiply(-y,X@w))))))
prob=cp.Problem(cp.Minimize(cp.sum(cp.logistic(cp.multiply(X@w,-y)))))
#prob=cp.Problem(cp.Minimize(f(w.value)))

#prob.solve(solver=cp.GUROBI)
prob.solve()
w_opt=w.value
loss_opt=cp.sum(cp.logistic(cp.multiply(-y,X@w.value))).value/N

print('Optimal w is \n',w_opt)
print('\nOptimal loss is ',loss_opt)


# In[72]:


#(3.1.2)
def SGD(b,eta,X=X,y=y,Niter=500,Nsim=25):
    d=100
    N=1000
    
    def f(w):
        return np.mean(np.log(1+np.exp(np.multiply(-y,X@w))))
    
    def df(w,X,y):
        dw=np.zeros(d)
        N=len(X)
        for i in range(N):
            dw=dw+1/N*np.exp(-y[i]*w.T@X[i,:])/(1+np.exp(-y[i]*w.T@X[i,:]))*(-y[i]*X[i,:])
        return dw
    
    fs=np.zeros(Niter)
    for _ in range(Nsim):
        w=np.zeros(d) #d=100
        for t in range(Niter):
            ind=np.random.choice(range(1000), size=b, replace=True)
            dw=df(w,X[ind],y[ind])
            w-=eta*dw
            fs[t]+=f(w)/Nsim
    return fs


# In[52]:


bs=[1,10,100,1000]
etas=[1,0.3,0.1,0.01]
F=np.zeros((16,500))

i=0
for b in bs:
    for eta in etas:
        f=SGD(b,eta)
        F[i,:]=f
        i+=1


# In[53]:


f500=F[:,-1]
print('f_hat(500):\n\t etas={}'.format(etas))
for i in range(4):
    print('b={}: {}'.format(bs[i],f500[4*i:4*(i+1)].round(6)))


# In[59]:


#(3.1.3) Set eta and vary b
Niter=500
f_star=loss_opt
for i in range(4):
    eta=etas[i]
    plt.figure(figsize=(16,12))
    for j in range(4):
        plt.plot(list(range(Niter)),F[4*j+i,:]-f_star,label='eta={},b={}'.format(etas[i],bs[j]))
    plt.xlabel('Iteration t')
    plt.ylabel('log(f(w_t)-f_min)')
    plt.yscale('log')
    plt.title('eta={}'.format(eta))
    plt.legend()
    plt.show()


# In[60]:


#(3.1.4) Set b and vary eta
Niter=500
f_star=loss_opt
for i in range(4):
    b=bs[i]
    plt.figure(figsize=(16,12))
    for j in range(4):
        plt.plot(list(range(Niter)),F[4*i+j,:]-f_star,label='eta={},b={}'.format(etas[j],bs[i]))
    plt.xlabel('Iteration t')
    plt.ylabel('log(f(w_t)-f_min)')
    plt.yscale('log')
    plt.title('b={}'.format(b))
    plt.legend()
    plt.show()


# #(3.5)
# 
# Effects of batch size: Larger batch size always better in terms of per iteration but also comes with extra per iteration costs. When eta is large, increase batch size has more significant effect.
# 
# Effects of eta: As batch size increases, larger eta will have better results.

# **3.2**

# In[99]:


#(3.2)
def SAGA(b,eta=0.1,X=X,y=y,Niter=500,Nsim=25):
    d=100
    N=1000
    
    def f(w):
        return np.mean(np.log(1+np.exp(np.multiply(-y,X@w))))
    
    def df(w,X,y):
        dw=np.zeros(d)
        N=len(X)
        for i in range(N):
            dw=dw+1/N*np.exp(-y[i]*w.T@X[i,:])/(1+np.exp(-y[i]*w.T@X[i,:]))*(-y[i]*X[i,:])
        return dw
    
    fs=np.zeros(Niter)
    
    
    for _ in range(Nsim):
        w=np.zeros(d) #d=100
        g=np.zeros((N,d)) #gradient for every sample
        g=np.tile(df(w,X,y), (N, 1))

        for t in range(Niter):
            ind=np.random.choice(range(1000), size=b, replace=True)
            dw=df(w,X[ind],y[ind])
            G=dw-np.mean(g[ind,:],axis=0)+np.mean(g,axis=0)
            w-=eta*G
            g[ind,:]=dw
            #g=dw
            fs[t]+=f(w)/Nsim
    return fs


# In[100]:


bs_SAGA=[1,10,100]
F_SAGA=np.zeros((3,500))

i=0
for b in bs_SAGA:
    f=SAGA(b)
    F_SAGA[i,:]=f
    i+=1


# In[101]:


#(3.2.1)
f500SAGA=F_SAGA[:,-1]
print('f_SAGA_hat(500) with eta=0.1:')
for i in range(3):
    print('b={}: {}'.format(bs_SAGA[i],f500SAGA[i].round(6)))


# In[102]:


#(3.2.2)
f500_eta01=F[[2,6,10],:]

for i in range(3):
    b=bs_SAGA[i]
    plt.figure(figsize=(16,12))
    plt.plot(list(range(Niter)),f500_eta01[i]-f_star,label='SGD')
    plt.plot(list(range(Niter)),F_SAGA[i,:]-f_star,label='SAGA')
    plt.xlabel('Iteration t')
    plt.ylabel('log(f(w_t)-f_min)')
    plt.yscale('log')
    plt.title('eta=0.1,b={}'.format(b))
    plt.legend()
    plt.show()


# (3.2.3)
# 
# When batch size is small, SAGA decreases the function value slower than SGD possibly because it reduced the variance of gradient. When batch size is large, SAGA converges to SGD because there's little difference between the memory g and actual gradient since a large portion of the dataset gets gradient update.

# In[ ]:




