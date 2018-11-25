# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal

#Einstellbar
n=2 #Anzahl der Stichproben
alpha=2
beta=1

maxXvalue=1#Es werden Stichproben gezogen in der Region [0,maxXvalue]
minXvalue=-maxXvalue


#Definitionen
test_range=np.linspace(minXvalue,maxXvalue,100)
sample_range=(npr.rand(n)*2)-1
#sample_range=np.linspace(minXvalue,maxXvalue,n)
#sample_range=[-0.2,0,0.2]

#generiere Funktion f(x,a)

coefficients=[1,1,0]

def f_x(x):
    return -0.3+0.5*x+0.2*np.random.normal(mu,sigma,1)[0]
 
##Generiere noisebehaftete Datensätze
#npr.rand(1)[0]*2)-1
mu=0.0
sigma=1
D=[f_x(x) for x in sample_range]  #Der Ausdruck     (sigma*npr.rand(1)+mu)[0]*2-1 ist unzureichend

##Generierung unserer Polynomkoeffizienten
def phi_i(x,n):
    return x**n

#Matrix groß Phi
Phi=np.array([[phi_i(x,k) for k in range(len(coefficients))] for x in sample_range]) 


#Matrix groß Sigma

SIGMA_n=npl.inv((1/(alpha))*np.identity(len(coefficients))+beta*np.transpose(Phi).dot(Phi))
MEAN_n=beta*(SIGMA_n.dot(np.transpose(Phi).dot(D)))

##Generiere Prior Verteilung


def calculate_prior(amount_of_ws):
    w_prior=[]
    for i in range(amount_of_ws):
        w_prior.append(np.random.normal(0,2,1)[0])
    return w_prior
    
w_probabilities=calculate_prior(len(coefficients)) #w_probabilities[0] steht für Koeffizient w0 usw.


#def calculate_p(mu, cov, grid):
#    return np.random.multivariate_normal(mean,cov

 


#sample_coefficients1=[0.1,-0.1,1]

#sample_coefficients2=[-1,0.7,1.3]
#sample_coefficients3=[1.3,-0.6,0.04]
#sample_coefficients4=[0.02,0.3,-0.9]
#testvalues1=[f_x(x,sample_coefficients1) for x in test_range]
#testvalues2=[f_x(x,sample_coefficients2) for x in test_range]
#testvalues3=[f_x(x,sample_coefficients3) for x in test_range]
#testvalues4=[f_x(x,sample_coefficients4) for x in test_range]
#sample_plots1, = plt.plot(test_range, testvalues1, label='data space')
#sample_plots2, = plt.plot(test_range, testvalues2, label='data space')
#sample_plots3, = plt.plot(test_range, testvalues3, label='data space')
#sample_plots4, = plt.plot(test_range, testvalues4, label='data space')

fvalues=[f_x(x) for x in sample_range]

#regression_plot, = plt.plot(test_range, Dvalues, label='Regressionsfunktion 20ten Grades (overfitting)')
#sample_plot, = plt.plot(sample_range, D_list[0], label='Datenpunkte')#zeigt die Datenpunkte (Stichproben) ohne fitting an
#true_plot, = plt.plot(test_range, fvalues, label='Tatsaechlicher Funktionsverlauf')
#sample_plot, = plt.plot(sample_range, D, label='Samples Funktionsverlauf')
#plt.legend(handles=[true_plot])
#plt.xticks(np.arange(minXvalue, maxXvalue+1, 1))
#plt.yticks(np.arange(0, math.ceil(max(fvalues)), 1))
plt.show()

##PLOTTING

def plotheatmap(w_0,w_1,w_2,s):
    x, y = np.mgrid[-5:5:.01, -5:5:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    plt.subplot(s)
    radius=[]
    square=[]
    plt.axes().set_aspect('equal', 'box')
    if (w_0==1 and w_1==1 and w_2==0):
        #für Variablen w0, w1
        rv = multivariate_normal(MEAN_n[:2], SIGMA_n[0:2,0:2])
        #Plotten der tatsächlichen Koeffizienten
        #w0,w1
        radius = [coefficients[0]]
        square = [coefficients[1]]
        plt.xlabel('w0')        
        plt.ylabel('w1')
    elif (w_0==0 and w_1==1 and w_2==1):
        #für Variablen w1, w2
        rv = multivariate_normal(MEAN_n[1:3], SIGMA_n[1:3,1:3])
        #Plotten der tatsächlichen Koeffizienten
        #w0,w1
        radius = [coefficients[1]]
        square = [coefficients[2]]
        plt.xlabel('w1')        
        plt.ylabel('w2')
    area = [3.14159]
        
    plt.title('posterior')
    plt.plot(radius, square, marker='+', linestyle='--', color='r', label='Square')
    plt.contourf(x, y, rv.pdf(pos))
    #plt.savefig('C:/Users/Tristan_local/OneDrive/myData/Machine Learning/Blatt 6/heat_.png', transparent=True)

    #plt.axis('box')
    plt.show()

def plotdatapoints(s):
    plt.subplot(s)
    plt.axes().set_aspect('equal', 'box')
    plt.title('data')
    plt.xlabel('x')        
    plt.ylabel('t')
    radius=D
    square=sample_range
    if (len(D)>=4 or len(radius)>=3):
        print(radius)
    
    plt.plot(square,radius, marker='o', linestyle='', color='r', label='Square')
    plt.axis('equal')
    #plt.savefig('C:/Users/Tristan_local/OneDrive/myData/Machine Learning/Blatt 6/data.png', transparent=True)
    plt.show()
    print("HEY")
    
def plotsamplegraphs(sample_coefficients):
    plt.axes().set_aspect('equal', 'box')
    plt.title('data')
    plt.xlabel('x')        
    plt.ylabel('t')

    for i in range(len(sample_coefficients)):
        testvalues=[f_x(x) for x in test_range]
        plt.plot(test_range, testvalues, label='data space')
    plt.show()

    plt.axis('equal')
    #plt.savefig('C:/Users/Tristan_local/OneDrive/myData/Machine Learning/Blatt 6/samplegraphs.png', transparent=True)
    
        
sample_coefficients1=[1,1,2]
sample_coefficients2=[1.1,1.1,1.8]
sample_coefficients3=[0.96,0.93,1.9]
sample_coefficients4=[1.07,1,1.78]

testcoefficients=[sample_coefficients1,sample_coefficients2,sample_coefficients3,sample_coefficients4]

      
plotheatmap(1,1,0,221)
#plotheatmap(0,1,1,222)
#plotdatapoints(223)
#plotsamplegraphs(testcoefficients)

