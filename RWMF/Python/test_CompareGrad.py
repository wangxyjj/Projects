import numpy as np
from numpy.random import multivariate_normal
#import math

import scipy.io as io

import os 
import pickle

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from Reweight_accessories import CompareGrad, residue, weight


# fix the random seed for replicability.
dir_path = os.getcwd()
version = 2
file_name = 'testrun_v'
file_output = dir_path+'/testdata/'+file_name+str(version)+'.mat'
fout = dir_path+'/testdata/'+file_name+str(version)+'.pickle'

test = True

if test:
    data_path = '/Users/XiangyuW/Google Drive/Research2015/ReweightMF/Code/Python/testdata/'
    file_input = data_path+'syth_Gaussian_100_1.mat'
    #file_input = data_path+ 'data_RWLR_train_3000_1.mat'
    data = io.loadmat(file_input)
    M = data['M']
    m,n = M.shape
    if 'k' not in data: 
        k = 2 # low rank approximation 
    else:
        k = int(data['k'])
        
else:   
    np.random.seed(123456789)   
    # random generate data by normal distribution
    m = 3   # dimension of the given data
    n = 10  # number of the given data
    k = 2   # low rank approximation 
    mu = np.zeros(m)
    sigma = np.identity(m)
    M = multivariate_normal(mu,sigma,n).T

damping = 0.001


output={'M':M,'damping':damping,'k':k,'dim':m,'numb_points':n}

print(M.shape)
# Ms = M[:,:5]
# 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# 
# ax.scatter(M[0,:], M[1,:], M[2,:], c='r', marker='o')
# ax.scatter(Ms[0,:], Ms[1,:], Ms[2,:],c='b')
# 
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# 
# plt.show()

U, S, V = np.linalg.svd(M,full_matrices=False)
#print("size of U is", U.shape, "size of S is", S.shape,"size of V is", V.shape)
print('size of U is %s;\n size of S is %s;\n size of V is %s.\n ' % (str(U.shape),str(S.shape),str(V.shape)))

X0 = U[:,:k]
Y0T = np.dot(X0.T,M)
W0 = weight(M,X0,damping)
print('The initial weight is',W0)
err0, Mk0 = residue(M,X0,Y0T,W0)
print('The initial residue is', err0)

J1, J2 = CompareGrad(M,X0,err0,damping,50)

output.update({'X0':X0,'J1':J1,'J2':J2})

try:
    io.savemat(file_output,output)
except Exception as e_savemat:
    print('Error in saving mat! The infor is: ', str(e_savemat))
    print('\n')
    
with open(fout,'wb') as f:
    try:
        pickle.dump(output,f)
    except Exception as e_pickle:
        print('Error in saving pickle! The infor is: ', str(e_pickle))
        print('\n')

