'''
Comparing the impacts of weight matrices with different condition numb
Generate a random matrix
Generate two weight matrix with different kappa
Fix these two weight matrices during calculation
w1 has a smaller kappa, while w2 has a larger one
'''
import numpy as np
from numpy.random import normal,multivariate_normal

import scipy.io as io

import os 
import pickle

from Reweight_accessories import byebye,Alter_OMF,Alter_OMF2,OMF_Pu

dir_path = os.getcwd()
ver_out = 9
fin = dir_path+'/testdata/testCondNumb_v4.pickle'

test = True

if test:
	fin = dir_path+'/testdata/testCondNumb_v00'+str(ver_out)+'.pickle'

try:
	with open(fin,'rb') as f:
		data = pickle.load(f)
		print('Successfully load data from %s!' % fin)
		#print('Data info: %s. \n' % str(data['__header__']))
		M = data['M']
		w1 = data['w1']
		w2 = data['w2']
		delta_err_min = data['delta_err_min']
		iter_max = data['iter_max']
		k = data['k']

		ver_out += 1
		#fout_mat = dir_path+'/testdata/testCondNumb_v'+str(ver_out)+'.mat'
		#fout = dir_path+'/testdata/testCondNumb_v'+str(ver_out)+'.pickle'

		output = {'fin':fin}
		
		output.updat({'w1':data['w1'], 'w2':data['w2'], 'M':data['M'],'kappa1':data['kappa1'],'kappa2':data['kappa2']})

except Exception as e_open:
	print('Err in load pickle data files and infor is: ', str(e_open))
	print('\n')
	print('We now create the data by random generation')
	print('\n')

	dim = 3
	numb = 1000
	mu1 = 2
	sigma1 = 0.5

	mu2 = 2
	sigma2 = 1

	w1 = np.abs(normal(mu1,sigma1,size = (1,numb)))
	kappa1 = np.amax(w1)/np.amin(w1)
	w2 = np.abs(normal(mu2,sigma2,size = (1,numb)))
	kappa2 = np.amax(w2)/np.amin(w2)

	mu = np.zeros(dim)
	sigma = np.identity(dim)
	M = np.abs(multivariate_normal(mu,sigma,numb).T)

	output = {'w1':w1, 'w2':w2, 'M':M,'kappa1':kappa1,'kappa2':kappa2}

	delta_err_min = 1.e-4
	iter_max = 500
	k =2 

	output.update({'k':k,'delta_err_min':delta_err_min,'iter_max':iter_max})

	# with open(fout,'wb') as f:
	# 	try:
	# 		pickle.dump(output,f)
	# 	except Exception as e_pickle:
	# 		print('Error in saving created data in pickle! The infor is: ', str(e_pickle))
	# 		print('\n')
	# 		byebye('Creating data file failed!')

fout_mat = dir_path+'/testdata/testCondNumb_v'+str(ver_out)+'.mat'
fout = dir_path+'/testdata/testCondNumb_v'+str(ver_out)+'.pickle'

M1 = M*w1
M2 = M*w2

dim,_ = M.shape

output.update({'M1':M1,'M2':M2})
#U, S, V = np.linalg.svd(M,full_matrices=False)
#X0 = U[:,:k]
#S0 = S[:k]
#Y0 = V[:k,:]*S0[None,:].T
mu = np.zeros(dim)
sigma = np.identity(dim)
X0 = multivariate_normal(mu,sigma,k).T
#X0 = np.identity(dim)[:,:k]

sum_col_X = np.sqrt(np.sum(np.square(X0),axis=0))
X0 = X0/sum_col_X

X01 = X0

sum_col_X = np.sqrt(np.sum(np.square(X0),axis=0))
X0 = X0/sum_col_X
X02 = X0


'''
U1, S1, V1 = np.linalg.svd(M1,full_matrices=False)
X01 = U1[:,:k]
S01 = S1[:k]
Y01 = V1[:k,:]*S01[None,:].T

U2, S2, V2 = np.linalg.svd(M2,full_matrices=False)
X02 = U2[:,:k]
S02 = S2[:k]
Y02 = V2[:k,:]*S02[None,:].T
'''

#X1,Y1,err1,err_list1 = Alter_OMF2(M1,X01,delta_err_min,iter_max)
#output.update({'X1':X1,'Y1':Y1,'err1':err1,'err_list1':err_list1})
X1,err1,err_list1,cn_list1 = OMF_Pu(M1,X01,delta_err_min,iter_max)
output.update({'X1':X1,'err1':err1,'err_list1':err_list1,'cn_list1':cn_list1})

#X2,Y2,err2,err_list2 = Alter_OMF2(M2,X01,delta_err_min,iter_max)
#output.update({'X2':X2,'Y2':Y2,'err2':err2,'err_list2':err_list2})
X2,err2,err_list2,cn_list2 = OMF_Pu(M2,X02,delta_err_min,iter_max)
output.update({'X2':X2,'err2':err2,'err_list2':err_list2,'cn_list2':cn_list2})


try:
	io.savemat(fout_mat,output)
except Exception as e_savemat:
	print('Error in saving mat! The infor is: ', str(e_savemat))
	print('\n')

with open(fout,'wb') as f:
	try:
		pickle.dump(output,f)
	except Exception as e_pickle:
		print('Error in saving pickle! The infor is: ', str(e_pickle))
		print('\n')






