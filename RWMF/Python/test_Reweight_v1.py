import numpy as np
from numpy.random import multivariate_normal

import scipy.io as io

import os 
import pickle

from Reweight_accessories import residue, weight, ReweightMF, ReweightMF_adjust
#from Reweight_accessories import draw_graph3D_v1
from Reweight_accessories import checkPSD

if __name__ == '__main__':

	dir_path = os.getcwd()
	version = 5
	file_output = dir_path+'/testdata/output_v'+str(version)+'.mat'
	fout = dir_path+'/testdata/output_v'+str(version)+'.pickle'

	test = True

	if test:
		data_path = '/Users/XiangyuW/Google Drive/Research2015/ReweightMF/Code/Python/testdata/'
		#file_input = data_path+'syth_Gaussian_100_1.mat'
		file_input = data_path+ 'data_RWLR_train_3000_1.mat'
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
		m = 3	# dimension of the given data
		n = 10	# number of the given data
		k = 2   # low rank approximation 
		mu = np.zeros(m)
		sigma = np.identity(m)
		M = multivariate_normal(mu,sigma,n).T

	damping = 0.001
	delta_converged = 1.e-4
	iter_max = 500
	
	output = {'damping':damping, 'k':k, 'dim':m, 'numb_points':n}

	U, S, V = np.linalg.svd(M,full_matrices=False)
	#print("size of U is", U.shape, "size of S is", S.shape,"size of V is", V.shape)
	print('size of M is %s;\n size of U is %s;\n size of S is %s;\n size of V is %s.\n ' \
	      % (str(M.shape), str(U.shape),str(S.shape),str(V.shape)))

	X0 = U[:,:k]
	Y0T = np.dot(X0.T,M)
	W0 = weight(M,X0,damping)
	print('The initial weight is',W0)
	err0, Mk0 = residue(M,X0,Y0T,W0)
	print('The initial residue is', err0)

	X1, W1,grad,Mk1, err1 = ReweightMF_adjust(M,X0,err0,damping,delta_converged,iter_max)
	X2, W2, Mk2, err2 = ReweightMF(M,X0,err0,damping,delta_converged,iter_max)
	

	output.update({'M':M, 'X1':X1, 'err_list1':err1, 'Mk1': Mk1})
	output.update({'X2':X2,'W2':W2,'grad':grad,'err_list2':err2, 'Mk2': Mk2})

	PSD_flag, Hess = checkPSD(M,X2,grad,damping)
	
	output.update({'PSD_flag':PSD_flag,'Hess':Hess})

	print('Not a saddle point?',PSD_flag,Hess)

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
		

	#draw_graph3D_v1(M,Mk,Mk0,X,X0,W,W0)
	










