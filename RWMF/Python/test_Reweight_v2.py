import numpy as np
from numpy import random 
from RWLR_support import get_initU, normal_col_l2

from Reweight_accessories import residue, weight, ReweightMF, ReweightMF_adjust
#from Reweight_accessories import draw_graph3D_v1
#from Reweight_accessories import checkPSD

from RWLSTM_lib import IO_Data

if __name__ == '__main__':

	ver = 1

	test_data_IO = IO_Data(version = ver,fout_name = 'data_Reweight_2nd_v')

	#fin_mat = '/Users/XiangyuW/Google Drive/Research2015/SNMFUKmeans/Code/USPSHandwritten/HandW_28x28.mat'
	#fin_mat = '/Users/XiangyuW/Google Drive/Research2015/ReweightMF/Code/Python/testdata/data_RWLR_train_3000.mat'
	#dataset_name = 'Handwritten'

	#random.seed(123456789)

	random_pick = False
	svd_initial = True

	if random_pick:
		dataset_name = 'Handwritten'
		#dataset_name = 'tf_minist'
		fin_mat = '/Users/XiangyuW/Google Drive/Research2015/SNMFUKmeans/Code/USPSHandwritten/HandW_28x28.mat'
		raw_data_train,raw_label_train,raw_data_test,raw_label_test = test_data_IO.use_dataset(dataset_name,fin_mat)
		M = raw_data_train.T
		output = {'fea':raw_data_train,'gnd':raw_label_train}
	else:
		dir_path = '/Users/XiangyuW/Google Drive/Research2015/ReweightMF/Code/Python/testdata/'
		#fin_name = 'MINIST_55000_train.mat'
		fin_name = 'data_RWLR_train_3000'
		fin_mat = dir_path+fin_name
		raw_data = test_data_IO.create_data_hw(fin_mat)
		M = raw_data[0].T
		output ={}
		
	#data = data[:,:10000]

	dim,n_points = M.shape

	k = 7**2


	print('\nThe projected low rank is ',k)

	init_portion = 0.8
	damping = 0.001
	delta_converged = 1.e-3
	iter_max = 500
	
	output.update({'damping':damping, 'k':k, 'dim':dim, 'numb_points':n_points})

	if svd_initial:
		X0,idexes0,_ = get_initU(M,init_portion,k,random_seed=123456)
	else:
		#random.seed(1346)
		mu = np.zeros(dim)
		sigma = np.identity(dim)
		X0 = random.multivariate_normal(mu,sigma,k).T
		X0 = normal_col_l2(X0)
		idexes0 = np.arange(n_points)

	
	Y0T = np.dot(X0.T,M)
	W0 = weight(M,X0,damping)
	print('The initial weight is',W0)
	err0, Mk0 = residue(M,X0,Y0T,W0)
	print('The initial residue is', err0)

	X1, W1,grad,Mk1, err1 = ReweightMF_adjust(M,X0,err0,damping,delta_converged,iter_max)
	output.update({ 'U0': X0,
				'U' : X1,
				'g' : W1,
				'err_list':err1				
				})
	
	test_data_IO.save_data(output)
	
	test_data_IO.update_fout('data_Reweight_2nd_Classic_v',v= ver)
	X2, W2, Mk2, err2 = ReweightMF(M,X0,err0,damping,delta_converged,iter_max)
	output.update({ 'U0': X0,
				'U' : X2,
				'g' : W2,
				'err_list':err2				
				})
	#PSD_flag, Hess = checkPSD(M,X2,grad,damping)
	
	#output.update({'PSD_flag':PSD_flag,'Hess':Hess})

	#print('Not a saddle point?',PSD_flag,Hess)

	test_data_IO.save_data(output)
		

	#draw_graph3D_v1(M,Mk,Mk0,X,X0,W,W0)
	










