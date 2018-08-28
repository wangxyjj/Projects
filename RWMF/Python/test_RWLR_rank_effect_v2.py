# Version 2: Aug-20
# Revision:
#			1) Add Logging
# test the RWLR model in different ranks 
from RWLSTM_lib import IO_Data
import numpy as np
from numpy import random 
import math
from RWLR_support import RWLR_Model,get_initU, normal_col_l2,RWLR_Model_Betta,Model_Controlling
from RWLR_support import RWLR_Model_Theta
from Reweight_accessories import residue,weight

from My_Logger import Logger

import matplotlib.pyplot as plt

"""
 IO setup
"""

ver = 1

test_data_IO = IO_Data(version = ver,fout_name = 'data_RWLR_v')
dir_path = '/Users/XiangyuW/Google Drive/Research2015/ReweightMF/Code/Python/testdata/'

logPath = dir_path+"/logs/"
log_obj = Logger(logPath)
logger = log_obj.logger


#fin_mat = '/Users/XiangyuW/Google Drive/Research2015/SNMFUKmeans/Code/USPSHandwritten/HandW_28x28.mat'
#fin_mat = '/Users/XiangyuW/Google Drive/Research2015/ReweightMF/Code/Python/testdata/data_RWLR_train_3000.mat'
#dataset_name = 'Handwritten'

#random.seed(123456789)

random_pick = False
svd_initial = True

if random_pick:
	#dataset_name = 'Handwritten'
	#dataset_name = 'tf_minist'
	dataset_name = 'YaleFace'
	#fin_mat = '/Users/XiangyuW/Google Drive/Research2015/SNMFUKmeans/Code/USPSHandwritten/HandW_28x28.mat'
	fin_mat = '/Users/XiangyuW/Google Drive/Research2015/SNMFUKmeans/Code/USPSHandwritten/face_19x19.mat'
	raw_data_train,raw_label_train,raw_data_test,raw_label_test = test_data_IO.use_dataset(dataset_name,fin_mat,test_portion = 0.1)
	data = raw_data_train.T
	output = {'fea':raw_data_train,'gnd':raw_label_train}
else:
#	input_path = '/Users/XiangyuW/Google Drive/Research2015/ReweightMF/Code/Python/testdata/input_datasets/'
	#fin_name = 'MINIST_55000_train.mat'
	#fin_name = 'data_RWLR_train_3000'
	#fin_name = 'MINIST11000'
	fin_name = 'YaleFace1.pickle'
	
	#dataset_name = 'Handwritten'	
	dataset_name = 'YaleFace'
	#dataset_name = 'MINIST11000'
	
	input_path = '/Users/XiangyuW/Google Drive/Research2015/ReweightMF/Code/Python/testdata/input_datasets/'
	#fin_name = 'MINIST_55000_train.mat'
	
	fin_mat = input_path+fin_name
	raw_data = test_data_IO.create_data_hw(fin_mat)
	data = raw_data[0].T
	output ={}
	
#data = data[:,:10000]

"""
 Set parameters
"""
k0 = 15**2
dim,n_points = data.shape

w_kernel = 'TruthVector'
par_kernel = 'PartialTruthVector'

n_minibatch = 10
init_portion = 0.3

output.update({
			'dim':dim, 
			'n_points':n_points, 
			'n_minibatch':n_minibatch,
			'weight_kernel':w_kernel,
			'partial_kernel':par_kernel,
			'init_portion':init_portion
			})

"""
 Define model
"""
max_iters = 100

#random initialization
mu = np.zeros(dim)
sigma = np.identity(dim)
U0 = random.multivariate_normal(mu,sigma,k0).T
U0 = normal_col_l2(U0)
idexes0 = np.arange(n_points)

#test_model = RWLR_Model(data, U0, idexes0, delta_err_min =0.001,n_iters=max_iters)

parameters = Model_Controlling(momentum = True,momentum_para = (0.85,0.15))
#test_model_betta = RWLR_Model_Betta(data, U0, idexes0,betta_parameters,delta_err_min =0.001,n_iters=max_iters)
test_model_theta = RWLR_Model_Theta(data, U0, idexes0,parameters,delta_err_min =0.001,n_iters=max_iters,logger = logger)

#~~~~~~~~~~~~~~~~~~~~~~~~I'm a beautiful separate line~~~~~~~~~~~~~~~~~

"""
 Run test
"""
seed = 123

rank_input = [12, 12, 12, 12]
#rank_input = [5, 10, 15, 20]
#rank_input = [15,15,15,15]
test_rank = [x**2 for x in rank_input]
test_model_theta.n_iters = 30
n_inner_loop = 30
version = 18.1

dirpath_test_data = dir_path+'archive_mat_data/'
f_data_name = dataset_name + '_ARWLR_AGD_ALM_rank_test_v'
test_data_IO.update_fout(f_data_name, path = dirpath_test_data,v = version)
#test_data_IO.update_fout('FACE_ARWLR_AGD_ALM_rank_test_v', v = version)
residues_list = [] 
weights_list = []

dirpath_fig = dir_path+'output_images/'
fig_rank_test = plt.figure()
colormap = ['k', 'r', 'g', 'b','c', 'm', 'y', 'w']
linemarker = '-+'

for i_test in range(len(test_rank)):
	
	"""
		Initialization
	"""
	k = test_rank[i_test]
	logger.info('\n***********Begins test no. %s.' % str(i_test+1))
	logger.info('The projected low rank is %d.' % k)
	
	init_portion = 0.1*(2**i_test)
	#init_portion = 0.1
	U0,idexes0,_ = get_initU(data,init_portion,k,random_seed=seed)
	test_model_theta.update_intials(U0,idexes0)

	"""
		Run test
	"""
	# 2nd Accelerated ALM FGD	

	#U_update_method = 'Critical'
	U_update_method = '1st'
	
	U1,g1,err_list1,g_list1 = test_model_theta.apply_RW_AGD_NC(w_kernel, par_kernel,U_update_method,n_inner_loop,oscillation_control = 3)
	label1 = U_update_method+' Accelerate Basis'

	U2,g2,err_list2,g_list2 = test_model_theta.apply_RW_AGD_NC_Beta(w_kernel, par_kernel,U_update_method,n_inner_loop,oscillation_control = 3)
	label2 = U_update_method+' Accelerate Basis and Weights'

	#U2,g3,err_list3,g_list3 = test_model_theta.apply_RW_AGD_NC3(w_kernel, par_kernel,U_update_method,n_inner_loop)
	#label3 = U_update_method+' Accelerate Weights Plus'


	residues_list.append([err_list1,err_list2])
	weights_list.append([g_list1,g_list2])

	try:
		print('Plot Fig no.', i_test+1)
		if i_test > 3:
			raise Exception('Number of test exceeds the number of  subfigures!')
		ax_err = fig_rank_test.add_subplot(2,2,i_test+1)
		ax_err.plot(np.array(err_list1),'k-+',label=label1)
		ax_err.plot(np.array(err_list2),'r-+',label=label2)
		#ax_err.plot(np.array(err_list3),'b-+',label=label3)
		#title = 'Residue Comparison when rank = '+ str(k)
		title = 'Rank '+str(k)+' Residue Comparison when init_portion = '+ str(init_portion)
		ax_err.set_title(title)
		ax_err.legend()
	except Exception as e_fig:
		print('Fail to plot subfig '+str(i_test+1)+'!')
		print('The error info is ', str(e_fig))
		print('\n')

f_fig_name = dirpath_fig+dataset_name+U_update_method+'_AGD_ALM_rank_test_v'+str(version)+'.png'	
plt.savefig(f_fig_name)
plt.show()

output.update({'resiudes_list':residues_list,
				'weights_list':weights_list,
				'fout_name':test_data_IO.fout})
test_data_IO.save_data(output)

logger.info('\n~~~~~~~~~~~~Finished!~~~~~~~~~~~~~')