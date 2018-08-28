# test the RWLR model 
from RWLSTM_lib import IO_Data
import numpy as np
from numpy import random 
import math
from RWLR_support import RWLR_Model,get_initU, normal_col_l2,RWLR_Model_Betta,Model_Controlling
from RWLR_support import RWLR_Model_Theta
from Reweight_accessories import residue,weight

import matplotlib.pyplot as plt


"""
 IO setup
"""
ver = 1

test_data_IO = IO_Data(version = ver,fout_name = 'data_RWLR_Basic_v')
dir_path = '/Users/XiangyuW/Google Drive/Research2015/ReweightMF/Code/Python/testdata/'

#fin_mat = '/Users/XiangyuW/Google\ Drive/Research2015/SNMF_clustering/SNMFUKmeans/Code/USPSHandwritten/HandW_28x28.mat'
#fin_mat = '/Users/XiangyuW/Google\ Drive/Research2015/ReweightMF/Code/Python/testdata/data_RWLR_train_3000.mat'
#dataset_name = 'Handwritten'

#random.seed(123456789)

random_pick = False
svd_initial = True

if random_pick:
	#dataset_name = 'Handwritten'
	dataset_name = 'MINIST'
	#dataset_name = 'YaleFace'
	#fin_mat = '/Users/XiangyuW/Google\ Drive/Research2015/SNMF_clustering/SNMFUKmeans/Code/USPSHandwritten/HandW_28x28.mat'
	#fin_mat = '/Users/XiangyuW/Google Drive/Research2015/SNMF_clustering/SNMFUKmeans/Code/facedata/face_19x19.mat'
	fin_mat = '/Users/XiangyuW/Google Drive/Research2015/ReweightMF/Code/Python/testdata/input_datasets/MINIST_55000_train.mat'
	
	raw_data_train,raw_label_train,raw_data_test,raw_label_test = test_data_IO.use_dataset(dataset_name,fin_mat,test_portion = 0.8)
	data = raw_data_train.T
	
	output = {'fea':raw_data_train,'gnd':raw_label_train}
	test_data_IO.update_fout(dataset_name, v = 11000)
	test_data_IO.save_data(output,need_pickle = True)
	output = {}
else:
	input_path = '/Users/XiangyuW/Google Drive/Research2015/ReweightMF/Code/Python/testdata/input_datasets/'
	fin_name = 'data_RWLR_train_3000'
	#fin_name = 'MINIST_55000_train.mat'
	#fin_name = 'MINIST5500'
	dataset_name = 'Handwritten'
	#dataset_name = 'MINIST5500'
	#dataset_name = 'YaleFace'
	#fin_name = 'YaleFace1'
	fin_mat = input_path+fin_name
	raw_data = test_data_IO.create_data_hw(fin_mat)
	data = raw_data[0].T
	output ={}
	
#data = data[:,:10000]

"""
 Set parameters
"""

dim,n_points = data.shape

w_kernel = 'TruthVector'
par_kernel = 'PartialTruthVector'

k = 15**2
print('\nThe projected low rank is ',k)
n_minibatch = 10
init_portion = 0.3
r_seed = 1234567

output.update({
			'k':k, 'dim':dim, 
			'n_points':n_points, 
			'n_minibatch':n_minibatch,
			'weight_kernel':w_kernel,
			'partial_kernel':par_kernel,
			'init_portion':init_portion
			})

"""
 Initialization
"""

if svd_initial:
	U0,idexes0,_ = get_initU(data,init_portion,k,random_seed = r_seed)
else:
	#random.seed(1346)
	mu = np.zeros(dim)
	sigma = np.identity(dim)
	U0 = random.multivariate_normal(mu,sigma,k).T
	U0 = normal_col_l2(U0)
	idexes0 = np.arange(n_points)
	#W0 = weight(data,U0,0.001)
	#err0, Mk0 = residue(data,U0,np.dot(U0.T,data),W0)

"""
 Define model
"""
max_iters = 100

#test_model = RWLR_Model(data, U0, idexes0, delta_err_min =0.001,n_iters=max_iters)

parameters = Model_Controlling(momentum = True,momentum_para = (0.85,0.15))
#test_model_betta = RWLR_Model_Betta(data, U0, idexes0,betta_parameters,delta_err_min =0.001,n_iters=max_iters)
test_model_theta = RWLR_Model_Theta(data, U0, idexes0,parameters,delta_err_min =0.001,n_iters=max_iters)

#~~~~~~~~~~~~~~~~~~~~~~~~I'm a beautiful separate line~~~~~~~~~~~~~~~~~

"""
 Run test
"""

dirpath_test_data = dir_path+'archive_mat_data/'

#*****************
# Plot output setting up
version = 2.1
residues_plot_list = []
w_attributes_plot_list = []
legend_list = []
#*****************


# 1st Accelerated ALM FGD
test_model_theta.n_iters = 30
n_inner_loop = 30
f_data_name = dataset_name+'_ARWLR_1stAGD_ALM_v'
test_data_IO.update_fout(f_data_name, path = dirpath_test_data,v = version)

U_update_method = '1st'
U,g,err_list,g_list = test_model_theta.apply_RW_AGD_NC(w_kernel, par_kernel,U_update_method,n_inner_loop)
output.update({ 'U0': U0,
				'U' : U,
				'g' : g,
				'err_list':err_list,
				'g_list':g_list, 
				'fout_name':test_data_IO.fout
				})
test_data_IO.save_data(output)

residues_plot_list.append(err_list)
w_attributes_plot_list.append(g_list)

label_err = '1st Accelerate Basis'
legend_list.append(label_err)
#*****************************************

'''
# 2nd Accelerated ALM FGD
test_model_theta.n_iters = 30
n_inner_loop = 30
f_data_name = dataset_name+'_ARWLR_2ndAGD_ALM_v'
test_data_IO.update_fout(f_data_name, path = dirpath_test_data,v = version)

U_update_method = 'Critical'
U,g,err_list,g_list = test_model_theta.apply_RW_AGD_NC(w_kernel, par_kernel,U_update_method,n_inner_loop)
output.update({ 'U0': U0,
				'U' : U,
				'g' : g,
				'err_list':err_list,
				'g_list':g_list, 
				'fout_name':test_data_IO.fout
				})
test_data_IO.save_data(output)

residues_plot_list.append(err_list)
w_attributes_plot_list.append(g_list)

label_err = '2nd Accelerate Basis'
legend_list.append(label_err)
#*****************************************
'''


# 1st Accelerated ALM FGD Beta
test_model_theta.n_iters = 30
n_inner_loop = 30
f_data_name = dataset_name+'_ARWLR_1stAGD_ALM_Beta_v'
test_data_IO.update_fout(f_data_name, path = dirpath_test_data,v = version)
#test_data_IO.update_fout('FACE_ARWLR_AGD_ALM_Beta_v', v = version)
U_update_method = '1st'
U,g,err_list,g_list = test_model_theta.apply_RW_AGD_NC_Beta(w_kernel, par_kernel,U_update_method,n_inner_loop)
output.update({ 'U0': U0,
				'U' : U,
				'g' : g,
				'err_list':err_list,
				'g_list':g_list, 
				'fout_name':test_data_IO.fout
				})
test_data_IO.save_data(output)

residues_plot_list.append(err_list)
w_attributes_plot_list.append(g_list)

label_err = '1st Accelerate Weights and Basis'
legend_list.append(label_err)
#*****************************************

'''
# 2nd Accelerated ALM FGD Beta
test_model_theta.n_iters = 30
n_inner_loop = 30
f_data_name = dataset_name+'_ARWLR_2ndAGD_ALM_Beta_v'
test_data_IO.update_fout(f_data_name, path = dirpath_test_data,v = version)

#test_data_IO.update_fout('FACE_ARWLR_AGD_ALM_Beta_v', v = version)
U_update_method = 'Critical'
U,g,err_list,g_list = test_model_theta.apply_RW_AGD_NC_Beta(w_kernel, par_kernel,U_update_method,n_inner_loop)
output.update({ 'U0': U0,
				'U' : U,
				'g' : g,
				'err_list':err_list,
				'g_list':g_list, 
				'fout_name':test_data_IO.fout
				})
test_data_IO.save_data(output)

residues_plot_list.append(err_list)
w_attributes_plot_list.append(g_list)

label_err = '2nd Accelerate Weights and Basis'
legend_list.append(label_err)
#*****************************************
'''


# Classic ALM 
test_model_theta.n_iters = 30
n_inner_loop = 30
f_data_name = dataset_name+'_ARWLR_ALM_v'
test_data_IO.update_fout(f_data_name, path = dirpath_test_data,v = version)
#test_data_IO.update_fout('FACE_ARWLR_ALM_v', v = version)
U,g,err_list,g_list = test_model_theta.apply_RW_ALM_FGD(w_kernel, par_kernel,n_inner_loop)
output.update({ 'U0': U0,
				'U' : U,
				'g' : g,
				'err_list':err_list,
				'g_list':g_list, 
				'fout_name':test_data_IO.fout
				})

test_data_IO.save_data(output)

residues_plot_list.append(err_list)
w_attributes_plot_list.append(g_list)

label_err = 'Classic ALM '
legend_list.append(label_err)
#*****************************************


#~~~~~~~~~~~~~~~~~~~~~~~~I'm a beautiful separate line~~~~~~~~~~~~~~~~~

"""
 Plot output
"""

try:
	print('\n Plot Output Figs.')
	
	dirpath_fig = dir_path+'output_images/'
	
	f_fig_name = dirpath_fig+dataset_name+'_ConvergeComp_AGD_ALM_v'+str(version)+'.png'
	fig_err = plt.figure()
	

	colormap = ['k', 'r', 'g', 'b','c', 'm', 'y', 'w']
	linemarker = '-+'
	legend_loc = 'upper right'

	if len(colormap) < len(residues_plot_list) or len(colormap) < len(w_attributes_plot_list):
		raise Exception('Number of colors is not enough! Reset the color map!')

	ax_err = fig_err.add_subplot(2,2,1)

	handles_err = []

	err_title = 'Rank '+str(k)+' Residues with init '+ str(init_portion)
	ax_err.set_title(err_title)
	for i in range(len(residues_plot_list)):

		marker = colormap[i]+linemarker
		temp_err = residues_plot_list[i]
		handle, = ax_err.plot(np.array(temp_err),marker)
		handles_err.append(handle)

	ax_err.legend(handles_err,legend_list,loc=legend_loc)	
	#ax_err.set_xlabel('Iterations')
	
	# build the weight figs
	w_fig_titles = ['The Max of Weights','The Mean of Weights','The Mini of Weights']	
	ax_g_list = []
	for i in range(len(w_fig_titles)):
		ax_g_list.append(fig_err.add_subplot(2,2,i+2))	
		ax_g_list[i].set_title(w_fig_titles[i])		
	
	#plot weights of each method into each fig
	for i in range(len(w_attributes_plot_list)):
		data = np.array(w_attributes_plot_list[i])
		marker = colormap[i]+linemarker
		label_data = legend_list[i]
		for j in range(len(ax_g_list)):
			ax_g_list[j].plot(data[:,j],marker,label = label_data)
	
	for i in range(len(ax_g_list)):
		ax_g_list[i].legend()
		#ax_g_list[i].set_xlabel('Iterations')
	
	
	plt.savefig(f_fig_name)
	plt.show()
	print('The output fig is saved as '+f_fig_name+'.')
	
except Exception as e_fig:
	print('Err in plotting fig and the infor is: ', str(e_fig))
	print('\n')

print('\nFinished!')


""" Old Archives#********************************************************************************

'''
# Basic RW using first version
test_data_IO.update_fout('MINIST_RWLR_v', v = version)
U,g,err_list,err_g_list = test_model.apply_RW_Basic_FGD(w_kernel, par_kernel,adjust_on = False)
output.update({ 'U0': U0,
				'U' : U,
				'g' : g,
				'err_list':err_list,
				'err_g_list':err_g_list, 
				'fout_name':test_data_IO.fout
				})

test_data_IO.save_data(output)
'''

'''
# Basic RW using Relaxed version
test_model_theta.n_iters = 200

test_data_IO.update_fout('MINIST_ARWLR_Basic_Relaxed_v', v = version)
U,g,err_list,err_g_list = test_model_theta.apply_RW_Basic_FGD_relaxed(w_kernel, par_kernel)
output.update({ 'U0': U0,
				'U' : U,
				'g' : g,
				'err_list':err_list,
				'err_g_list':err_g_list, 
				'fout_name':test_data_IO.fout
				})

test_data_IO.save_data(output)

#*****************************************
'''

'''
# Basic RW using Betta version
test_model_betta.n_iters = 200

test_data_IO.update_fout('MINIST_ARWLR_Basic_v', v = version)
U,g,err_list,err_g_list = test_model_betta.apply_RW_Basic_FGD(w_kernel, par_kernel)
output.update({ 'U0': U0,
				'U' : U,
				'g' : g,
				'err_list':err_list,
				'err_g_list':err_g_list, 
				'fout_name':test_data_IO.fout
				})

test_data_IO.save_data(output)
#*****************************************
'''

'''
test_data_IO.update_fout('MINIST_RWLR_Classic_v',v= version)
U,g,err_list,err_g_list = test_model.apply_RW_Basic_FGD(w_kernel, par_kernel, adjust_on = False)

output.update({ 'U0': U0,
				'U' : U,
				'g' : g,
				'err_list':err_list,
				'err_g_list':err_g_list 
				
				})
test_data_IO.save_data(output)


#test_data_IO.update_fout('data_RWLR_SGD_v',v = 2)
#U,g,err_list = test_model.apply_RW_SGD_FGD_FullW(w_kernel,par_kernel,n_minibatch)


test_data_IO.update_fout('data_RWLR_SGD_Classic_v',v = version)
U,g,err_list = test_model.apply_RW_SGD_FGD_FullW(w_kernel,par_kernel,n_minibatch,adjust_on=False)
output.update({ 'U0': U0,
				'U' : U,
				'g' : g,
				'err_list':err_list
				
				})

test_data_IO.save_data(output)


#test_data_IO.update_fout('data_RWLR_SGD_NAG_v')
#U,theta,g,err_list = test_model.apply_RW_SGD_M_FGD(w_kernel,par_kernel,n_minibatch, NAG_on = True)


#test_data_IO.update_fout('data_RWLR_SGD_CM_v',v = 2)
#U,theta,g,err_list = test_model.apply_RW_SGD_M_FGD(w_kernel,par_kernel,n_minibatch)

#test_data_IO.update_fout('data_RWLR_SGD_ShortW_CM_v',v = 2)
#U,g,err_list = test_model.apply_RW_SGD_FGD_ShortW(w_kernel,par_kernel,n_minibatch, adjust_on=True)

'''

'''
output.update({ 'U0': U0,
				'U' : U,
				'g' : g,
				'err_list':err_list,
				'err_g_list':err_g_list 
				
				})

test_data_IO.save_data(output)
'''
""" #********************************************************************************


