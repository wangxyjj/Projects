#import numpy as np
from RWLSTM_lib import IO_Data, Preprocessed_Input, RWLSTM_Model_basic
from RWLSTM_lib import train_Model_basic,generate_unrealiable_labels

test_data_IO = IO_Data()

test_mode = True

if test_mode:
	fin_mat = 'tf_minist'
	dataset_name = 'tf_minist'
else:
	fin_mat = '/Users/XiangyuW/Google Drive/Research2015/SNMFUKmeans/Code/USPSHandwritten/HandW_28x28.mat'
	dataset_name = 'Handwritten'


raw_data_train,raw_label_train,raw_data_test,raw_label_test = test_data_IO.use_dataset(dataset_name,fin_mat)
																				
num_train,dim = raw_data_train.shape 

pertubed_label_train,_ = generate_unrealiable_labels(raw_label_train,0.2)

RWLR_parameters = {
	'w_kernel' : 'TruthVector',
	'par_kernel' : 'PartialTruthVector',
	'k' : 7**2,
	'n_minibatch' : 10,
	'init_portion': 0.5,
	'delta_err_min': 1.e-3,
	'n_iters': 40,
	'random_seed':None
}

projected_data_train = RWLR_by_labels(raw_data_train,pertubed_label_train,RWLR_parameters)

batch_size = 128
num_timesteps = 28
input_fea_size = 28
epoch_size = 20
hidden_size = 28
n_classes = 10
n_epochs = 50

input_train = Preprocessed_Input(raw_data_train,raw_label_train,batch_size,num_timesteps)

LSTM_model = RWLSTM_Model_basic(input_train, 
							input_fea_size,hidden_size,n_classes,n_epochs,epoch_size)

train_Model_basic(LSTM_model,test_mode)



