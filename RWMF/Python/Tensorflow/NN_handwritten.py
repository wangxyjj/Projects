'''
NN model on Handwritten data by using Tensor Flow 
'''
import tensorflow as tf
import numpy as np
import os 
import pickle

from sklearn.model_selection import train_test_split

from hw_funs import creat_hw_pickle,byebye

dir_path = os.getcwd()
version = 0

fout_mat = dir_path + '/testdata/res_hw_v'+str(version)+'.mat'
fout = dir_path + '/testdata/res_hw_v'+str(version)+'.pickle'

fin_mat = '/Users/XiangyuW/Google Drive/Research2015/SNMFUKmeans/Code/USPSHandwritten/HandW_28x28.mat'
fin = dir_path + '/testdata/handwritten.pickle'

try:
	with open(fin,'rb') as f:
		data = pickle.load(f)
		print('Successfully load data from %s!' % fin)
		print('Data info: %s. \n' % str(data['__header__']))
except Exception as e_open:
	print('Err in load pickle data files and infor is: ', str(e_open))
	print('\n')
	print('We now create the pickle file from ', fin_mat)
	print('\n')
	data = creat_hw_pickle(fin_mat,fin)
	if data == -1:
		byebye('Loading data file failed!')
	else:
		print('Data info: %s. \n' % str(data['__header__']))

'''
Each column represents one feature
Each row represents one points
M size: # of points x # of features
label size: # of labels x 1
'''
M = np.array(data['fea'])
label = np.array(data['gnd'])
n,dim = M.shape
img_h = data['faceH']
img_w = data['faceW']

output = {'M':M, 'label':label, 'dim':dim, 'n':n}
M_train,M_test,l_train,l_test = train_test_split(M,label,test_size = 0.2, random_state = 0)

output.update({'M_train':M_train,'M_test':M_test,'l_train':l_train,'l_test':l_test})










