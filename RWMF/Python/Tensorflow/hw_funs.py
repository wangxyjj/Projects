import sys
import scipy.io as io
import pickle
#from sklearn.model_selection import KFold, ShuffleSplit

def creat_hw_pickle(fin,fout):
	try:
		data = io.loadmat(fin)
	except Exception as e_mat:
		print('Error in load mat file! The infor is: ', str(e_mat))	
		print('\n')
		return -1

	with open(fout,'wb') as f:
		try:
			pickle.dump(data,f)
		except Exception as e_pickle:
			print('Err in save pickle! The infor is: ', str(e_pickle))
			print('\n')
			return -1

	return data

def byebye(e_info):
	print('Crucial error: %s. Abort now!' % e_info)
	sys.exit(e_info)

