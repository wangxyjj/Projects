'''
vanilla LSTM (containing forget, input, output gates) 
+ RWLR for each time batch
'''
import numpy as np
from numpy import random
import tensorflow as tf
from tensorflow.contrib import rnn
import scipy.io as io
import os
import pickle

from Reweight_accessories import byebye

from sklearn.model_selection import train_test_split
#from test_RWLR_basic_v1 import epoch_size
from tensorflow.examples.tutorials.mnist import input_data
from builtins import Exception

import math

def load_mat(fin):
	try:
		data = io.loadmat(fin)
		print('Successfully load data from mat %s!' % fin)
	except Exception as e_mat:
		print('Error in loading mat file! The infor is: ', str(e_mat))	
		print('\n')
		return -1

	return data	

def load_data_pickle_mat(fin):
	try:
		with open(fin,'rb') as f:	
			data = pickle.load(f)
			print('Successfully load data from pickle %s!' % fin)
			#print('Data info: %s. \n' % str(data['__header__']))
	except Exception as e_open:
		print('Err in loading pickle files and the infor is: ', str(e_open))
		print('\n')
		print('We now try loding data from mat %s!' % fin)
		print('\n')
		data = load_mat(fin)

	return data


class IO_Data(object):
	'''
	Read the data matrix from the given file and output this data into 
	the specified output file.

	Default: self.data has the size numb_points x numb_feas
			 self.label has the size 1 x numb_points	
	'''
	def __init__(self,dir_path = os.getcwd()+'/testdata/',version = 0, fout_name = 'data_output_v'):
		
		#fout = dir_path + fout_name
		self.path = dir_path
		self.v = str(version)
		self.fout = self.path + fout_name + self.v
		self.fin = None

	def update_fout(self,fout_new,path = None, v = None):
		if path:
			self.path = path
			
		if v:
			self.v = str(v)
		
		self.fout = self.path+fout_new+self.v
	
	def read_all_data(self,fin):
		'''
		Simply return all data as a dictionary.
		'''
		
		return load_data_pickle_mat(fin)

	def create_data_hw(self,fin):
		'''
		Create the training and testing data from file from handwritten 
		Each column represents one feature
		Each row represents one points
		M size: # of points x # of features
		label size: # of labels x 1
		'''
		data = self.read_all_data(fin)

		if data == -1:
			byebye('Loading data failed!')
		else:
			print('Data loaded successfully from %s !' % fin)
			self.fin = fin

			M = np.array(data['fea'])
			label = np.array(data['gnd'])
			return (M,label)
			
	def use_dataset(self,name_dataset,fin,test_portion = 0.25):
		'''
		Select dataset
		'''
		if name_dataset == 'Handwritten':
			data = self.create_data_hw(fin)
			m,_ = data[1].shape
			label = np.zeros((m,10))
			for i in range(m):
				label[i,data[1][i]] = 1
			M_train,M_test,label_train,label_test = train_test_split(data[0],label,test_size = test_portion, random_state = 42)
			
		elif name_dataset == 'YaleFace':
			data = self.create_data_hw(fin)
			m,_ = data[1].shape
			label = np.zeros((m,np.amax(data[1])))
			try:
				for i in range(m):
					label[i,data[1][i]-1] = 1
			except Exception as e_load:
				print('\nThe error infor is ', str(e_load))
				byebye('Loading data failed!')
				
			M_train,M_test,label_train,label_test = train_test_split(data[0],label,test_size = test_portion, random_state = 32)
		
		elif name_dataset == 'MINIST':
			data = self.create_data_hw(fin)
			M_train,M_test,label_train,label_test = train_test_split(data[0],data[1],test_size = test_portion, random_state = 30)
				
			
		elif name_dataset == 'tf_minist':
			data = input_data.read_data_sets("/tmp/data/", one_hot = True)
			M_train = data.train.images
			label_train = data.train.labels
			M_test = data.test.images
			label_test = data.test.labels
			self.fin = fin
			
		else:
			byebye('Loading data failed!')
		
	
		return M_train, label_train, M_test, label_test
	
	def save_data(self,output,need_pickle = False):
		fout_mat = self.fout+'.mat'
		fout_pickle = self.fout+'.pickle'
		try:
			io.savemat(fout_mat,output)
			print('\n Saving as mat file.')
			print('data is saved in %s.' % fout_mat)
		except Exception as e_savemat:
			print('Error in saving mat! The infor is: ', str(e_savemat))
			print('\n')

		if need_pickle:
			with open(fout_pickle,'wb') as f:
				try:
					pickle.dump(output,f)
					print('\n Saving as pickle file.')
					print('data is saved in %s.' % fout_pickle)
				except Exception as e_pickle:
					print('Error in saving pickle! The infor is: ', str(e_pickle))
					print('\n')


def batch_producer(data,label, batch_size):
	data_len = data.shape[0]
	batch_len = data_len//batch_size
	raw_data = data[: batch_size * batch_len,:].reshape((batch_size, batch_len,data.shape[1]))
	raw_data_label = label[:batch_size * batch_len].reshape((batch_size, batch_len,label.shape[1]))
	return raw_data, raw_data_label


class Preprocessed_Input:
	'''
	Process the input data
	A basic version to generate a 3D tensor (batch_size x num_timesteps x num_feas)
	'''
	def __init__(self, data,label,batch_size,num_timesteps):

		self.num_feas = data.shape[1]
		self.batch_size = batch_size #number of points in one batch
		self.num_timesteps = num_timesteps #numb of time steps, X1, X2, X3...Xt
		#self.epoch_size = ((data.shape[0] // batch_size) - 1) // num_timesteps #numb of iterations in one timestep
		self.epoch_size = data.shape[0]//batch_size
		self.data, self.label = batch_producer(data,label,batch_size)

	def next_batch(self,batch_idx):
		if batch_idx > self.epoch_size or batch_idx < 0:
			return -1
		else:
			#begin_idx = batch_idx*self.num_timesteps
			#end_idx = (batch_idx+1)*self.num_timesteps 

			#return a batch of raw data with original fea size
			#not divided by num_timesteps
			x = self.data[:,batch_idx,:]
			y = self.label[:,batch_idx] 
			return x,y

def generate_unrealiable_labels(Y,portion):
	'''
	perturb portion*n_points labels

	row vectors represent points

	'''
	n_points = Y.shape[0]
	idxes_raw = np.arange(n_points)
	random.shuffle(idxes_raw)

	p_size = math.floor(portion * n_points)

	if p_size >= n_points or p_size <= 0:

		byebye('Not a valid perturbation portion! Exit!')

	idxes_perturbed = idxes_raw[:p_size]

	Y_p = Y[idxes_perturbed,:]

	random.shuffle(Y_p)

	Y[idxes_perturbed,:] = Y_p

	return np.array(Y), idxes_perturbed



	
"""
class Hyperparameters:
	'''
	We can apply the random search for necessary parameters
	'''
	def __init__(self,n_classes,hidden_size):
		self.n_classes = n_classes
		self.hidden_size = hidden_size # numb of features of input data
		#self.n_layers = n_layers
"""

class RWLSTM_Model_basic:

	def __init__(self,input_obj,input_fea_size,hidden_size,n_classes,n_epochs,epoch_size = None):	
		
		self.input_obj = input_obj
		
		self.num_timesteps = input_obj.num_timesteps
		self.batch_size = input_obj.batch_size
		if epoch_size == None:
			self.epoch_size = input_obj.epoch_size
		else: 
			self.epoch_size = epoch_size

		self.input_fea_size = input_fea_size

		self.n_classes = n_classes
		self.hidden_size = hidden_size 
		self.n_epochs = n_epochs
		## indicate this model is for training or validating 
		#self.is_training = is_training
		
		# Set up the input batch storage
		#input placeholder for one chunk
		self.X = tf.placeholder(tf.float32,[None,self.num_timesteps,self.input_fea_size])
		
		#Set up output label
		self.Y = tf.placeholder(tf.float32,[None,self.n_classes])
		
		## Set up the initial state storage: the received outputs h_{t-1} and s_{t-1})
		## respectively from itself and the previous time chunk
		#self.init_state = tf.placeholder(tf.float32, 
		#	[self.parameters.n_layers, 2, self.batch_size, self.parameters.hidden_size])


	def build_Model_basic(self):
		# Set the output layer
		out_layer = {'weights':tf.Variable(tf.random_normal([self.hidden_size,self.n_classes])),
		'biases':tf.Variable(tf.random_normal([self.n_classes]))}

		''' Not needed
		## Build state list for each layer
		#state_per_layer_list = tf.unstack(self.init_state,axis = 0) 
		## Set state tuple (c, h), in that order. Where c is the hidden state and h is the output.
		## Only used when state_is_tuple=True.
		#LSTM_state_tuple = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],state_per_layer_list[idx][1]) 
		#	for idx in range(self.parameters.n_layers)])
		'''
		input_X = tf.unstack(self.X,self.num_timesteps,1)
		
		# bulid the LSTM cell (basic one, no, dynamic, no peephole)
		# the max numb of time steps is 200!
		cell =  rnn.BasicLSTMCell(self.hidden_size,forget_bias=1,state_is_tuple=True)
		outputs, _ = rnn.static_rnn(cell, input_X, dtype=tf.float32)
		# output size batch_size x n_classes
		output = tf.matmul(outputs[-1],out_layer['weights']) + out_layer['biases']

		return output

def train_Model_basic(model,learning_rate= None,test_mode = True):
#	if test_mode:
#		mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

	prediction = model.build_Model_basic()
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = model.Y) )
	if learning_rate != None:
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	else:
		optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	init=tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(model.n_epochs):
			epoch_loss = 0
			for batch_idx in range(model.epoch_size):
				epoch_x, epoch_y = model.input_obj.next_batch(batch_idx)
				epoch_x = epoch_x.reshape((model.batch_size,model.num_timesteps,model.input_fea_size))
				_, c = sess.run([optimizer, cost], feed_dict={model.X: epoch_x, model.Y: epoch_y})
				epoch_loss += c

			print('Epoch: ', epoch, 'completed out of',model.n_epochs,'loss:',epoch_loss)
			
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(model.Y, 1))
			
		if test_mode:
			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
			test_data = mnist.test.images[:128,:].reshape((-1, model.num_timesteps, model.input_fea_size))
			test_label = mnist.test.labels[:128,:]
			try:
				print('Accuracy:',accuracy.eval({model.X : test_data, model.Y : test_label}))
			except Exception as e_accuracy_eval:
				print(str(e_accuracy_eval))