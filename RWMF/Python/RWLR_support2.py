import numpy as np
from numpy import random

from Reweight_accessories import residue,weight
from Reweight_accessories import byebye,residue
from astropy.constants.codata2010 import alpha

import math

def normal_col_l2(X,tol = 0, damping = 1.e-4):
	''' l2 normalize matrix by column'''
	sum_col_X = np.sqrt(np.sum(np.square(X),axis=0))
	sum_col_X[sum_col_X<=tol] += damping
	return X/sum_col_X

def normal_col_l1(X,tol = 0, damping = 1.e-4):
	''' l1 normalize matrix by column'''
	sum_col_X = np.sum(X,axis=0)
	sum_col_X[sum_col_X<=tol] += damping
	return X/sum_col_X

def residue_w(distMj,g):
	'''
	Weighted Residue 
	inputs should be 1-dim ndarray or row vector
	g = w^2
	'''
	return np.sqrt(np.sum(distMj*g))

class ProjectM:
	'''
	All operations are conducted in np.matrix type!
	All return type are ndarray
	'''
	def __init__(self,U):
		self._U = np.matrix(U)	
		#self._M = np.matrix(M)
		self._Pu = self._U*(self._U).T

	def update_U(self,Unew):
		self._U = np.matrix(Unew)
		self._Pu = self._U*(self._U).T


	def get_Pu(self):
		return np.array(self._Pu)

	def get_U(self):
		return np.array(self._U)


	def dist_Pu(self,M):
		'''
		Project data M to the subspace spanned by U
		return Projector Pu, Projected matrix Mk and distance vector distMj
		'''
		M = np.matrix(M)
		Mk = self._Pu * M
		
		distMj = np.sum(np.square(M-Mk),axis=0)

		return np.array(distMj)

	def dist_UVT(self,VT,M):
		'''
		Distances to Mk = UVT
		'''
		M = np.matrix(M)
		VT = np.matrix(VT)
		Mk = self._U * VT
		distMj = np.sum(np.square(M-Mk),axis=0)

		return np.array(distMj)


class Weights:
	
	'''
	Generate ndarray type weights by different kernels.
	input: an array of {di} representing the projection-distance vector 
			di = ||Mi - PuMi||_2^2 or dij = [Mij - (PuMi)j]^2
	output: an array W of {wi} by wi = kernel(di) or wij = kernel(dij)
			For the diagonal matrix nxn, len(W) = n 
			For the full matrix mxl, len(W) = n = mxl
			We will consider the sparse version
	'''

	def __init__(self,dist = None,damping = 1.e-4):
		
		self._dist = dist
		self._damping = damping
	

	def update_dist(self,dist):
		self._dist = np.array(dist)

	def update_damping(self,damping_new):
		self._damping = damping_new

	def get_cur_dist(self):
		return self._dist

	def prob_wj(self):
		dist_damped = self._dist + self._damping
		return dist_damped/np.sum(dist_damped)

	def kernel_tv(self):
		'''
		Return the truth vector type weight
		gi = wi^2 = log(S/di)
		'''
		
		dvS = self.prob_wj()	
		return np.log(1/dvS)

	def partial_tv(self):
		'''
		Fisrt order derivative of truth vector weights
		'''
		dvS = self.prob_wj()
		return dvS - np.ones(dvS.shape)

	def get_kernel(self,name_kernel):
		if name_kernel == 'WeightProb':
			return self.prob_wj()
		elif name_kernel == 'TruthVector':
			return self.kernel_tv()
		elif name_kernel == 'PartialTruthVector':
			return self.partial_tv()
		else: 
			print('No such weight kernel!')
			return None
"""
class WeightedGrad:
	'''
	Weighted gradient is constructed as MGMT - PuMGMT
	Input: the ndarray or row-vector coefficient matrix C = w^2 and data matrix M
	Output: ndarray type gradient matrix 
	Jacobi = PuMGMT - MGMT (update rule x = x - hJacobi)
	All operations are performed in np.matrix type!
	'''
	def __init__(self,Pu,coef):
		self._Pu = np.matrix(Pu)
		self._coef = np.matrix(coef)

	def update_Pu(self,Pu):
		self._Pu = np.matrix(Pu)

	def update_coef(self,coef):
		self._coef = np.matrix(coef)

	def _get_Jacobi(self,M):
		MCMT = np.multiply(M,self._coef)*M.T
		return self._Pu*MCMT - MCMT

	def _get_Hessian(self,M):
		return np.multiply(M,self._coef)*M.T
		

	def grad_1st(self,M,stepsize):
		M = np.matrix(M)
		J = self._get_Jacobi(M)
		return -stepsize*J

	def grad_Newton(self,M,stepsize):
		M = np.matrix(M)
		J = self._get_Jacobi(M)
		H = self._get_Hessian(M)
		return -stepsize*H.I*J
'''
	def get_grad(self,grad_type,M,stepsize):
		if grad_type == '1st_order':
			return grad_1st(M,stepsize)
		if grad_type == 'Newton':
			return grad_Newton(M,stepsize)
		else:
			byebye('No such gradident type')
'''
"""

def get_weighted_stepsize(M,g,compressed_size = None):
	'''
	Lip = ||MgMT||_2
	'''
	M = np.matrix(M)
	g = np.matrix(g)

	if compressed_size is None:
		#w = np.sqrt(g)
		#Mw = np.multiply(M,w) # M*W
		#McMT = Mw*Mw.T #M*W^2*MT
		Mg = np.multiply(M,g) #M*g
		McMT = Mg*M.T #M*g*MT
		Lip = np.linalg.norm(McMT,2) # Lipschitz coefficient

	if Lip < 1.e-3:
		print('Caution! Very small Lipschitz!')
		Lip += 1.e-3

	stepsize = 1/Lip

	return stepsize
				

def get_weighted_factored_grad(M,g,Pu,U):
	'''
	g is a row vector
	g = w^2
	grad = (I-Pu)MgMT*U
	All operations are done in np.matrix type
	'''
	M = np.matrix(M)
	g = np.matrix(g)
	Pu = np.matrix(Pu)
	U = np.matrix(U)

	#w = np.sqrt(g)
	#Mw = np.multiply(M,w) # M*W
	#McMT = Mw*Mw.T #M*W^2*MT
	Mg = np.multiply(M,g) #M*g
	McMT = Mg*M.T #M*g*MT
	
	dim = Pu.shape[0]

	Pu_v = np.eye(dim) - Pu
	grad = Pu_v*(McMT*U) # (I-Pu)M*W^2*MT*U
	
	return np.array(grad)

def get_minibatch(M,minibatch_size):
	'''
	Columns represent points.
	'''
	minibatches_list = []
#	M = random.shuffle(M)

	for i in range(0, M.shape[1], minibatch_size):
		M_mini = M[:,i:i+minibatch_size]
		minibatches_list.append(M_mini)

	return minibatches_list

def get_weight(dist,w_obj,weight_kernel,partial_kernel,use_adjust,threshold = None):
	
	w_obj.update_dist(dist)
	g = w_obj.get_kernel(weight_kernel)
	print('\n The max g is ',np.amax(g),' the min g is ', np.amin(g))
	partial_g = w_obj.get_kernel(partial_kernel)
	
	if use_adjust:
						
		print('\n The max reduced partial_g is ',np.amax(-partial_g),
			' the min reduced partial_g is ', np.amin(-partial_g))

		g_total = g + partial_g

		print('\n The max g_total is ',np.amax(g_total),
			' the min g_total is ', np.amin(g_total))

		if threshold is not None:	
		
			selected = math.floor(threshold*dist.shape[1])
			sorted_idx = np.argsort(g_total).reshape((g_total.shape[1],))
			top_idx = sorted_idx[-selected:]
			below_idx = sorted_idx[:selected]
		
			g_total[:,top_idx] = 0.9*g[:,top_idx] + partial_g[:,top_idx]
			g_total[:,below_idx] = g[:,below_idx]+0.9*partial_g[:,below_idx]
		
	
			print('\n The changed max g_total is ',np.amax(g_total),
			' the changed min g_total is ', np.amin(g_total))
	else:
		g_total = g

	return np.array(g_total), np.array(g), np.array(partial_g)

"""
running_control = { 
	'adjust_on': adjust_on,
	'momentum_on': momentum_on,
	'full_weight_on': full_weight_on,
	'init_portion':init_portion
	}

"""

class Model_Controlling:

	def __init__(self,adjust = True,momentum = False,
		full_weight = True,init_portion= 0.5, 
		momentum_para = (0.9,1),threshold_para = None):
		self.adjust_on = adjust
		self.momentum_on = momentum
		self.full_weight_on = full_weight
		self.init_portion = init_portion
		self.momentum_para = momentum_para
		self.threshold_para = threshold_para

class RWLR_Model:

	def __init__(self,data,k,X0 = None, 
		delta_err_min = 1.e-4,damping = 0.001,n_iters = 500):
		'''
		Input data is using one column vector to represent one data point.
		'''

		self.data = np.array(data)

		if X0 is not None:
			self.init_U = np.array(normal_col_l2(X0))
		else:
			self.init_U = X0

		self.delta_err_min = delta_err_min
		self.damping = damping
		self.n_iters = n_iters

		self.k = k

	def apply_RW_Basic_FGD(self,weight_kernel,partial_kernel,model_para_obj):
		# all variables are ndarray type, not np.matrix type!

		M = self.data

		n_points = M.shape[1]

		if self.X0 is None:
			sample_size = int(math.floor(model_para_obj.init_portion*M.shape[1]))
			idxes_raw = np.arange(n_points)
			random.shuffle(idxes_raw)
			print('The initial randon shuffle is ', idxes_raw)
			init_M = M[:,idxes_raw[:sample_size]]
			U, _, _ = np.linalg.svd(init_M,full_matrices=False)
			init_U = U[:,self.k]
		else:
			init_U = self.init_U

		projectedM_obj = ProjectM(init_U)

		weight_obj = Weights(damping = self.damping) # initialize weight obj

		Ut = init_U

		if model_para_obj.momentum_on:

			theta_t = Ut
			mu = model_para_obj.momentum_para[0]
			alpha = model_para_obj.momentum_para[1]

		err_list = [] # inialize err list

		for iter_t in range(self.n_iters):
			if model_para_obj.adjust_on:
				print('\nThe %s-th iteration of basic FGD' % str(iter_t+1))
			else:
				print('\nThe %s-th iteration of basic FGD_Classic' % str(iter_t+1))
				
			Put = projectedM_obj.get_Pu() # get current Pu

			# calculate weights
			dist_t = projectedM_obj.dist_Pu(M)

			weight_obj.update_dist(dist_t)

			g_total_t,gt,_ = get_weight(dist_t,weight_obj,weight_kernel,
				partial_kernel,model_para_obj.adjust_on,model_para_obj.threshold_para)	

			# calculate the gradient
			h = get_weighted_stepsize(M,g_total_t) 

			if model_para_obj.momentum_on:
				grad = get_weighted_factored_grad(M,g_total_t,Put,theta_t)
				Ut = normal_col_l2(mu*Ut + h*grad)
				theta_t += alpha*Ut
				theta_t = normal_col_l2(theta_t)

			else:			
				grad = get_weighted_factored_grad(M,g_total_t,Put,Ut)

				Ut = normal_col_l2(Ut + h*grad)

			projectedM_obj.update_U(Ut) #update Ut

			errt = residue_w(dist_t,gt) #update error

			err_list.append(errt)
			
			if iter_t == 0:
				err_pre = errt
				print('\nThe inital err is %.3f .' % errt)
			else:

				delta_errt = abs(err_pre - errt)
				
				err_pre = errt

				print('\nCurrent err is %(error).3f and the error difference is %(delta).3f' % {'error':errt,'delta':delta_errt})

				if delta_errt < self.delta_err_min:
					print('\nBreak the loop at %d-th iteration, since little improvement!' % iter_t)
					break

		return np.array(Ut), np.array(gt),err_list

	def apply_RW_SGD_FGD(self,weight_kernel,partial_kernel,n_minibatch,model_para_obj):
		'''
		Minibatch version
		'''
		#random.seed(365412789)
		M = self.data

		idx_M = np.arange(M.shape[1])
		random.shuffle(idx_M)
		print('\nThe first random shuffle of SGD',idx_M)

		minibatch_size = idx_M.shape[0] // n_minibatch 
		idx_M = idx_M[:minibatch_size*n_minibatch]
		M = M[:,idx_M]

		idx_M = idx_M[None,:] # change to a row vector
		
		minibatch_idxes = get_minibatch(idx_M,minibatch_size)
		
		if self.X0 is None:
			init_idx = random.randint(0,len(minibatch_idxes))
			init_batch_idxes = minibatch_idxes[init_idx].reshape((minibatch_size,))
			
			init_M = M[:,init_batch_idxes]
			U, _, _ = np.linalg.svd(init_M,full_matrices=False)
			init_U = U[:,:self.k]
		
		projectedM_obj = ProjectM(init_U) #initialize distM obj

		weight_obj = Weights(damping = self.damping) # initialize weight obj		

		Ut = init_U

		if model_para_obj.momentum_on:
			theta_t = Ut
			mu = model_para_obj.momentum_para[0]
			alpha = model_para_obj.momentum_para[1]

		err_list = [] # inialize err list

		for iter_t in range(self.n_iters):
			if model_para_obj.adjust_on:
				print('\n~~~~~The %s-th iteration of SGD' % str(iter_t+1))
			else:
				print('\n~~~~~The %s-th iteration of Classic_SGD' % str(iter_t+1))
			
			if iter_t == 0:
				idx = init_idx
				batch_idxes_t = init_batch_idxes
				Mt = init_M
			else:
				idx = random.randint(0,len(minibatch_idxes))
				batch_idxes_t = minibatch_idxes[idx].reshape((minibatch_size,))
				Mt = M[:,batch_idxes_t]
				
			print('\n The %d-th minibatch is selected.\n' % idx)

			

			Put = projectedM_obj.get_Pu() # get current Pu

			# calculate the weights

			if model_para_obj.full_weight_on:
				dist_t = projectedM_obj.dist_Pu(M)
			else:
				print('\n Using short weights mode.')
				dist_t = projectedM_obj.dist_Pu(Mt)

			weight_obj.update_dist(dist_t)

			g_total_t,gt,_ = get_weight(dist_t,weight_obj,weight_kernel,
				partial_kernel,model_para_obj.adjust_on,model_para_obj.threshold_para)	
			
			if model_para_obj.full_weight_on:
				g_total_t = g_total_t[:,batch_idxes_t]

			# calculate the gradients

			h = get_weighted_stepsize(Mt,g_total_t) 			
			
			if model_para_obj.momentum_on:
				grad = get_weighted_factored_grad(Mt,g_total_t,Put,theta_t)
				Ut = normal_col_l2(mu*Ut + h*grad)
				theta_t += alpha*Ut
				theta_t = normal_col_l2(theta_t)

			else:			
				grad = get_weighted_factored_grad(Mt,g_total_t,Put,Ut)

				Ut = normal_col_l2(Ut + h*grad)

				
			projectedM_obj.update_U(Ut)
			
			if model_para_obj.full_weight_on:
				errt, _ = residue(M,Ut,np.dot(Ut.T,M),np.sqrt(gt))
			else:
				wt = weight(M,Ut,0.001)
				errt, _ = residue(M,Ut,np.dot(Ut.T,M),wt)

			err_list.append(errt)

			if iter_t == 0:
				err_pre = errt
				print('\n The initial err is %.3f .' % errt)
			else:

				delta_errt = abs(err_pre - errt)
				err_pre = errt

				print('\n Current err is %(error).3f and the err difference is %(delta).3f' 
					% {'error':errt,'delta':delta_errt})

				if delta_errt < self.delta_err_min:
					print('\nBreak the loop at %d-th iteration, since little improvement!' % iter_t)
					break
				
		return np.array(Ut),np.array(np.square(wt)),err_list

def update_Pu_momentum(M,g,U_pre,P_pre,theta_pre,para_tuple,stepsize):

	mu = para_tuple[0]

	aplph = para_tuple[1]

	grad = get_weighted_factored_grad(M,g,P_pre,theta_pre)

	# apply 1st grad descent on U
	U = normal_col_l2(mu*U_pre + stepsize*grad)

	# update theta 
	theta += normal_col_l2(aplph*U)

	return U, theta



