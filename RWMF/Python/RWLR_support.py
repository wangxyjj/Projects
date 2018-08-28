import numpy as np
from numpy import random

from Reweight_accessories import residue,weight
from Reweight_accessories import byebye,residue
from astropy.constants.codata2010 import alpha

import math

import os

import logging

from My_Logger import Logger

def normal_col_l2(X,tol = 1.e-6, damping = 1.e-5):
	''' l2 normalize matrix by column'''
	sum_col_X = np.sqrt(np.sum(np.square(X),axis=0))
	sum_col_X[sum_col_X<=tol] = damping
	return X/sum_col_X

def normal_col_l1(X,tol = 1.e-7, damping = 1.e-5):
	''' l1 normalize matrix by column'''
	sum_col_X = np.sum(X,axis=0)
	sum_col_X[sum_col_X<=tol] = damping
	return X/sum_col_X

def residue_w(distMj,g):
	'''
	Weighted Residue 
	inputs should be 1-dim ndarray or row vector
	g = w^2
	'''
	return np.sqrt(np.sum(distMj*g))

def record_error(err_pre,err_t,gt,index_iter,oscillation_control,delta_err_min,err_list,g_list):

	max_in_pre_errlist = max(err_list[-oscillation_control:])

	delta_errt = abs(err_pre - err_t)


	if delta_errt < delta_err_min:
		print('\nBreak the loop at %d-th iteration, since little improvement!' % index_iter)
		return -1	
	elif err_t > max_in_pre_errlist:
		print('\n Oscillation begins at %d-th iteration! Forced to stop!' % index_iter)
		return -2
	else:
		# record the residues and weights infor
		err_list.append(err_t)		

		g_list.append((np.amax(gt),np.mean(gt),np.amin(gt)))					

		print('--Current err is %(error).3f and the improvement is %(delta).3f' % {'error':err_t,'delta':delta_errt})

		return err_t



class ProjectM:
	'''
	All operations are conducted in np.matrix type!
	All return type are ndarray
	'''
	def __init__(self,U):
		self._U = np.matrix(U)
		#self._M = np.matrix(M)
		self._Pu = self._U*(self._U).T
		#self._Pu = self.make_orthonormal(self._U)
	
	def make_orthonormal(self,U):
		U = np.matrix(U)
		UTU = np.linalg.inv(U.T*U)
		Pu = U*UTU*U.T
		return np.array(Pu)
	
	def update_U(self,Unew):
		self._U = np.matrix(Unew)
		self._Pu = self._U*(self._U).T
		#self._Pu = self.make_orthonormal(self._U)
	
	def update_Pu(self,Pu_new):
		self._Pu = Pu_new


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

	def __init__(self,dist = None,damping = 1.e-5):
		
		self._dist = np.array(dist)
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
	#print('\n The max g is ',np.amax(g),' the min g is ', np.amin(g))
	partial_g = w_obj.get_kernel(partial_kernel)
	
	if use_adjust:
		
		'''				
		print('\n The max reduced partial_g is ',np.amax(-partial_g),
			' the min reduced partial_g is ', np.amin(-partial_g))
		'''
		g_total = g + partial_g
		
		if threshold is not None:	
			# calculate the gradient
		
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

class Weight_Generator:
	
	def __init__(self,weight_model,kernel,partial_kernel,
		dist_t = None,kernel_mode = True, gd_mode = True,damping = 1.e-5): 

		'''
		The current dist used in Weight_Generator is defaultly contained 
		in w_obj. Only the previous dist_t needs to be passed as an input. 
		'''

		self.w_obj = weight_model
		self.w_kernel = kernel
		self.par_kernel = partial_kernel
		self.k_mode = kernel_mode
		self.gd_mode = gd_mode
		self.dist_t = dist_t
		self.damping = damping

	def update_damping(self,new_damping):
		self.damping = new_damping

	def update_dist_t(self,new_dist_t):
		self.dist_t = new_dist_t
		self.w_obj.update_dist(new_dist_t)

	def approx_kernel_g(self):
		pass

	def approx_partial_g(self): 
		'''
		Current formula:
		partial_gi = di(x) * sum_j^n[ partial_gi_dj(xt) * partial_dj(x) ]
		partial_g = sum_i^n[ partial_gi ]
		'''
		if not self.dist_t:
			byebye('dist_t not defined!')

		elif self.par_kernel == 'PartialTruthVector':
			'''
			For TruthVector:
			partial_gi = di(x) * (1/s_t - 1/d_i_t) + (s(x)-d_i(x))/s_t
			'''
			dist = self.w_obj.get_cur_dist()

			dist_damped = dist + self.damping

			sum_damped =  np.sum(dist_damped)

			dist_t_damped = self.dist_t + self.damping

			sum_t_damped =  np.sum(dist_t_damped)
			
			partial_gi_di_t = np.ones(dist_t_damped.shape)/sum_t_damped - 1/dist_t_damped

			partial_gi_dj = (sum_damped - dist_damped)/sum_t_damped

			partial_g = dist*partial_gi_di_t + partial_gi_dj

			#partial_g[partial_g<0] = 0 

			return partial_g

		else:

			byebye('No such kernel!')
	
	def get_kernel_weight(self):
		
		if self.k_mode:
			g = self.w_obj.get_kernel(self.w_kernel)
		else:
			g = self.approx_kernel_g()
		
		return g
	
	def get_partial_weight(self):
		
		if self.gd_mode:
			partial_g = self.w_obj.get_kernel(self.par_kernel)
		else:
			partial_g = self.approx_partial_g()
		
		return partial_g		



	def get_total_weight(self):
		'''
		Extend the origial get_weight to have vanished gradient processing and
		can be applied to general cases (Mini optimization for g and gvd).

		'''
		
		g = self.get_kernel_weight()

		#print('\n The max g is ',np.amax(g),' the min g is ', np.amin(g))
			
		partial_g = self.get_partial_weight()

		return g+partial_g

def test_projection(M,g,partial_g,U):
	'''
	column vectors represent points
	'''
	M = np.matrix(M)
	U = np.matrix(U)
	k = U.shape[1]
	
	print('UTU = ', U.T*U)
	
	w1 = np.sqrt(-partial_g)
	w2 = np.sqrt(g)
	
	Mw1 = np.multiply(M,w1)
	Mw2 = np.multiply(M,w2)
	
	M_B = Mw1*Mw2.T
	
	trMB = np.trace(M_B)
	
	U_B, S_B, _ = np.linalg.svd(M_B,full_matrices=False)
	
	print('\nThe trace of B is', trMB)
	
	
	U_B = U_B[:,:k]
	
	print('\nThe trace of P_B*B is ', np.trace(U_B*U_B.T*M_B))
	
	#U = U_B*(U_B.T*U)
	
	U = normal_col_l2(U)
	
	M_p = U*U.T*M_B
	
	trMp = np.trace(M_p)
	
	print('\nThe trace of PuB is ', trMp)
	
	print('|Tr(PuB)|-|Tr(B)| = ', abs(trMp)-abs(trMB))
	
	_,S_p,_ = np.linalg.svd(M_p,full_matrices = False)
	
	

def get_initU(data,portion,k,random_on = True,random_seed = None):
	'''
	column vectors represent points
	'''	
	sample_size = int(math.floor(portion*data.shape[1]))
	idxes_raw = np.arange(data.shape[1])
	
	if random_on:				
		if random_seed is not None:
			np.random.seed(random_seed)
		random.shuffle(idxes_raw)
		print('The initial randon shuffle is ', idxes_raw)
	
	idxes_sampled = idxes_raw[:sample_size]
	
	Ms = data[:,idxes_sampled]
	Us, Ss, _ = np.linalg.svd(Ms,full_matrices=False)
	U0 = Us[:,:k]
	
	return U0,idxes_sampled,Ss


	
class RWLR_Model:

	def __init__(self,data,X0,idxes0,Y0T = None,delta_err_min = 1.e-4,damping = 0.001,n_iters = 500):
		'''
		Input data is using one column vector to represent one data point.
		'''

		self.data = np.array(data)
		self.init_U = np.array(normal_col_l2(X0))
		self.init_idxes = idxes0

		self.delta_err_min = delta_err_min
		self.damping = damping
		self.n_iters = n_iters

		self.k = X0.shape[1]

		#init_sample_size = int(floor(0.1*data.shape[1]))
		#Ms = random.shuffle(M)[:,:init_sample_size]

	def apply_RW_Basic_FGD(self,weight_kernel,partial_kernel, adjust_on = True):
		# all variables are ndarray type, not np.matrix type!

		M = self.data

		projectedM_obj = ProjectM(self.init_U) #initialize distM obj

		weight_obj = Weights(damping = self.damping) # initialize weight obj

		init_dist = projectedM_obj.dist_Pu(M) #initial distance
		
		init_g_total,init_g,init_part_g = get_weight(init_dist,weight_obj,
								weight_kernel,partial_kernel,adjust_on)
		
		init_err = residue_w(init_dist,init_g)
		print('The initial error is ', init_err)
		#test_projection(M,init_g,init_part_g,self.init_U)

		#init_g = init_g[:,self.init_idxes]
		#init_part_g = init_part_g[:,self.init_idxes]
		#U_p, _ = make_projection(M[:,self.init_idxes],init_g,init_part_g,self.init_U)
		#projectedM_obj.update_U(U_p)
		

		err_list = [init_err] # initialize err list

		dist_t = init_dist
		Ut = self.init_U
		Put = projectedM_obj.get_Pu()
		#h = get_weighted_stepsize(M,init_g_total)
		
		err_pre = init_err
		
		err_g_list = [np.amax(init_g)]
		
		g_pre = init_g
		
		for iter_t in range(self.n_iters):

			print('\nThe %s-th iteration of basic FGD' % str(iter_t+1))

			# calculate the weight	
			g_total_t,gt,_ = get_weight(dist_t,weight_obj,weight_kernel,partial_kernel,adjust_on)	

			h = get_weighted_stepsize(M,init_g_total)
			
			n_inner_loop = 20
			
			
			#print('Enter the inner loop')
			
			inner_err_pre = err_pre
			
			for i in range(n_inner_loop):
				
				print('Enter the inner %s-th loop' % str(i+1))
			
				grad = get_weighted_factored_grad(M,g_total_t,Put,Ut)
			
				Ut = Ut + h*grad

				Ut = normal_col_l2(Ut)

				projectedM_obj.update_U(Ut)
			
				Put = projectedM_obj.get_Pu() # update Pu

				dist_t = projectedM_obj.dist_Pu(M) #update distance

				inner_errt = residue_w(dist_t,gt) #update error
				
				inner_delta_errt = abs(inner_err_pre - inner_errt)
				
				inner_err_pre = inner_errt
				
				if inner_delta_errt < 0.01:
					print('Break the inner loop at %s-th iteration, since little improvement!' % str(i+1))
			
			#errt = inner_errt
			
			wt = weight(M,Ut,0.001)
			errt, _ = residue(M,Ut,np.dot(Ut.T,M),wt)	
			
			err_list.append(errt)
			
			# ***record the variation of weight
			#err_gt = np.linalg.norm(gt-g_pre)
			
			err_g_list.append(np.amax(gt))
			
			g_pre = gt #update previous weight
			#*********************************
			
			delta_errt = abs(err_pre - errt)
				
			err_pre = errt

			print('Current err is %(error).3f and the improvement is %(delta).3f' % {'error':errt,'delta':delta_errt})

			if delta_errt < self.delta_err_min:
				print('Break the loop at %s-th iteration, since little improvement!' % str(iter_t+1))
				break

		return np.array(Ut), np.array(gt),err_list, err_g_list

	def apply_RW_SGD_M_FGD(self,weight_kernel,partial_kernel,n_minibatch, NAG_on = False):
		'''
		Minibatch version
		classical momentum
		U_{t+1}	= mu*Ut - h* grad(theta_t)
				= mu*Ut - h* grad(P_theta_t)theta_t

		theta_{t+1} = theta_t + alpha*theta_{t+1}
		'''
		M = self.data
		idx_M = np.arange(self.data.shape[1])
		random.shuffle(idx_M)
		M = M[:,idx_M]

		minibatch_size = idx_M.shape[0] // n_minibatch 
		idx_M = idx_M[:minibatch_size*n_minibatch]
		M = M[:,:minibatch_size*n_minibatch]

		idx_M = idx_M[None,:] # change to a row vector
		minibatch_idxes = get_minibatch(idx_M,minibatch_size)

		mu = 0.9

		alpha = 0.5

		projectedM_obj = ProjectM(self.init_U) #initialize distM obj

		weight_obj = Weights(damping = self.damping) # initialize weight obj		
		
		Ut = self.init_U

		theta_t = Ut 
		
		#theta_pre = theta_t
		
		dist_t = projectedM_obj.dist_Pu(M) #initial distance

		err_list = [] # initialize err list
		
		for iter_t in range(self.n_iters):
			if NAG_on:
				print('The %s-th iteration of NAG SGD' % str(iter_t+1))
			else:
				print('The %s-th iteration of CM SGD' % str(iter_t+1))
			
			idx = random.randint(0,len(minibatch_idxes))

			batch_idxes_t = minibatch_idxes[idx].reshape((minibatch_size,))
			Mt = M[:,batch_idxes_t]			
			
			Ptheta_t = projectedM_obj.get_Pu()

			#if iter_t > 0:
			#	Ptheta_t += alpha*np.dot(theta_pre,theta_pre.T)

			g_total_t,gt,_ = get_weight(dist_t,weight_obj,weight_kernel,partial_kernel,True)

			g_total_t = g_total_t[:,batch_idxes_t]

			# calculate the gradient

			h = get_weighted_stepsize(Mt,g_total_t) 
			print('The learning rate is %f.' % h)
				
			if NAG_on:
				theta_temp = normal_col_l2(theta_t+mu*Ut)
				grad = get_weighted_factored_grad(Mt,g_total_t,Ptheta_t,theta_temp)
			else:
				grad = get_weighted_factored_grad(Mt,g_total_t,Ptheta_t,theta_t)
			#grad = get_weighted_factored_grad(Mt,g_total_t,Ptheta_t,Ut)

			Ut = normal_col_l2(mu*Ut + h*grad)
			
			#theta_pre = theta_t

			theta_t += alpha*Ut
			
			theta_t = normal_col_l2(theta_t)

			projectedM_obj.update_U(Ut)

			dist_t = projectedM_obj.dist_Pu(M) #update distance

			errt = residue_w(dist_t,gt) #update error 

			err_list.append(errt)

			if iter_t == 0:
				err_pre = errt
				print('The initial err is %.3f .' % errt)
			else:

				delta_errt = abs(err_pre - errt)
				
				err_pre = errt

				print('Current err is %(error).3f and the improvement is %(delta).3f' % {'error':errt,'delta':delta_errt})

				if delta_errt < self.delta_err_min:
					print('Break the loop at %d-th iteration, since little improvement!' % iter_t)
					break



		return np.array(Ut),np.array(theta_t),np.array(gt),err_list

	def apply_RW_SGD_FGD_FullW(self,weight_kernel,partial_kernel,n_minibatch,adjust_on = True):
		'''
		Minibatch version
		In each step, the weigh is first calculated considering all points. Then
		select the weights corresponding to the minibatch.
		'''
		#random.seed(365412789)
		M = self.data
		idx_M = np.arange(self.data.shape[1])
		random.shuffle(idx_M)
		print('\nThe first random shuffle of SGD',idx_M)
		#M = M[:,idx_M]

		minibatch_size = idx_M.shape[0] // n_minibatch 
		idx_M = idx_M[:minibatch_size*n_minibatch]
		M = M[:,:minibatch_size*n_minibatch]

		idx_M = idx_M[None,:] # change to a row vector
		minibatch_idxes = get_minibatch(idx_M,minibatch_size)

		init_idx = 0
		init_batch_idxes = minibatch_idxes[init_idx].reshape((minibatch_size,))
		
		init_M = M[:,init_batch_idxes]
		U, _, _ = np.linalg.svd(init_M,full_matrices=False)
		init_U = U[:,:self.k]
		
		projectedM_obj = ProjectM(init_U) #initialize distM obj

		weight_obj = Weights(damping = self.damping) # initialize weight obj		

		Ut = init_U

		dist_t = projectedM_obj.dist_Pu(M) #initial distance

		err_list = [] # inialize err list

		for iter_t in range(self.n_iters):
			if adjust_on:
				print('The %s-th iteration of SGD' % str(iter_t+1))
			else:
				print('The %s-th iteration of SGD_Classic' % str(iter_t+1))

			idx = random.randint(0,len(minibatch_idxes))
			print('\n The %d-th minibatch is selected.\n' % idx)

			batch_idxes_t = minibatch_idxes[idx].reshape((minibatch_size,))
			Mt = M[:,batch_idxes_t]

			Put = projectedM_obj.get_Pu() # get current Pu

			#g_total_t,gt,_ = get_weight(dist_t,weight_obj)

			g_total_t,gt,_ = get_weight(dist_t,weight_obj,weight_kernel,partial_kernel,adjust_on)
						
			g_total_t = g_total_t[:,batch_idxes_t]

			h = get_weighted_stepsize(Mt,g_total_t)
			#h = 0.001

			grad = get_weighted_factored_grad(Mt,g_total_t,Put,Ut)		

			Ut = normal_col_l2(Ut + h*grad)

			projectedM_obj.update_U(Ut)

			dist_t = projectedM_obj.dist_Pu(M) #update distance

			errt = residue_w(dist_t,gt) #update error

			err_list.append(errt)

			if iter_t == 0:
				err_pre = errt
				print('The initial err is %.3f .' % errt)
			else:

				delta_errt = abs(err_pre - errt)
				
				err_pre = errt

				print('Current err is %(error).3f and the improvement is %(delta).3f' % {'error':errt,'delta':delta_errt})

				if delta_errt < self.delta_err_min:
					print('Break the loop at %d-th iteration, since little improvement!' % iter_t)
					break

		return np.array(Ut),np.array(gt),err_list
	
	
	def apply_RW_SGD_FGD_ShortW(self,weight_kernel,partial_kernel,n_minibatch,adjust_on = True,momentum_on = True):
		'''
		Minibatch version
		The weight in each SGD step is calculated using each batch Mt.
		'''
		random.seed(365412789)
		M = self.data
		idx_M = np.arange(self.data.shape[1])
		random.shuffle(idx_M)
		print('\nThe first random shuffle of SGD',idx_M)
		#M = M[:,idx_M]

		minibatch_size = idx_M.shape[0] // n_minibatch 
		idx_M = idx_M[:minibatch_size*n_minibatch]
		M = M[:,:minibatch_size*n_minibatch]

		idx_M = idx_M[None,:] # change to a row vector
		
		minibatch_idxes = get_minibatch(idx_M,minibatch_size)
		
		init_idx = 0
		init_batch_idxes = minibatch_idxes[init_idx].reshape((minibatch_size,))
		
		init_M = M[:,init_batch_idxes]
		U, _, _ = np.linalg.svd(init_M,full_matrices=False)
		init_U = U[:,:self.k]
		
		projectedM_obj = ProjectM(init_U) #initialize distM obj

		weight_obj = Weights(damping = self.damping) # initialize weight obj		

		Ut = init_U
		if momentum_on:
			theta_t = Ut
			mu = 0.9
			alpha = 1

		err_list = [] # inialize err list

		for iter_t in range(self.n_iters):
			if adjust_on:
				print('\n~~~~~The %s-th iteration of SGD_short_weight' % str(iter_t+1))
			else:
				print('\n~~~~~The %s-th iteration of Classic_SGD_short_weight' % str(iter_t+1))
			
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

			dist_t = projectedM_obj.dist_Pu(Mt)
			weight_obj.update_dist(dist_t)

			# calculate the gradient
			gt = weight_obj.get_kernel(weight_kernel)						
			
			print('\n The max g is ',np.amax(gt),' the min g is ', np.amin(gt))		
			
			if adjust_on:			
				partial_gt = weight_obj.get_kernel(partial_kernel)
				print('\n The max reduced partial_g is ',np.amax(-partial_gt),
					' the min reduced partial_g is ', np.amin(-partial_gt))	
				
				g_total_t = gt + partial_gt
				
				print('\n The max g_total_t is ',np.amax(gt),
					' the min g_total_t is ', np.amin(gt))	
			else:
				g_total_t = gt
	
			h = get_weighted_stepsize(Mt,g_total_t) 			
			
			if momentum_on:
				grad = get_weighted_factored_grad(Mt,g_total_t,Put,theta_t)
				Ut = normal_col_l2(mu*Ut + h*grad)
				theta_t += alpha*Ut
				theta_t = normal_col_l2(theta_t)

			else:			
				grad = get_weighted_factored_grad(Mt,g_total_t,Put,Ut)

				Ut = normal_col_l2(Ut + h*grad)

				
			projectedM_obj.update_U(Ut)

			
			wt = weight(M,Ut,0.001)
			errt, _ = residue(M,Ut,np.dot(Ut.T,M),wt)

			err_list.append(errt)

			if iter_t == 0:
				err_pre = errt
				print('\n The initial err is %.3f .' % errt)
			else:

				delta_errt = abs(err_pre - errt)
				err_pre = errt

				print('\n Current err is %(error).3f and the improvement is %(delta).3f' % {'error':errt,'delta':delta_errt})

				if delta_errt < self.delta_err_min:
					print('\nBreak the loop at %d-th iteration, since little improvement!' % iter_t)
					break
				
		return np.array(Ut),np.array(np.square(wt)),err_list

class Model_Controlling:

	def __init__(self,momentum = False,
		full_weight = True,init_portion= 0.5, 
		momentum_para = (0.9,1),threshold_para = None):
		self.momentum_on = momentum
		self.full_weight_on = full_weight
		self.init_portion = init_portion
		self.momentum_para = momentum_para
		self.threshold_para = threshold_para

class RWLR_Model_Betta(RWLR_Model):

	def __init__(self,data,X0,idxes0,control_para_obj,Y0T = None,
		delta_err_min = 1.e-4,damping = 0.001,n_iters = 500,logger = None):

		super().__init__(data,X0,idxes0,Y0T,delta_err_min,damping,n_iters)
		self.control_para_obj = control_para_obj

		if logger:
			self.logger = logger
		else:
			cwd = os.getcwd()

			logPath = cwd+"/logs/"

			self.log_obj = Logger(logPath)
			self.logger = self.log_obj.logger

	def _update_U_momentum(self,M,g,U_pre,P_pre,theta_pre,
		stepsize = None):
		
		'''
		Classical momentum:
		U_{t+1} = mu*Ut - h* grad(theta_t)
				= mu*Ut - h* grad(P_theta_t)theta_t
		theta_{t+1} = theta_t + alpha*theta_{t+1}
		
		stepsize = -h
		'''

		mu = self.control_para_obj.momentum_para[0]

		aplph = self.control_para_obj.momentum_para[1]

		grad = get_weighted_factored_grad(M,g,P_pre,theta_pre)

		if stepsize is None:
			stepsize = get_weighted_stepsize(M,g)

		# apply 1st grad descent on U
		U = normal_col_l2(mu*U_pre + stepsize*grad)

		# update theta 
		theta =normal_col_l2(theta_pre + aplph*U)

		return U, theta
	
	def _update_U_1st_GD(self,M,g,U_pre,P_pre,stepsize = None):
		'''
		stepsize = -h
		'''

		grad = get_weighted_factored_grad(M,g,P_pre,U_pre)

		if stepsize is None:
			stepsize = get_weighted_stepsize(M,g)

		U = normal_col_l2(U_pre + stepsize*grad)	

		return U	
	
	def _update_U_CriticalPoint(self,M,g,U,damping = 0):

		U = np.matrix(U)
		M = np.matrix(M)

		W = np.sqrt(g)

		# update V
		VT = U.T*M

		# update X
		Mw = np.multiply(M,W)
		VTDw = np.multiply(VT,W)
		if damping >0:
			B = (VTDw*VTDw.T+damping*np.eye(VTDw.shape[0])).I  # B = inv(VtTDwVt)
		else:
			B = (VTDw*VTDw.T).I

		Unew = Mw*VTDw.T*B

		return np.array(Unew)
	

	def _update_U_innerloop(self,M,g,U0,Pu0,n_inner_loop):

		Ut = U0

		Put = Pu0

		if self.control_para_obj.momentum_on:
			theta_t = Ut

		stopsign = False

		h = get_weighted_stepsize(M,g)

		for i in range(n_inner_loop):

			self.logger.debug('Enter The %s-th iteration of inner loop' % str(i+1))
			
			#h = 1/math.sqrt(i+1)
			if self.control_para_obj.momentum_on:
				
				Ut, theta_t = self._update_U_momentum(M,g,Ut,Put,theta_t,stepsize = h)
				
				Put = np.dot(theta_t,theta_t.T)
		
			else:

				Ut = self._update_U_1st_GD(M,g,Ut,Put,stepsize = h)
				#Ut = self._update_U_CriticalPoint(M,g,Ut)

				Put = np.dot(Ut,Ut.T)

			if i<1:
				Ut_pre = Ut
			else:
				relative_delta_t = np.linalg.norm(Ut - Ut_pre)
				Ut_pre = Ut
				self.logger.debug('The improvement of inner step is %f' % relative_delta_t)
				if relative_delta_t < 1.e-3:
					stopsign = True
			if stopsign:
				self.logger.debug('Jump out of the inner loop since converged enough!')
				break

		return Ut

	def _initialize_FGD(self,M,weight_kernel):

		projectedM_obj = ProjectM(self.init_U) #initialize distM obj

		weight_obj = Weights(damping = self.damping) # initialize weight obj

		init_dist = projectedM_obj.dist_Pu(M) #initial distance

		
		#init_g = weight(M,self.init_U,0.001)

		weight_obj.update_dist(init_dist)

		init_g = weight_obj.get_kernel(weight_kernel)

		init_err = residue_w(init_dist,init_g)

		print('The initial err is ',init_err)

		err_list = [init_err] # initialize err list

		h = get_weighted_stepsize(M,init_g)

		return projectedM_obj, weight_obj, err_list, init_g, init_dist, h


	def apply_RW_Basic_FGD(self,weight_kernel,partial_kernel):

		'''
		'Basic' means all the data points are involved in computation.
		'''
		
		M = self.data

		projectedM_obj, weight_obj, err_list, init_g, dist_t, h = self._initialize_FGD(M,weight_kernel)

		Ut = self.init_U

		Put = projectedM_obj.get_Pu()

		err_g_list = [(np.amax(init_g),np.amin(init_g))]
		print('The initial max-min of weights are: ', err_g_list[0])

		
		if self.control_para_obj.momentum_on:

			theta_t = Ut

		err_pre = err_list[0]
		
		for iter_t in range(self.n_iters):
			if self.control_para_obj.momentum_on:
				print('\nThe %s-th iteration of Accelerated FGD' % str(iter_t+1))
			else:
				print('\nThe %s-th iteration of basic FGD_Classic' % str(iter_t+1))
			
			if iter_t == 0:
				g_total_t = init_g
				gt = init_g
			else:
				adjust_on = False
				g_total_t,gt,_ = get_weight(dist_t,weight_obj,weight_kernel,partial_kernel,adjust_on)
			
			#h = get_weighted_stepsize(M,g_total_t) 

			if self.control_para_obj.momentum_on:
				if iter_t == 0:
					wt = np.sqrt(g_total_t)
				else:
					wt = weight(M,Ut,0.001)	
				
				Ut, theta_t = self._update_U_momentum(M,g_total_t,Ut,Put,theta_t,stepsize = h)
				
				projectedM_obj.update_U(theta_t) #update theta_t for weights
				#projectedM_obj.update_U(Ut) #update theta_t for weights
			else:

				Ut = self._update_U_1st_GD(M,g_total_t,Ut,Put,stepsize = h)

				projectedM_obj.update_U(Ut) #update Ut

			Put = projectedM_obj.get_Pu() # update Pu
				
			dist_t = projectedM_obj.dist_Pu(M) #update distance
				
			if self.control_para_obj.momentum_on:
				errt, _ = residue(M,Ut,np.dot(Ut.T,M),wt)	
				#errt = residue_w(dist_t,gt) #update error	
				err_g_list.append((np.amax(wt)**2,np.amin(wt)**2))
			else:			

				errt = residue_w(dist_t,gt) #update error
				err_g_list.append((np.amax(gt),np.amin(gt)))

			err_list.append(errt)			

			delta_errt = abs(err_pre - errt)
				
			err_pre = errt

			print('Current err is %(error).3f and the improvement is %(delta).3f' % {'error':errt,'delta':delta_errt})

			if delta_errt < self.delta_err_min:
				print('\nBreak the loop at %d-th iteration, since little improvement!' % iter_t)
				break

		if self.control_para_obj.momentum_on:
			g = np.square(wt)
		else:
			g = gt

		return np.array(Ut), np.array(g),err_list, err_g_list

	def apply_RW_ALM_FGD(self,weight_kernel,partial_kernel,n_inner_loop,oscillation_control = 10):
		
		'''
		Apply the classic ALM framework of GD.
		For each weight, we iteratively update Ut by GD until converged or 
		max iteration number reached.
		'''

		M = self.data

		projectedM_obj, weight_obj, err_list, init_g, dist_t, h = self._initialize_FGD(M,weight_kernel)

		Ut = self.init_U

		Put = projectedM_obj.get_Pu()

		g_list = [(np.amax(init_g),np.mean(init_g),np.amin(init_g))]#[np.amin(init_g)]
		#print('The initial max-min of weights are: ', err_g_list[0])

		err_pre = err_list[0]
		
		for iter_t in range(self.n_iters):
			if self.control_para_obj.momentum_on:
				print('\nThe %s-th iteration of momentum ALM FGD' % str(iter_t+1))
			else:
				print('\nThe %s-th iteration of normal ALM FGD' % str(iter_t+1))
			
			if iter_t == 0:
				gt = init_g
			else:
				_,gt,_ = get_weight(dist_t,weight_obj,weight_kernel,partial_kernel,False)
			
			#h = get_weighted_stepsize(M,g_total_t) 

			Ut = self._update_U_innerloop(M,gt,Ut,Put,n_inner_loop)

			projectedM_obj.update_U(Ut) #update Ut

			Put = projectedM_obj.get_Pu() # update Pu
				
			dist_t = projectedM_obj.dist_Pu(M) #update distance

			errt = residue_w(dist_t,gt) #update error
			
			max_in_pre_errlist = max(err_list[-oscillation_control:])
	
			delta_errt = abs(err_pre - errt)
			
			index_iter = iter_t + 1

			if delta_errt < self.delta_err_min:
				print('\nBreak the loop at %d-th iteration, since little improvement!' % index_iter)
				break	
			elif errt > max_in_pre_errlist:
				print('\n Oscillation begins at %d-th iteration! Forced to stop!' % index_iter)
				break
			else:
				# record the residues and weights infor
				err_list.append(errt)		
	
				g_list.append((np.amax(gt),np.mean(gt),np.amin(gt)))					
					
				err_pre = errt

				print('--Current err is %(error).3f and the improvement is %(delta).3f' % {'error':errt,'delta':delta_errt})			
			
			'''
			max_in_pre_errlist = max(err_list[-oscillation_control:])

			err_list.append(errt)		

			g_list.append((np.amax(gt),np.mean(gt),np.amin(gt)))	

			delta_errt = abs(err_pre - errt)
				
			err_pre = errt

			print('Current err is %(error).3f and the improvement is %(delta).3f' % {'error':errt,'delta':delta_errt})

			if delta_errt < self.delta_err_min or errt >= max_in_pre_errlist:
				print('\nBreak the loop at %d-th iteration, since little improvement!' % iter_t)
				break
			'''
		return np.array(Ut), np.array(gt),err_list, g_list
		

class RWLR_Model_Gamma(RWLR_Model_Betta):

	def __init__(self,data,X0,idxes0,control_para_obj,Y0T = None,
		delta_err_min = 1.e-4,damping = 0.001,n_iters = 500,logger = None):

		super().__init__(data,X0,idxes0,control_para_obj,Y0T,delta_err_min,damping,n_iters,logger)



	def apply_RW_Basic_FGD_relaxed(self,weight_kernel,partial_kernel):
		'''
		Using relaxed weight updating in this method.
		'''
		M = self.data

		projectedM_obj, weight_obj, err_list, init_g, dist_t, h = super()._initialize_FGD(M,weight_kernel)

		Ut = self.init_U

		Put = projectedM_obj.get_Pu()

		err_g_list = [(np.amax(init_g),np.amin(init_g))]
		print('The initial max-min of weights are: ', err_g_list[0])

		# initialize Weight_Generator

		w_processor = Weight_Generator(weight_obj,weight_kernel,partial_kernel,gd_mode = False)


		err_pre = err_list[0]

		dist_pre = dist_t

		for iter_t in range(self.n_iters):

			print('\nThe %s-th iteration of basic FGD_Classic using relaxation' % str(iter_t+1))

			if iter_t == 0:
				g_total_t = init_g
			else:
				w_processor.update_dist_t(dist_pre)
				g_total_t = w_processor.get_total_weight()

			print('---The max g is ',np.amax(g_total_t),' the min g is ', np.amin(g_total_t))
			
			#h = get_weighted_stepsize(M,g_total_t) 

			Ut = super()._update_U_1st_GD(M,g_total_t,Ut,Put,stepsize = h)

			projectedM_obj.update_U(Ut) #update Ut

			Put = projectedM_obj.get_Pu() # update Pu

			dist_pre = dist_t #record the previous dist
				
			dist_t = projectedM_obj.dist_Pu(M) #update distance

			gt = weight_obj.get_kernel(weight_kernel) #current weight

			weight_obj.update_dist(dist_t) #update weight_obj

			#gt = weight_obj.get_kernel(weight_kernel) #current weight

			errt = residue_w(dist_t,gt) #update error
			err_g_list.append((np.amax(gt),np.amin(gt)))

			err_list.append(errt)			

			delta_errt = abs(err_pre - errt)
				
			err_pre = errt

			print('---Current err is %(error).3f and the improvement is %(delta).3f' % {'error':errt,'delta':delta_errt})

			if delta_errt < self.delta_err_min:
				print('\nBreak the loop at %d-th iteration, since little improvement!' % iter_t)
				break
		
		g = gt

		return np.array(Ut), np.array(g),err_list, err_g_list

	def apply_RW_AGD_NC(self,weight_kernel,partial_kernel,U_update_method,n_inner_loop,oscillation_control = 5):
		'''
		Using Accelerated Alternative GD:
			First apply ALM, then apply AGD on F(x)= g(x).T*d(x)
			Pick the solution if residue reduced
		'''
		print('********\nStarting the AGD-type ALM FGD---Accelerate Basis********\n')
		M = self.data

		projectedM_obj, weight_obj, err_list, init_g, dist_t, _ = self._initialize_FGD(M,weight_kernel)

		Ut = self.init_U

		gt = init_g

		Put = projectedM_obj.get_Pu()

		g_list = [(np.amax(init_g),np.mean(init_g),np.amin(init_g))]#[np.amin(init_g)]

		w_processor = Weight_Generator(weight_obj,weight_kernel,partial_kernel)

		err_pre = err_list[0]

		kappa_pre = 0
		kappa_t = 1
		
		U_pre = Ut
		
		for iter_t in range(self.n_iters):
			
			print('\nThe %s-th iteration of Accelerated ALM FGD' % str(iter_t+1))
			
			#h = get_weighted_stepsize(M,g_total_t) 

			# ALM update U
			if U_update_method == '1st':	
				Ut = self._update_U_innerloop(M,gt,Ut,Put,n_inner_loop)
			elif U_update_method == 'Critical':			
				Ut = self._update_U_CriticalPoint(M,gt,Ut)

			projectedM_obj.update_U(Ut) #update Ut
			Put = projectedM_obj.get_Pu() # update Pu
				
			dist_t = projectedM_obj.dist_Pu(M) #update distance

			w_processor.update_dist_t(dist_t)
			gt = w_processor.get_kernel_weight()

			#_,gt,_ = get_weight(dist_t,weight_obj,weight_kernel,partial_kernel,False)

			err_t = residue_w(dist_t,gt)
			print('----The ALM err is ', err_t)
			
			'''
			err_pre = record_error(err_pre,err_t,gt,iter_t+1,oscillation_control,self.delta_err_min,err_list,g_list)

			if err_pre < 0:
				break
			'''
			
			# Acceleration
			betta_t = (kappa_pre - 1)/kappa_t
			#k = iter_t +1
			#betta_t = k/(k+3)

			#Zt = normal_col_l2(Ut + betta_t*(Ut - U_pre))
			Zt = Ut + betta_t*(Ut - U_pre)
			
			U_pre = Ut
			
			projectedM_obj.update_U(Zt) #update Ut as accelerated Zt
			Pzt = projectedM_obj.get_Pu() # update Pu
				
			dist_zt = projectedM_obj.dist_Pu(M) #update distance

			w_processor.update_dist_t(dist_zt)
			g_zt = w_processor.get_kernel_weight()

			#_,g_zt,_ = get_weight(dist_zt,weight_obj,weight_kernel,partial_kernel,False)

			err_zt = residue_w(dist_zt,g_zt)
			#err_zt = residue_w(dist_zt,gt)
			print('----The accelerated err is ', err_zt)

			if err_zt < err_t:
				print('------Found a better point! Improvment is ', err_t - err_zt)
				Ut = Zt
				dist_t = dist_zt
				gt = g_zt
				Put = Pzt
				err_t = err_zt

			kappa_pre = kappa_t
			kappa_t = (math.sqrt(kappa_t**2+1)+1)/2
			
			
			err_pre = record_error(err_pre,err_t,gt,iter_t+1,oscillation_control,self.delta_err_min,err_list,g_list)
			if err_pre < 0:
				break
			
			
		return np.array(Ut), np.array(gt),err_list, g_list							


	def apply_RW_AGD_NC_Beta(self,weight_kernel,partial_kernel,U_update_method,n_inner_loop,oscillation_control = 5):
		'''
		Using Accelerated Alternative GD:
			First apply ALM, then apply AGD on F(x)= g(x).T*d(x)
			Pick the solution if residue reduced
		'''
		print('********\nStarting the AGD-type ALM FGD ----- Accelerate Weights and Basis ********\n')
		M = self.data

		projectedM_obj, weight_obj, err_list, init_g, dist_t, _ = self._initialize_FGD(M,weight_kernel)

		Ut = self.init_U

		gt = init_g

		Put = projectedM_obj.get_Pu()

		g_list = [(np.amax(init_g),np.mean(init_g),np.amin(init_g))]#[np.amin(init_g)]

		w_processor = Weight_Generator(weight_obj,weight_kernel,partial_kernel)

		err_pre = err_list[0]

		kappa_pre = 0
		kappa_t = 1
		
		for iter_t in range(self.n_iters):
			
			print('\nThe %s-th iteration of Accelerated ALM FGD' % str(iter_t+1))
			
			U_pre = Ut

			g_pre = gt
			
			#h = get_weighted_stepsize(M,g_total_t) 

			# ALM update U
			if U_update_method == '1st':	
				Ut = self._update_U_innerloop(M,gt,Ut,Put,n_inner_loop)
			elif U_update_method == 'Critical':
				Ut = self._update_U_CriticalPoint(M,gt,Ut)

			projectedM_obj.update_U(Ut) #update Ut
			Put = projectedM_obj.get_Pu() # update Pu
				
			dist_t = projectedM_obj.dist_Pu(M) #update distance

			w_processor.update_dist_t(dist_t)
			gt = w_processor.get_kernel_weight()

			#_,gt,_ = get_weight(dist_t,weight_obj,weight_kernel,partial_kernel,False)

			err_t = residue_w(dist_t,gt)
			print('----The ALM err is ', err_t)
			if iter_t > 0:
				# Acceleration
				betta_t = (kappa_pre - 1)/kappa_t
				#k = iter_t +1
				#betta_t = k/(k+3)

				gz_t = gt + betta_t*(gt - g_pre)
				
				err_gzt = residue_w(dist_t,gz_t)
				
				
				print('----The accelerated-weight err is ', err_gzt)

				if err_gzt < err_t:
					print('------Found a better point after accelerating weights! Improvment is ', err_t - err_gzt)
	
					gt = gz_t
					#err_t = err_gzt
					
					U_pre = Ut
					
					if U_update_method == '1st':	
						Ut = self._update_U_innerloop(M,gt,Ut,Put,n_inner_loop)
					elif U_update_method == 'Critical':						
						Ut = self._update_U_CriticalPoint(M,gt,Ut)
						
					projectedM_obj.update_U(Ut)
					dist_t = projectedM_obj.dist_Pu(M)
					#gt = w_processor.get_kernel_weight()
					err_t = residue_w(dist_t,gt)
					
					
				Zt = Ut + betta_t*(Ut - U_pre)
				
				projectedM_obj.update_U(Zt) #update Ut as accelerated Zt
				Pzt = projectedM_obj.get_Pu() # update Pu
					
				dist_zt = projectedM_obj.dist_Pu(M) #update distance

				w_processor.update_dist_t(dist_zt)
				g_zt = w_processor.get_kernel_weight()

				#_,g_zt,_ = get_weight(dist_zt,weight_obj,weight_kernel,partial_kernel,False)
				
				err_zt = residue_w(dist_zt,g_zt)
				#err_zt = residue_w(dist_zt,gt)
				print('----The accelerated err of accelerating basis is ', err_zt)

				if err_zt < err_t:
					print('------Found a better point after accelerating basis! Improvment is ', err_t - err_zt)
					Ut = Zt
					dist_t = dist_zt
					gt = g_zt
					Put = Pzt
					err_t = err_zt
	
				#else:
				#	projectedM_obj.update_U(Ut) #set back to Ut

			kappa_pre = kappa_t
			kappa_t = (math.sqrt(kappa_t**2+1)+1)/2
			
			max_in_pre_errlist = max(err_list[-oscillation_control:])
	
			delta_errt = abs(err_pre - err_t)
			
			index_iter = iter_t + 1

			if delta_errt < self.delta_err_min:
				print('\nBreak the loop at %d-th iteration, since little improvement!' % index_iter)
				break	
			elif err_t > max_in_pre_errlist:
				print('\n Oscillation begins at %d-th iteration! Forced to stop!' % index_iter)
				break
			else:
				# record the residues and weights infor
				err_list.append(err_t)		
	
				g_list.append((np.amax(gt),np.mean(gt),np.amin(gt)))					
					
				err_pre = err_t

				print('--Current err is %(error).3f and the improvement is %(delta).3f' % {'error':err_t,'delta':delta_errt})
		return np.array(Ut), np.array(gt),err_list, g_list			
				

class RWLR_Model_Theta(RWLR_Model_Gamma):

	def update_intials(self,U0_new,idxes0_new):
		self.init_U = U0_new
		self.init_idxes = idxes0_new
		self.k = U0_new.shape[1]

	def apply_RW_AGD_NC_2(self,weight_kernel,partial_kernel,U_update_method,n_inner_loop,oscillation_control = 5):
		'''
		Using Accelerated Alternative GD:
			First apply ALM, then apply AGD on F(x)= g(x).T*d(x)
			Pick the solution if residue reduced
		'''
		#print('********\nStarting the AGD-type ALM FGD---Accelerate Basis Plus********\n')
		self.logger.info('********Starting the AGD-type ALM FGD---Accelerate Basis Plus********\n')
		M = self.data

		projectedM_obj, weight_obj, err_list, init_g, dist_t, _ = self._initialize_FGD(M,weight_kernel)

		Ut = self.init_U

		gt = init_g

		Put = projectedM_obj.get_Pu()

		g_list = [(np.amax(init_g),np.mean(init_g),np.amin(init_g))]#[np.amin(init_g)]

		w_processor = Weight_Generator(weight_obj,weight_kernel,partial_kernel)

		err_pre = err_list[0]

		kappa_pre = 0
		kappa_t = 1
		
		U_pre = Ut
		g_pre = gt
		
		for iter_t in range(self.n_iters):
			
			self.logger.info('The %s-th iteration of Accelerated ALM FGD' % str(iter_t+1))
			
			#h = get_weighted_stepsize(M,g_total_t) 

			# ALM update U
			if U_update_method == '1st':	
				Ut = self._update_U_innerloop(M,gt,Ut,Put,n_inner_loop)
			elif U_update_method == 'Critical':			
				Ut = self._update_U_CriticalPoint(M,gt,Ut)

			projectedM_obj.update_U(Ut) #update Ut
			Put = projectedM_obj.get_Pu() # update Pu
				
			dist_t = projectedM_obj.dist_Pu(M) #update distance

			w_processor.update_dist_t(dist_t)
			gt = w_processor.get_kernel_weight()

			#_,gt,_ = get_weight(dist_t,weight_obj,weight_kernel,partial_kernel,False)

			err_t = residue_w(dist_t,gt)
			self.logger.debug('----The ALM err is ', err_t)
			
			# Acceleration
			betta_t = (kappa_pre - 1)/kappa_t
			#k = iter_t +1
			#betta_t = k/(k+3)

			#Zt = normal_col_l2(Ut + betta_t*(Ut - U_pre))
			Zt = Ut + betta_t*(Ut - U_pre)
			gt_prime = gt + betta_t*(gt - g_pre)

			U_pre = Ut
			g_pre = gt
			
			projectedM_obj.update_U(Zt) #update Ut as accelerated Zt
			Pzt = projectedM_obj.get_Pu() # update Pu
				
			dist_zt = projectedM_obj.dist_Pu(M) #update distance

			w_processor.update_dist_t(dist_zt)
			g_zt = w_processor.get_kernel_weight()

			#_,g_zt,_ = get_weight(dist_zt,weight_obj,weight_kernel,partial_kernel,False)

			err_zt = residue_w(dist_zt,g_zt)
			err_zt_prime = residue_w(dist_zt,gt_prime)

			if err_zt_prime < err_zt:
				self.logger.debug('----Change the weights')
				g_zt = gt_prime
				err_zt = err_zt_prime

			#err_zt = residue_w(dist_zt,gt)
			self.logger.debug('----The accelerated err is ', err_zt)

			if err_zt < err_t:
				self.logger.debug('------Found a better point! Improvment is ', err_t - err_zt)
				Ut = Zt
				dist_t = dist_zt
				gt = g_zt
				Put = Pzt
				err_t = err_zt

			kappa_pre = kappa_t
			kappa_t = (math.sqrt(kappa_t**2+1)+1)/2
			
			max_in_pre_errlist = max(err_list[-oscillation_control:])
	
			delta_errt = abs(err_pre - err_t)
			
			index_iter = iter_t+1

			if delta_errt < self.delta_err_min:
				self.logger.info('\nBreak the loop at %d-th iteration, since little improvement!' % index_iter)
				break	
			elif err_t > max_in_pre_errlist:
				self.logger.info('\n Oscillation begins at %d-th iteration! Forced to stop!' % index_iter)
				break
			else:
				# record the residues and weights infor
				err_list.append(err_t)		
	
				g_list.append((np.amax(gt),np.mean(gt),np.amin(gt)))					
					
				err_pre = err_t

				self.logger.info('--Current err is %(error).3f and the improvement is %(delta).3f' % {'error':err_t,'delta':delta_errt})

		return np.array(Ut), np.array(gt),err_list, g_list							


	def apply_RW_AGD_NC_Beta(self,weight_kernel,partial_kernel,U_update_method,n_inner_loop,oscillation_control = 5):
		'''
		Using Accelerated Alternative GD:
			First apply ALM, then apply AGD on F(x)= g(x).T*d(x)
			Pick the solution if residue reduced
		'''
		self.logger.info('\n********Starting the AGD-type ALM FGD ----- Accelerate Weights and Basis ********')
		M = self.data

		projectedM_obj, weight_obj, err_list, init_g, dist_t, _ = self._initialize_FGD(M,weight_kernel)

		Ut = self.init_U

		gt = init_g

		Put = projectedM_obj.get_Pu()

		g_list = [(np.amax(init_g),np.mean(init_g),np.amin(init_g))]#[np.amin(init_g)]

		w_processor = Weight_Generator(weight_obj,weight_kernel,partial_kernel)

		err_pre = err_list[0]

		kappa_pre = 0
		kappa_t = 1
		
		U_pre = Ut
		g_pre = gt
		
		for iter_t in range(self.n_iters):
			
			self.logger.info('The %s-th iteration of Accelerated ALM FGD\n' % str(iter_t+1))

			#g_pre = gt
			
			#h = get_weighted_stepsize(M,g_total_t) 

			# ALM update U
			if U_update_method == '1st':	
				Ut = self._update_U_innerloop(M,gt,Ut,Put,n_inner_loop)
			elif U_update_method == 'Critical':
				Ut = self._update_U_CriticalPoint(M,gt,Ut)

			projectedM_obj.update_U(Ut) #update Ut
			Put = projectedM_obj.get_Pu() # update Pu
				
			dist_t = projectedM_obj.dist_Pu(M) #update distance

			w_processor.update_dist_t(dist_t)
			gt = w_processor.get_kernel_weight()

			#_,gt,_ = get_weight(dist_t,weight_obj,weight_kernel,partial_kernel,False)

			err_t = residue_w(dist_t,gt)
			self.logger.debug('----The ALM err is %f ' % err_t)

			# Acceleration
			betta_t = (kappa_pre - 1)/kappa_t
			#k = iter_t +1
			#betta_t = k/(k+3)

			gz_t = gt + betta_t*(gt - g_pre)
			
			g_pre = gt
			
			err_gzt = residue_w(dist_t,gz_t)
			
			#switch_4_debug = True

			if err_gzt < err_t:			
								
				if U_update_method == '1st':	
					Ut_prime = self._update_U_innerloop(M,gz_t,Ut,Put,n_inner_loop)
				elif U_update_method == 'Critical':						
					Ut_prime = self._update_U_CriticalPoint(M,gz_t,Ut)
					
				projectedM_obj.update_U(Ut_prime)
				Put_prime = projectedM_obj.get_Pu() # update Pu
				
				dist_t = projectedM_obj.dist_Pu(M)
				gz_t = w_processor.get_kernel_weight()
				err_gzt = residue_w(dist_t,gz_t)
				self.logger.debug('----The accelerated-weight err is %f' % err_gzt)

				if err_gzt < err_t:
					self.logger.debug('------Found a better point after accelerating weights! Improvement is %s' % str(err_t - err_gzt))
	
					gt = gz_t
					err_t = err_gzt
					
					U_pre = Ut
					Ut = Ut_prime
					Put = Put_prime
					
				
					
			Zt = Ut + betta_t*(Ut - U_pre)
			
			U_pre = Ut
			
			projectedM_obj.update_U(Zt) #update Ut as accelerated Zt
			Pzt = projectedM_obj.get_Pu() # update Pu
				
			dist_zt = projectedM_obj.dist_Pu(M) #update distance

			w_processor.update_dist_t(dist_zt)
			g_zt = w_processor.get_kernel_weight()

			#_,g_zt,_ = get_weight(dist_zt,weight_obj,weight_kernel,partial_kernel,False)
			
			err_zt = residue_w(dist_zt,g_zt)
			#err_zt = residue_w(dist_zt,gt)
			self.logger.debug('----The accelerated err of accelerating basis is %f ' % err_zt)

			if err_zt < err_t:
				self.logger.debug('------Found a better point after accelerating basis! The improvement is %s' % str(err_t - err_zt))
				Ut = Zt
				dist_t = dist_zt
				gt = g_zt
				Put = Pzt
				err_t = err_zt

			kappa_pre = kappa_t
			kappa_t = (math.sqrt(kappa_t**2+1)+1)/2
			
			max_in_pre_errlist = max(err_list[-oscillation_control:])
	
			delta_errt = abs(err_pre - err_t)
			
			index_iter = iter_t+1

			if delta_errt < self.delta_err_min:
				self.logger.info('Break the loop at %d-th iteration, since little improvement!\n' % index_iter)
				break	
			elif err_t > max_in_pre_errlist:
				self.logger.info('Oscillation begins at %d-th iteration! Forced to stop!\n' % index_iter)
				break
			else:
				# record the residues and weights infor
				err_list.append(err_t)		
	
				g_list.append((np.amax(gt),np.mean(gt),np.amin(gt)))					
					
				err_pre = err_t

				self.logger.info('--Current err is %(error).3f and the improvement is %(delta).3f' % {'error':err_t,'delta':delta_errt})

		return np.array(Ut), np.array(gt),err_list, g_list			
	

	