import sys
import numpy as np
from numpy.matlib import eye
import math

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from sklearn import linear_model
#from sklearn.metrics import mean_squared_error, r2_score

def residue(M,X,YT,W):
	Xm = np.matrix(X)
	YTm = np.matrix(YT)
	Mk = Xm*YTm
	deltaM = np.multiply((M-Mk),W)
	return np.linalg.norm(deltaM,'fro'), np.array(Mk)

def weight(M,X,damping):
	X = np.matrix(X)
	M = np.matrix(M)
	deltaM = M - (X*X.T)*M
	if damping != None:
		distMj = np.sum(np.square(deltaM),axis=0)+damping
	else:
		distMj = np.sum(np.square(deltaM),axis=0)	
	W = distMj/np.sum(distMj)
	W = np.sqrt(np.log(1/W))
	
	return np.array(W)

def distance(M,P,damping):
	P = np.matrix(P)
	M = np.matrix(M)
	deltaM = M - P*M
	if damping != None:
		distMj = np.sum(np.square(deltaM),axis=0)+damping
	else:
		distMj = np.sum(np.square(deltaM),axis=0)	

	return np.array(distMj), np.asscalar(np.sum(distMj))


def ReweightMF(M,X0,err0,damping,delta_err_min,iter_max):

	iter_t = 1

	Xt = np.matrix(X0)
	M = np.matrix(M)

	errt = err0
	#errt = np.linalg.norm(M - (Xt*Xt.T)*M,'fro')
	err_list = [errt]
	
	delta_errt = 1000

	while (delta_errt > delta_err_min) and (iter_t <= iter_max):
		print('\n*******This is %s-th iteration of classic method******' % str(iter_t))

		err_pre = errt

		# update W
		Wt = weight(M,Xt,damping)


		# update Y
		YtT = Xt.T*M

		# update X
		Mw = np.multiply(M,Wt)
		
		YTDw = np.multiply(YtT,Wt)
		B = (YTDw*YTDw.T).I  # B = inv(YtTDwYt)
		
		Xt = Mw*YTDw.T*B

		# calculate the t-th residue
		
		#Mkt = (Xt*Xt.T)*M
		#errt = np.linalg.norm(M - Mkt,'fro')
		errt, Mkt = residue(M,Xt,YtT,Wt)
		print('\n The residue is', errt)
		
		err_list.append(errt)
		# check the convergence
		delta_errt = abs(err_pre - errt)
		print('\n The residue difference is', delta_errt)

		iter_t += 1


	return np.array(Xt), Wt, np.array(Mkt), np.array(err_list)


def ReweightMF_adjust(M,X0,err0,damping,delta_err_min,iter_max):
	'''
	In this gradient method, we express weights as a function of U and 
	the gradient is taken from J = g(U)d(U) in terms of U. 

	Using 2nd order method f' = 0
	'''
	delta_errt = 10000
	iter_t = 1
	
	
	M = np.matrix(M)
	Ut = np.matrix(X0)
	
	errt = err0

	#errt = np.linalg.norm(M - (Ut*Ut.T)*M,'fro')
	err_list = [errt]

	while (delta_errt > delta_err_min) and (iter_t <= iter_max):
		print('\n*******This is %s-th iteration of our method******' % str(iter_t))
		err_pre = errt

		# calculate the gradient
		deltaM = M - (Ut*Ut.T)*M

		if damping != None:

			distMj = np.sum(np.square(deltaM),axis=0)+damping

		else:

			distMj = np.sum(np.square(deltaM),axis=0)

		dvS = distMj/np.sum(distMj)	
		g = np.log(1/dvS)

		grad = g+dvS - np.ones(dvS.shape)

		Vt = M.T*Ut

		B = (np.multiply(Vt.T,grad)*Vt).I

		Ut = np.multiply(M,grad)*Vt*B

		# calculate the t-th residue
		#Mkt = (Ut*Ut.T)*M
		#errt = np.linalg.norm(M - Mkt,'fro')
		errt, Mkt = residue(M,Ut,Ut.T*M,np.sqrt(g))
		print('\n The residue is', errt)
		

		err_list.append(errt)
		#check the convergence
		delta_errt = abs(err_pre - errt)
		print('\n The residue difference is', delta_errt)
		
		iter_t += 1

	return np.array(Ut), np.array(g), np.array(grad),np.array(Mkt), np.array(err_list)

def CompareGrad(M,X0,err0,damping,n_rounds):
	'''
		comparing the effects of two gradients using 1st order derivative 
		grad1 is the adjusted
		grad2 is the classical
		step size using 1/L, L = max sigular
	'''
	M = np.matrix(M)
	X0 = np.matrix(X0)

	rounds = n_rounds
	
	X1 = X0
	X2 = X0

	P1 = X0*X0.T
	P2 = P1
	
	m = X0.shape[0] 

	J1 =[err0]
	J2 =[err0]

	for i in range(rounds):
		#calculate gradient using best weight
		distMj2, S2 = distance(M,P2,damping)
		w2 = np.sqrt(np.log(S2/distMj2))
		print('\n Classic: \n The max w2 is ',np.amax(w2),
			' \n the min w2 is ', np.amin(w2))
		Mw2 = np.multiply(M,w2)
		MwM2 = Mw2*Mw2.T
		h2 = 1/np.linalg.norm(MwM2,2)
		grad2 = (eye(m) - P2)*MwM2

		X2 = X2 + h2*grad2*X2
		
		P2 = X2*X2.T
		
		res2 = np.linalg.norm(Mw2 - P2*Mw2,'fro')

		J2.append(res2)

		#calculate gradient using joint weight
		distMj1, S1 = distance(M,P1,damping)
		dvS1 = distMj1/S1	
		g1 = np.log(1/dvS1)
		print('\n Adjust: \n The max w1 is ',math.sqrt(np.amax(g1)),
			' \n the min w1 is ', math.sqrt(np.amin(g1)))
		w1 = np.sqrt(g1+dvS1 - 1.5*np.ones(dvS1.shape))	
		print('\n The max w_total1 is ',np.amax(w1),
			' \n the min w_total1 is ', np.amin(w1))
		Mw1 = np.multiply(M,w1)
		MwM1 = Mw1*Mw1.T
		h1 = 1/np.linalg.norm(MwM1,2)
		grad1 = (eye(m) - P1)*MwM1

		X1 = X1 + h1*grad1*X1
		P1 = X1*X1.T
		
		res1 = np.linalg.norm(np.multiply(M - P1*M,np.sqrt(g1)),'fro')

		J1.append(res1)
		
		print('The ',i+1,'-th iteration finished.\n')

	return np.array(J1),np.array(J2)
	



def ReweightMF_adjust2(M,X0,err0,damping,delta_err_min,iter_max):
	'''
	In this gradient method, we express weights as a function of U and 
	the gradient is taken from J = g(U)d(U) in terms of U. 
	'''
	delta_errt = 10000
	iter_t = 1
	
	
	M = np.matrix(M)
	Ut = np.matrix(X0)
	
	errt = err0

	#errt = np.linalg.norm(M - (Ut*Ut.T)*M,'fro')
	err_list = [errt]

	obj_value = [np.linalg.norm(M-Ut*Ut.T*M,'fro')]

	while (delta_errt > delta_err_min) and (iter_t <= iter_max):
		err_pre = errt

		# calculate the gradient
		deltaM = M - (Ut*Ut.T)*M

		if damping != None:

			distMj = np.sum(np.square(deltaM),axis=0)+damping

		else:

			distMj = np.sum(np.square(deltaM),axis=0)

		dvS = distMj/np.sum(distMj)	
		g = np.log(1/dvS)

		grad = g+dvS - np.ones(dvS.shape)

		Vt = M.T*Ut

		B = (np.multiply(Vt.T,grad)*Vt).I

		Ut = np.multiply(M,grad)*Vt*B

		# calculate the t-th residue
		#Mkt = (Ut*Ut.T)*M
		#errt = np.linalg.norm(M - Mkt,'fro')
		errt, Mkt = residue(M,Ut,Ut.T*M,np.sqrt(g))

		objt = np.sum(np.multiply(grad,distMj))
		
		obj_value.append(objt)

		err_list.append(errt)
		#check the convergence
		delta_errt = abs(err_pre - errt)

		iter_t += 1

	return np.array(Ut), np.array(g), np.array(grad),np.array(Mkt), np.array(err_list)


def checkPSD(M,U,grad,damping):
	'''
	verify a square matrix is PSD or not
	'''

	U = np.matrix(U)
	M = np.matrix(M)

	Pu = U*U.T
	deltaM = M - Pu*M

	if damping != None:

		distMj = np.sum(np.square(deltaM),axis=0)+damping

	else:

		distMj = np.sum(np.square(deltaM),axis=0)

	Sinv = 1/np.sum(distMj)
	Sinv2 = Sinv**2


	coeff1 = np.array(2*Sinv - 1/distMj - distMj*Sinv2)

	H1 = np.multiply(M,grad)*M.T

	H2 = (deltaM[:,0]*M[:,0].T)**2

	H2 = np.multiply(coeff1[:,0],H2)

	i = 1
	while i< grad.shape[1]:
		H2_i = (deltaM[:,i]*M[:,i].T)**2
		H2_i = np.multiply(coeff1[:,i],H2_i)
		H2 += H2_i
		i+=1

	H = H1+H2

	trace_H= np.trace(H)

	determin = np.linalg.det(H)

	if (determin>=0) and (trace_H>=0):
		return True, H
	else:
		return False, H


def draw_graph3D_v1(M,Mk,X,X0,W,W0):
	'''
		plot the 3D graph about M and Mk
		!!!!Untested, Undebuged!!!!
	'''

	if M.ndim > 3: 
		return False
	else:
		n = W.shape[1]
		I_sort = np.argsort(W)[::-1]
		portion = math.floor(n*0.5)
		I_s = I_sort[:portion]
		Ms = M[:,I_s]
		Mw = M[:,I_sort]

		# plot two hyperplanes respectively spanned by
		# [v1,v2](the best k approx) and 
		# [v3,v4] (the reweighted k-dim subspace) 
		# 

		v1 = X0[:,0]; v2= X0[:,1];
		v1 = v1/math.sqrt(np.sum(v1**2))
		v2 = v2/math.sqrt(np.sum(v2**2))

		v3 = X[:,0]; v4= X[:,1];
		v3 = v3/math.sqrt(np.sum(v3**2))
		v4 = v4/math.sqrt(np.sum(v4**2))

		grids = np.arange(-5, 5, 0.1)

		x1, x2 = np.meshgrid(grids, grids)

		plane1_x = x1*v1[0]+x2*v2[0]
		plane1_y = x1*v1[1]+x2*v2[1]
		plane1_z = x1*v1[2]+x2*v2[2]

		plane2_x = x1*v3[0]+x2*v4[0]
		plane2_y = x1*v3[1]+x2*v4[1]
		plane2_z = x1*v3[2]+x2*v4[2]


		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		ax.scatter(Mw[0,:], Mw[1,:], Mw[2,:], c='r', marker='o')
		ax.scatter(Ms[0,:], Ms[1,:], Ms[2,:])

		ax.plot_surface(plane1_x, plane1_y, plane1_z)
		ax.plot_surface(plane2_x, plane2_y, plane2_z)

		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')

		plt.show()



def byebye(e_info):
	'''Kill Program'''
	print('Crucial error: %s. Abort now!' % e_info)
	sys.exit(e_info)	


def Alter_OMF(M,Xt,delta_err_min,iter_max):
	# Alternative ordinary matrix factorization using SKlearn
	# !!!!! Actually, does not work properly !!!!

	M = np.matrix(M)
	Xt = np.matrix(Xt)

	delta_errt = 10000
	iter_t = 1

	# Create linear regression object
	regr = linear_model.LinearRegression()
	
	# OLS fixing X, solve Y for min||M-XY||_2
	regr.fit(Xt, M)
	Yt = np.matrix(regr.coef_)
	Yt = Yt.T
	
	err0 = np.linalg.norm(M-Xt*Yt,'fro')
	errt = err0
	err_list = []

	while (delta_errt > delta_err_min) and (iter_t <= iter_max):

		err_pre = errt
		
		# OLS fixing Y, solve X for min||Y.TX.T -M.T||_2
		regr.fit(Yt.T, M.T)
		Xt = np.matrix(regr.coef_)

		# OLS fixing X, solve Y for min||M-XY||_2
		regr.fit(Xt, M)
		Yt = np.matrix(regr.coef_)
		Yt = Yt.T
		
		errt = np.linalg.norm(M-Xt*Yt,'fro')
		if iter_t > 1:
			err_list.append(errt)

		delta_errt = err_pre - errt

		iter_t += 1 

	return Xt.A,Yt.A,errt,np.array(err_list)

def Alter_OMF2(M,Xt,delta_err_min,iter_max):
	# Alternative ordinary matrix factorization
	# By Hessian f' =0 

	M = np.matrix(M)
	Xt = np.matrix(Xt)

	delta_errt = 10000
	iter_t = 1
	
	Yt = (Xt.T*Xt).I*Xt.T*M
	print(str(Xt.T*Xt))

	err0 = np.linalg.norm(M-Xt*Yt,'fro')
	errt = err0
	err_list = []
	# Create linear regression object

	while (delta_errt > delta_err_min) and (iter_t <= iter_max):

		err_pre = errt

		XtT = Xt.T
		
		#fixing Y, solve X for min||Y.TX.T -M.T||_2
		Xt = (Yt*Yt.T).I*Yt*M.T
		Xt = Xt.T
		print(str(iter_t)+' th res for X is '+str(np.linalg.norm(M-Xt*Yt)))
		# fixing X, solve Y for min||M-XY||_2
		Yt = (XtT*Xt).I*XtT*M
		#print(str(iter_t)+' th res for Y is '+str(np.linalg.norm(M-Xt*Yt)))
		
		errt = np.linalg.norm(M-Xt*Yt,'fro')
		if iter_t >1:
			err_list.append(errt)

		delta_errt = err_pre - errt

		#Yt = Yt1
		#Xt = Xt1
		iter_t += 1 

	return Xt.A,Yt.A,errt,np.array(err_list)


def OMF_Pu(M,Ut,delta_err_min,iter_max):
	# Ordinary matirx factorization using Pu with 1st order gradient

	M = np.matrix(M)
	Ut = np.matrix(Ut)
	Put = Ut@Ut.T # @ is the specific operator syntax standing for matrix inner product, equaling to *. Python 3.5+ 

	m,_ = M.shape

	delta_errt = 10000
	iter_t = 1

	err0 = np.linalg.norm(M-Put*M,'fro')
	errt = err0
	err_list = []

	M_tilde = M*M.T

	h = 1/np.linalg.norm(M_tilde,2)
	kappa_list = []

	while (delta_errt > delta_err_min) and (iter_t <= iter_max):

		err_pre = errt
		grad = (eye(m) - Put)*M_tilde

		Ut = Ut + h*grad*Ut
		sum_col = np.sqrt(np.sum(np.square(Ut),axis=0))
		Ut = Ut/sum_col		
		Put = Ut*Ut.T
		#Ut, _ = np.linalg.qr(Ut)

		#Put,_ = np.linalg.qr(Ut*Ut.T)
		#sum_col = np.sqrt(np.sum(np.square(Put),axis=0))
		#Put = Put/sum_col
		
		#print(str(iter_t)+' th rank of Pu is '+str(np.linalg.matrix_rank(Put)))

		_,sig,_ = np.linalg.svd(Ut.T*M*M.T*Ut)
		sig[sig<=0] = 1.e-4
		kappa = np.amax(sig)/np.amin(sig)
		print(str(iter_t)+' th Kappa is '+str(kappa))

		errt = np.linalg.norm(M-Put*M,'fro')
		if iter_t >2:
			err_list.append(errt)
			kappa_list.append(kappa)

		delta_errt = err_pre - errt

		iter_t += 1

	return Ut.A,errt,np.array(err_list),np.array(kappa_list)




	