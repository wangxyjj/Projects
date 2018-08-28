def residue(M,X,YT,W):
	Mk = np.dot(X,YT)
	deltaM = (M-Mk)*W[:, None]
	return np.linalg.norm(deltaM,'fro')