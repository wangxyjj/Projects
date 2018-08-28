function [X,W,R,err_f,R_list] = ReweightMF(M,X0,W0,R0,Err,iter_max)

err_t = 10000;
iter_t = 1;

Xt = X0;
Wt = W0;

Rt = R0;

[m,n] = size(M);

R_list = Rt;

while (err_t > Err) && (iter_t < iter_max)
    
    Rprev = Rt;
    
    % update W
    ProjXt = Xt*Xt';
    DistM = M - ProjXt*M;
    DistMj = sum(DistM.^2);
    %DistMj(DistMj<0.0001) = DistMj(DistMj<0.0001)+0.01;%damping
    DistMj = DistMj + 0.001;%damping
    Wt = DistMj/sum(DistMj);
    Wt = sqrt(log(1./Wt));
  
    % update X   
    Yt = Xt'* M;
      
    DwY = bsxfun(@times,Wt,Yt);
    
    for i = 1:m
        mj = M(i,:).* Wt;
        Xt(i,:) = mj*DwY'/(DwY*DwY');       
    end       
    
    Rt = norm(bsxfun(@times,Wt,M - Xt*Xt'*M),'fro');
    
    R_list = [R_list Rt];
    
    err_t = abs(Rt-Rprev)
    
    iter_t =iter_t+1;
end    

X = Xt;
W = Wt;
err_f = err_t;
R = Rt;



