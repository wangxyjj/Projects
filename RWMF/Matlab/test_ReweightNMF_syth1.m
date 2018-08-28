% run min ||(M-XY)W|| s.t. W truch vector weight
% M obtained by 3-dim random Gaussian
% size(M) = (m,n)
% k is the low rank approximation

close all;
clear all;

mu = [0;0;0];
sigma = 5;
dim = 3;
n = 100;
%M = normrnd(repmat(mu,1,n),sigma,dim,n); % generate M by gaussian distribution

load('syth_Gaussian_100_1.mat');

k = 2;

% figure
% plot3(M(1,:),M(2,:),M(3,:),'b+');
% title('Illustration of ReweightMF on random 100 Gaussian data points');
% hold on;

[U,S,V]=svd(M,'econ');% use the best k-subspace of M as initial
X0 = U(:,1:k);
%Y0 = V(:,1:k);

% W0 = sum(M.^2)/sum(sum(M.^2));
% W0 = log(sqrt(1./W0));
% W0 = ones(1,100);

Mk0 = (X0*X0')*M;
DistM = M - Mk0;
DistMj = sum(DistM.^2);
%DistMj(DistMj<0.0001) = DistMj(DistMj<0.0001)+0.001;%damping
DistMj = DistMj + 0.001;%damping
W0 = DistMj/sum(DistMj);
W0 = sqrt(log(1./W0));



% 
% % plot the projection on best k approx
% plot3(Mk0(1,:),Mk0(2,:),Mk0(3,:),'b*'); 
% hold on
% 


deltaM0 =  bsxfun(@times,W0,M-Mk0);
residue0 = norm(deltaM0,'fro');

Err_min = 1.e-4; % set the min residue and max numb of iterations
iter_max = 500;

[X,W,residue,Err_cov,R_list] = ReweightMF(M,X0,W0,residue0,Err_min,iter_max);

Mk = (X*X')*M;

% plot3(Mk(1,:),Mk(2,:),Mk(3,:),'k.','MarkerSize',5);


% hold on

% circle the large-80%-weight points 

[~,I_sort] = sort(W,'descend');

stop = floor(n*0.5);

I_s = I_sort(1:stop);

Ms = M(:,I_s);

% plot3(Ms(1,:),Ms(2,:),Ms(3,:),'rO');

% hold on

% plot two surfaces respectively spanned by [v1,v2](the best k approx) and 
% [v3,v4] (the reweighted k-dim subspace) 

v1 = X0(:,1); v2= X0(:,2);
v1 = v1/sqrt(sum(v1.^2));
v2 = v2/sqrt(sum(v2.^2));


v3 = X(:,1); v4= X(:,2);
v3 = v3/sqrt(sum(v3.^2));
v4 = v4/sqrt(sum(v4.^2));


figure

M_w = M(:,I_sort);
C = summer(n);
scatter3(M_w(1,:),M_w(2,:),M_w(3,:),[],C,'filled');
hold on

plot3(Ms(1,:),Ms(2,:),Ms(3,:),'rO');

colormap('white')
gridsize = [-1 1 -1 1]*14;

fsurf(@(s,t)v1(1)*s+v2(1)*t,@(s,t)v1(2)*s+v2(2)*t,@(s,t)v1(3)*s+v2(3)*t,gridsize,'EdgeColor','b');
fsurf(@(s,t)v3(1)*s+v4(1)*t,@(s,t)v3(2)*s+v4(2)*t,@(s,t)v3(3)*s+v4(3)*t,gridsize,'EdgeColor','r');

hold on

legend('Original data','50% heavy-weight points','Best 2-dim surface','Reweighted surface');
%legend('Original data','50% heavy-weight points','Reweighted surface');

% Plot the 3 principles of original data 
Principle = n_col_l2(U).*16;
% for i =1:size(U,2)
%     temp = [-Principle(:,i),Principle(:,i)];
%     plot3(temp(1,:),temp(2,:),temp(3,:),'b--','linewidth',2)
%     hold on
% end    

% Only plot the third principle
temp = [-Principle(:,3),Principle(:,3)];
plot3(temp(1,:),temp(2,:),temp(3,:),'b--','linewidth',2);

hold on

v5 = cross(v3,v4);
v5 = v5*16;
temp = [-v5 v5];
plot3(temp(1,:),temp(2,:),temp(3,:),'r--','linewidth',2);


%scatter3(M_w(1,:),M_w(2,:),M_w(3,:),[],[0.5 0.5 0.5])


title(sprintf('Illustration of ReweightMF on random 100 Gaussian data points\n (Point colors decay from heavy to light as weights descend)'));
hold off;



