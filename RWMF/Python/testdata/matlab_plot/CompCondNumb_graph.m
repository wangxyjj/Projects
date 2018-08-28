close all
clear all

f = 'testCondNumb_v9.mat';
load(f)

l1 = length(err_list1);
l2 = length(err_list2);


dim =size(X1,1); k =size(X1,2);

Residue1_x = [1:l1];
%Residue1_y = err_list1/max(err_list1);
Residue1_y = err_list1;

Residue2_x = [1:l2];
%Residue2_y = err_list2/max(err_list2);
Residue2_y = err_list2;

% %scaling to 
% Residue1_y = Residue1_y - min(Residue1_y);
% scale = (max(Residue1_y)-min(Residue1_y))/(max(Residue2_y)-min(Residue2_y));
% Residue1_y = Residue1_y*scale+min(Residue2_y);

%Residue1_y = Residue1_y - max(Residue1_y)+max(Residue2_y);




[U1,S1,V1] = svd(M1,0);
[U2,S2,V2] = svd(M2,0);

Mk1 = U1(:,1:k)*U1(:,1:k)'*M1;
Mk2 = U2(:,1:k)*U2(:,1:k)'*M2;

opt1 = norm(M1 - Mk1,'fro')
opt2 = norm(M2 - Mk2,'fro')


Mkappa1 = S1(1,1)/S1(dim,dim);
Mkappa2 = S2(1,1)/S1(dim,dim);

figure


plot(Residue1_x,Residue1_y,'r.-');
ylim([min([err_list1 err_list2])-1 max([err_list1 err_list2])])
hold on

plot(Residue2_x,Residue2_y,'b.-');
hold on

plot(l1+1,opt1/max(err_list1),'k.','MarkerSize',5);
hold on
plot(l2+1,opt2/max(err_list2),'k.','MarkerSize',5);
title('Convergence comparison')


%legend(strcat('Init \kappa of W1: ',num2str(kappa1),' Init \kappa of M1: ',num2str(Mkappa1)),...
%    strcat('Init \kappa of W2: ',num2str(kappa2),' Init \kappa of M2: ',num2str(Mkappa2)));

legend('M1','M2')

xlabel('Iterations');
ylabel('Residues');

figure
plot(Residue1_x,cn_list1, 'r.-');
hold on
plot(Residue2_x,cn_list2, 'b.-');

legend('M1','M2')
xlabel('Iterations')
ylabel('\kappa = \delta_{max} / \delta_{min}')
title('The variation of Kappa')