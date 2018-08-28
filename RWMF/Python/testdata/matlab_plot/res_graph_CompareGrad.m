%close all
clear all

f = 'testrun_v2.mat';
%f = 'testCompareGrad_HW_v2.mat';
load(f)

GradJ_x = 1:length(J1);
GradJ_y = log(J1);

Grad_x = 1:length(J2);
Grad_y = log(J2);

% load('output_v1.mat')
% Newton_x = 1:length(err_list);
% Newton_y = log(err_list);
% 
% load('output_v5.mat')
% err_list3 = [err_list1 err_list2];
% NewtonJ_x = 1:length(err_list3);
% NewtonJ_y = log(err_list3);


figure

plot(Grad_x,Grad_y,'g.-','LineWidth',0.5,'MarkerSize',1);
hold on

% plot(Newton_x,Newton_y,'k--','LineWidth',1.5,'MarkerSize',15);
% hold on

plot(GradJ_x,GradJ_y,'r.-','LineWidth',0.5,'MarkerSize',1);
hold on

% plot(NewtonJ_x,NewtonJ_y,'b.-','LineWidth',1,'MarkerSize',15);
% hold on

%legend('1stGrad','2ndGrad','1stGrad\_New','2ndGrad\_New');
legend('1stGrad','1stGrad\_New');


xlabel('Iterations')
ylabel('log of Residues')