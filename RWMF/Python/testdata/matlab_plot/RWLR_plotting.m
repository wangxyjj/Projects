close all
clear


f_a = 'MINIST_ARWLR_AGD_ALM_v17.mat';
f_al = 'MINIST_ARWLR_ALM_v17.mat';

% f_classic = 'MINIST_RWLR_Classic_v10.mat';
% f_sgd = 'data_RWLR_SGD_v1.mat';
% f_cm = 'data_RWLR_SGD_ShortW_CM_v1.mat';
% f_nag = 'data_RWLR_SGD_NAG_v0.mat';
% f_sgd_classic = 'data_RWLR_SGD_Classic_v1.mat';
% f_sgd_shortw = 'data_RWLR_SGD_ShortW_CM_Classic_v1.mat';
% 
% f_2nd = 'data_Reweight_2nd_v1.mat';
% f_2nd_classic = 'data_Reweight_2nd_Classic_v1.mat';



logflag = 0;

[x_a,y_a,y_g_a] = load_err_and_errg(f_a,logflag);
[x_al,y_al,y_g_al] = load_err_and_errg(f_al,logflag);

x_a = x_a.*30;
x_al = x_al.*30;

%y_index = 1:length(y_a)-1;
%selected = [1,y_index(mod(y_index,20)==0)+1];
%y_a = y_a(selected);
%selected = [1,y_index(mod(y_index,20)==0)];
%y_g_a = y_g_a(selected);
%x_a = 1:length(y_a);


% 
% [x_cl,y_cl,y_g_cl] = load_err_and_errg(f_classic,logflag);
% 
% [x_s,y_s] = load_coordinates(f_sgd,logflag);
% [x_cm,y_cm] = load_coordinates(f_cm,logflag);
% [x_nag,y_nag] = load_coordinates(f_nag,logflag);
% [x_s_c,y_s_c] = load_coordinates(f_sgd_classic,logflag);
% [x_ss,y_ss] = load_coordinates(f_sgd_shortw,logflag);
% 
% [x_2nd,y_2nd] = load_coordinates(f_2nd,logflag);
% [x_2nd_cl,y_2nd_cl] = load_coordinates(f_2nd_classic,logflag);




figure()

%plot(x_cl(1:length(x_cl)),y_cl(1:length(x_cl)),'b.-','LineWidth',0.5);
plot(x_a,y_a,'r+-');
hold on
plot(x_al,y_al,'.-');
hold on
%plot(x_b,y_b,'r.-','LineWidth',0.5);
%hold on
%plot(x_2nd,y_2nd,'k.-','LineWidth',0.5);
%hold on
%plot(x_2nd_cl,y_2nd_cl,'g.-','LineWidth',0.5);


%y_a - y_al

%legend('Accel','Classic-ALM')

legend('Accelerated-ALM','Classic-ALM')

%legend('Classic-ALM','Our method','Our-2nd','Classic-2nd')

xlabel('Iterations');
if logflag == 1
    ylabel('Log of Residues');
else
    ylabel('Residues');
end

% figure()
% 
% y_g_al = y_g_al(2:length(y_g_al));
% y_g_b = y_g_b(2:length(y_g_b));
% 
% plot(x_a(1:length(y_g_a)),y_g_a,'b.-','LineWidth',0.5);
% hold on
% plot(x_al(1:length(y_g_al)),y_g_al,'g.-','LineWidth',0.5);
% hold on
% plot(x_b(1:length(y_g_b)),y_g_b,'r.-','LineWidth',0.5);
% hold off
% 
% title('max weight')
% 
% legend('Accel','Accel-lite','Classic-ALM')
% xlabel('Iterations');
% ylabel('max Weight');
% 
% figure()
% %plot(x_s,y_s,'r.-');
% %hold on
% 
% %hold on
% %plot(x_nag,y_nag,'k.-');
% %hold on
% %plot(x_s_c,y_s_c,'b--','LineWidth',0.5);
% %hold on
% plot(x_ss(1:200),y_ss(1:200),'b--');
% hold on 
% plot(x_cm,y_cm,'r.-','LineWidth',0.5);
% 
% %legend('MinibatchSGD-FGD', 'MinibatchSGD-Short-CM-FGD','MinibatchSGD-Classic-FGD','MinibatchSGD-Short-CM-Classic-FGD')
% 
% %legend('MinibatchSGD-Short-CM-FGD','MinibatchSGD-Short-CM-Classic-FGD')
% 
% legend('MinibatchSGD-ClassicALM', 'MinibatchSGD-Ours')
% 
% xlabel('Iterations')
% if logflag == 1
%     ylabel('Log of Residues');
% else
%     ylabel('Residues');
% end


