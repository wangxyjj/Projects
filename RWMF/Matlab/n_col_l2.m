function M_new = n_col_l2(M)
% %l1 norm sense normalization by column 
    temp = sqrt(sum(M.^2));
    D = diag(1./temp); 
    M_new = M*D; 
