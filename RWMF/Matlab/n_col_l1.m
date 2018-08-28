function M_new = n_col_l1(M)
%l1 norm sense normalization by column 
    D = diag(1./sum(M)); 
    M_new = M*D; 
