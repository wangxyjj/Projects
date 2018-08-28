C = 1; % const
mj = [1;2];
N = 100;

f_dis = zeros(1,N);
u_dis = zeros(1,N);

for ii = 1:100
    u1 = random('Normal', 1, 1, 1, 2);
    u2 = random('Normal', 1, 1, 1, 2);
    
    u1 = u1/sqrt(sum(u1.^2));
    u2 = u2/sqrt(sum(u2.^2));
    
    Pu1 = u1'*u1;
    Pu2 = u2'*u2;
    
    d1 = norm(mj - Pu1*mj)^2;
    d2 = norm(mj - Pu2*mj)^2;
    
    s1 = d1+C;
    s2 = d2+C;
    
    f1 = -C/(s1*d1) * (eye(2)-Pu1)*2*mj*mj';
    f2 = -C/(s2*d2) * (eye(2)-Pu2)*2*mj*mj';

    f_dis(ii) = norm(f1-f2,'fro');
    u_dis(ii) = norm(Pu1-Pu2,'fro');
end

f_dis./u_dis



