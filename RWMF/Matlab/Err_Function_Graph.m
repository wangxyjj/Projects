close all;
clear 

points = 0.1:0.1:5;
[X,Y] = meshgrid(points,points);

X2 = X.^2;
Y2 = Y.^2;

S = X2+Y2;

Z = S.*log(S) - X2.*log(X2)-Y2.*log(Y2);

figure()

surf(X,Y,Z)
%title('The error function');
%xlabel('x1'); ylabel(''); zlabel('f(d1,d2)')