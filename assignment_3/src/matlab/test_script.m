% clc;
% A = rand(100000,3800);
% B = rand(3800,50);
% 
% tic
% A*B;
% toc
% 
% tic
% A = gpuArray(A);
% B = gpuArray(B);
% toc
% 
% tic
% A*B;
% toc

clc;

tic
w1 = gpuArray(w);
h1 = gpuArray(h);
b1 = gpuArray(b);
s1 = w1*h1+b1;
toc