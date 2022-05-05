function [s1,s,P,H] = ComputeNetwork(X,W1,W2,b1,b2)
%COMPUTENETWORK Summary of this function goes here
%   Detailed explanation goes here

tic
s1 = W1*X + b1;
toc
H = max(s1,0);
s = W2*H + b2;
P = exp(s)./sum(exp(s));

end

