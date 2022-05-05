function [s1,s,P,H,u] = ComputeNetworkDropout(X,W1,W2,b1,b2,p)
%COMPUTENETWORK Summary of this function goes here
%   Detailed explanation goes here

s1 = W1*X + b1;
H = max(s1,0);
u = (rand(size(H))<p)/p;
H = H.*u;
s = W2*H + b2;
P = exp(s)./sum(exp(s));

end

