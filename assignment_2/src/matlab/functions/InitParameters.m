function [W1,b1,W2,b2] = InitParameters(X_test,Y_train,d,m)
%INITPARAMETERS Summary of this function goes here
%   Detailed explanation goes here
    W1 = normrnd(0,1/sqrt(d),[m,size(X_test,1)]);
    b1 = zeros([m,1]);
    W2 = normrnd(0,1/sqrt(m),[size(Y_train,1),m]);
    b2 = zeros([size(Y_train,1),1]);
end

