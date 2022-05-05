function [grad_W, grad_b] = ComputeGradientsMultiple(X, Y, P, W, lambda)

% grad W is gradient matrix of the cost J relative to W and has size K×d.
% grad b is gradient vector of the cost J relative to b and has size K×1.
% J => cost

% Determine Pbatch and Gbatch
K = size(Y,1);
Pbatch =  P;
Gbatch = (1/K)*(Pbatch-Y);

% Get number of examples
nb = size(X,2);

% Determine gradients
dLdW = (1/nb)* Gbatch * X';
dLdB = (1/nb)*Gbatch*ones(size(X,2),1);

% return gradients
grad_W = dLdW + 2*lambda*W;
grad_b = dLdB;

end

