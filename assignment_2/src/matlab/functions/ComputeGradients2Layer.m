function [grad_W1, grad_b1,grad_W2, grad_b2] = ComputeGradients2Layer(X,H, Y, P, W1, W2, lambda)

% grad W is gradient matrix of the cost J relative to W and has size K×d.
% grad b is gradient vector of the cost J relative to b and has size K×1.
% J => cost


Pbatch =  P;
HBatch = H;

% Determine Gbatch
Gbatch = -(Y-Pbatch);

% Get number of examples
nb = size(X,2);

% Determine gradients
dLdW2 = (1/nb)* Gbatch * HBatch';
dLdB2 = (1/nb)*Gbatch*ones(size(X,2),1);

% Set Gbatch
Gbatch = W2'*Gbatch;
Gbatch = Gbatch .* (HBatch > 0)*1;

% Determine gradients
dLdW1 = (1/nb)* Gbatch * X';
dLdB1 = (1/nb)*Gbatch*ones(size(X,2),1);

% return gradients
grad_W2 = dLdW2 + 2*lambda*W2;
grad_b2 = dLdB2;
grad_W1 = dLdW1 + 2*lambda*W1;
grad_b1 = dLdB1;

end

