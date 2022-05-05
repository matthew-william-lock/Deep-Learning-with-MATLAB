function [grad_W, grad_b] = ComputeGradientsKLayer(X,H, Y, P, W, lambda)

    % Determine number of layers
    layers = length(W);
    
    % Get number of examples
    nb = size(X,2);
    
    % Determine Gbatch
    Pbatch =  P;
    Gbatch = -(Y-Pbatch);
    
    % Allocate gradients
    grad_W = cell(layers,1);
    grad_b = cell(layers,1);

    % Allocate H
    Htemp = cell(layers,1);
    Htemp(1) = {X};
    Htemp(2:end) = H;
    H = Htemp;
    
    % iterate though layers and Determine gradients
    for i = layers:-1:1
        
        % Determine gradients
        h = cell2mat(H(i));
        dLdW = (1/nb)* Gbatch * h';
        dLdB = (1/nb)*Gbatch*ones(size(X,2),1);
    
        % Set gradients
        w = cell2mat(W(i));
        grad_W(i) = {dLdW + 2*lambda*w};
        grad_b(i) = {dLdB};
    
        % Set Gbatch    
        Gbatch = w'*Gbatch;
        Gbatch = Gbatch .* (h > 0)*1;
    
    end

end

% OLD CODE

% Set Gbatch
% Gbatch = W2'*Gbatch;
% Gbatch = Gbatch .* (HBatch > 0)*1;
% 
% % Determine gradients
% dLdW1 = (1/nb)* Gbatch * X';
% dLdB1 = (1/nb)*Gbatch*ones(size(X,2),1);
% 
% % return gradients
% grad_W2 = dLdW2 + 2*lambda*W2;
% grad_b2 = dLdB2;
% grad_W1 = dLdW1 + 2*lambda*W1;
% grad_b1 = dLdB1;

