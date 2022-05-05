function [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradientsKLayerBatchNorm(X,H, Y, P, W, lambda, S_hat, NetParams,V,MEW,S_batch)

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
    grad_gamma = cell(layers-1,1);
    grad_beta = cell(layers-1,1);

    % Allocate H
    Htemp = cell(layers,1);
    Htemp(1) = {X};
    Htemp(2:end) = H;
    H = Htemp;
    
    % Preallocate arrays for ones
    ones_array = ones(size(X,2),1);
    ones_array_t = ones(size(X,2),1)';
    
    % iterate though layers and Determine gradients
    for i = layers:-1:1        

        % account for batch norm for all except last layer
        if i ~= layers && NetParams.use_bn

            s_hat = S_hat{i};

            gamma = NetParams.gammas{i};
            % beta = NetParams.betas{i};

            % Compute gradient for the scale and offset parameters for layer l
            dJdgamma = (1/nb)*(Gbatch.*s_hat)*ones_array;
            dJdbeta = (1/nb)*Gbatch*ones_array;
            grad_gamma{i} = dJdgamma;
            grad_beta{i} = dJdbeta;
            
            % Propagate the gradients through the scale and shift
            Gbatch = Gbatch.*(gamma*ones_array_t);

            % Propagate Gbatch through the batch normalization
            v = V{i};
            mew = MEW{i};
            s_batch = S_batch{i};
            sigma_1 = (v + 10^-8).^(-0.5);
            sigma_2 = (v + 10^-8).^(-1.5);
            G1 = Gbatch.*(sigma_1*ones_array_t);
            G2 = Gbatch.*(sigma_2*ones_array_t);
            D = s_batch - mew*ones_array_t;
            c = (G2.*D)*ones_array;
            Gbatch = G1 - (1/nb)* (G1*ones_array)*ones_array_t - (1/nb)*D.*(c*ones_array_t);

        end
        
        % Determine gradients
        h = H{i};
        dLdW = (1/nb)* Gbatch * h';
        dLdB = (1/nb)*Gbatch*ones_array;
    
        % Set gradients
        w = W{i};
        grad_W(i) = {dLdW + 2*lambda*w};
        grad_b(i) = {dLdB};
    
        % Set Gbatch (propogate gbatch to previous layer)
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

