function [S,S_hat,P,X,mew,v] = ComputeNetworkBatchNorm(X,W,B,NetParams)
    
    % Get batch size
    n = size(X,2);

    % Get number of layers
    layers = size(W,1);
    
    % Define S and H
    S = cell(layers-1,1);
    S_hat = cell(layers-1,1);
    H = cell(layers,1);
    mew = cell(layers-1,1);
    v = cell(layers-1,1);
    H{1} = X;

    % try preallocate s?
    
    
    for i = 1:layers-1
        
        w = (W{i});
        b = (B{i});
        h = (H{i});
       
        s = w*h+b;
        S(i)={s};
        

        % Apply batch norm
        if NetParams.use_bn   

            % Get gamma and bias
            %             gamma = NetParams.gammas{i};
            %             beta = NetParams.betas{i};
            
            if NetParams.use_precomputed
                % Extract means and var
                mean_s = NetParams.mean_average{i};
                var_s = NetParams.mean_variance{i};      
            else
                % Compute mean and variance
                mean_s = mean(s,2);
                var_s = var(s,0,2) * (n-1) / n;
                mew(i) = {mean_s};
                v(i) = {var_s};
            end
            
            s_hat = (diag(var_s+10^-8))^(-1/2)*(s - mean_s);
            S_hat(i)={s_hat};
            s = NetParams.gammas{i}.*s_hat+NetParams.betas{i};
        end

    H(i+1)={max(s,0)};

    end
    
    % Determine final layer
    %     w = ;
    %     b = ;
    %     h = ;

    s = W{layers}*H{layers}+B{layers};
    P = exp(s)./sum(exp(s));
    
    % Disregard initial layer (input)
    H = H(2:end);

    % Store relevant variables
    X = H;
    

end

% OLD FUNCITON
% s1 = W1*X + b1;
% H = max(s1,0);
% s = W2*H + b2;
% P = exp(s)./sum(exp(s));

