function [Wstar, bstar, gamma_star, beta_star, nt_hist,NetParams] = MiniBatchGDCyclicLearningPreciseBatchNorm(X, Y, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax,alpha,NetParams)

    % running average
    layers = length(W);
    mean_average = cell(layers-1,1);
    mean_variance = cell(layers-1,1);

    % Determine means and variances
    NetParams.use_precomputed = (0==1);
    [~,~,~,~,mew,v] = ComputeNetworkBatchNorm(X(:,1:100),W,B,NetParams);
    for k = 1:layers-1
        NetParams.mean_average{k} = mew{k} ;
        NetParams.mean_variance{k} = v{k} ;
    end
    NetParams.use_precomputed = (1==1);


    % Create minibatches and do learning
    for j=1:size(X,2)/GDparams.n_batch

        % fprintf('Batch %d / %d \n',j,size(X,2)/GDparams.n_batch);
        
        j_start = (j-1)*GDparams.n_batch + 1;
        j_end = j*GDparams.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        
        % Forward pass
        [S,S_hat,P,H,~,~] = ComputeNetworkBatchNorm(Xbatch,W,B,NetParams);
        [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradientsKLayerBatchNorm(Xbatch,H, Ybatch, P, W, lambda, S_hat, NetParams,NetParams.mean_variance,NetParams.mean_average,S);
        
        t = length(nt_hist)-1;
        l = floor(t/(2*n_s));

        if (2*l*n_s) <= t && t<= ((2*l+1)*n_s)
            nt = nmin + (t-2*l*n_s)/n_s * (nmax-nmin);
        else
            nt = nmax - (t-(2*l+1)*n_s)/n_s * (nmax-nmin);
        end

        nt_hist(end+1) = nt;
        
        % update gradients
        for i = 1:length(W)
            W{i} = W{i}-nt*grad_W{i};
            B{i} = B{i}-nt*grad_b{i};
            if i ~=layers && NetParams.use_bn
                NetParams.gammas{i} = NetParams.gammas{i} - nt*grad_gamma{i};
                NetParams.betas{i} = NetParams.betas{i} - nt*grad_beta{i};
            end
        end
        
    end

    Wstar = W;
    bstar = B;
    gamma_star = NetParams.gammas;
    beta_star = NetParams.betas;

end

