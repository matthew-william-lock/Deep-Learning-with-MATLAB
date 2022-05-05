function [Wstar, bstar, nt_hist] = MiniBatchGDCyclicLearning(X, Y, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax)

    % Create minibatches and do learning
    for j=1:size(X,2)/GDparams.n_batch
        
        j_start = (j-1)*GDparams.n_batch + 1;
        j_end = j*GDparams.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        %{
            W1 = cell2mat(W(1));
            W2 = cell2mat(W(2));
            b1 = cell2mat(B(1));
            b2 = cell2mat(B(2));
        %}

        [S,P,H] = ComputeNetwork(Xbatch,W,B);
        [grad_W, grad_b] = ComputeGradientsKLayer(Xbatch,H, Ybatch, P, W, lambda);

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
            W(i) = {cell2mat(W(i))-nt*cell2mat(grad_W(i))};
            B(i) = {cell2mat(B(i))-nt*cell2mat(grad_b(i))};
        end
        
    end

    Wstar = W;
    bstar = B;

end

