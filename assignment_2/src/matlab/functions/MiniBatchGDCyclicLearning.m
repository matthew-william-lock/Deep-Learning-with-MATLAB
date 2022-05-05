function [Wstar, bstar, nt_hist] = MiniBatchGDCyclicLearning(X, Y, GDparams, W, b, lambda, nt_hist, n_s, nmin, nmax)

    % Create minibatches and do learning
    for j=1:size(X,2)/GDparams.n_batch
        
        j_start = (j-1)*GDparams.n_batch + 1;
        j_end = j*GDparams.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        W1 = cell2mat(W(1));
        W2 = cell2mat(W(2));
        b1 = cell2mat(b(1));
        b2 = cell2mat(b(2));
        
        [~,~,P,H] = ComputeNetwork(Xbatch,W1,W2,b1,b2);
        [grad_W1, grad_b1,grad_W2, grad_b2] = ComputeGradients2Layer(Xbatch,H, Ybatch, P, W1, W2, lambda);

        t = length(nt_hist)-1;
        l = floor(t/(2*n_s));

        if (2*l*n_s) <= t && t<= ((2*l+1)*n_s)
            nt = nmin + (t-2*l*n_s)/n_s * (nmax-nmin);
        else
            nt = nmax - (t-(2*l+1)*n_s)/n_s * (nmax-nmin);
        end

        nt_hist(end+1) = nt;

        W(1) = {W1 - nt*grad_W1};
        W(2) = {W2 - nt*grad_W2};
        b(1) = {b1 - nt*grad_b1};
        b(2) = {b2 - nt*grad_b2};

    end

    Wstar = W;
    bstar = b;

end

