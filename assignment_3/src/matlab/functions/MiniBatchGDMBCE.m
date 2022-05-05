function [Wstar, bstar] = MiniBatchGDMBCE(X, Y, GDparams, W, b, lambda)

    % Create minibatches and do learning
    for j=1:size(X,2)/GDparams.n_batch
        
        j_start = (j-1)*GDparams.n_batch + 1;
        j_end = j*GDparams.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        P = EvaluateClassifierSigmoid(Xbatch, W,b);
        [grad_W, grad_b] = ComputeGradientsMultiple(Xbatch, Ybatch, P, W, lambda);
        W = W - GDparams.eta*grad_W;
        b = b - GDparams.eta*grad_b;

    end

    Wstar = W;
    bstar = b;

end

