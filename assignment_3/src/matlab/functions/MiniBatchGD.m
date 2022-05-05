function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)

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

        W(1) = {W1 - GDparams.eta*grad_W1};
        W(2) = {W2 - GDparams.eta*grad_W2};
        b(1) = {b1 - GDparams.eta*grad_b1};
        b(2) = {b2 - GDparams.eta*grad_b2};

    end

    Wstar = W;
    bstar = b;

end

