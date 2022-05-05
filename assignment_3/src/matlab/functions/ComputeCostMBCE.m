function J = ComputeCostMBCE(X, Y, W, b, lambda)
    
    % Get probabilities
    P = EvaluateClassifierSigmoid(X, W, b);
    
    % Calculate cost (same as taking loss on the diagonal?)
    % For single input, i.e. size(Y) = 10,1 => same as cost = Y'*-log(P)

    K = size(Y,1);

    cost = (1/K) .*( -(1-Y).*(log(1-P)) - Y.*log(P));
    cost = sum(sum(cost))/ size(P,2);

    % Calculate regularisation cost
    reg = lambda* norm(W,"fro")^2;

    J = cost + reg;
        

end