function J = ComputeCost(X, Y, W, b, lambda)
    
    % Get probabilities
    P = EvaluateClassifier(X, W, b);
    
    % Calculate cost (same as taking loss on the diagonal?)
    cost = sum(-log(P(Y==1))) / size(P,2);

    % Calculate regularisation cost
    reg = lambda* norm(W,"fro")^2;

    J = cost + reg;
        

end