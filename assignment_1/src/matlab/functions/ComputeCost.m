function J = ComputeCost(X, Y, W, b, lambda)
    
    % Get probabilities
    P = EvaluateClassifier(X, W, b);
    
    % Calculate cost (same as taking loss on the diagonal?)
    % For single input, i.e. size(Y) = 10,1 => same as cost = Y'*-log(P)
    cost = sum(-log(P(Y==1))) / size(P,2);
%     cost = trace(-Y'*log(P))/size(P,2)

    % Calculate regularisation cost
    reg = lambda* norm(W,"fro")^2;

    J = cost + reg;
        

end