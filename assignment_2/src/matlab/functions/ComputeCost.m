function [J,A] = ComputeCost(X, Y, W, b, lambda)
    
    % Get probabilities
    [~,~,P,~] = ComputeNetwork(X,cell2mat(W(1)),cell2mat(W(2)),cell2mat(b(1)),cell2mat(b(2)));
    
    % Calculate cost (same as taking loss on the diagonal?)
    % For single input, i.e. size(Y) = 10,1 => same as cost = Y'*-log(P)
    cost = sum(-log(P(Y==1))) / size(P,2);

    % Calculate regularisation cost
    reg = lambda* (norm(cell2mat(W(1)),"fro")^2 + norm(cell2mat(W(2)),"fro")^2);

    J = cost + reg;
    A=J;
        

end