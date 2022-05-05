function [J,A] = ComputeCost(X, Y, W, b, lambda)
    
    % Get probabilities
    [~,P,~] = ComputeNetwork(X,W,b);
    
    % Calculate cost (same as taking loss on the diagonal?)
    % For single input, i.e. size(Y) = 10,1 => same as cost = Y'*-log(P)
    cost = sum(-log(P(Y==1))) / size(P,2);

    % Calculate regularisation cost
    l2norm = 0;
    for i =1 : length(W)
        l2norm = l2norm + norm(cell2mat(W(i)),"fro")^2;
    end


    reg = lambda* l2norm;

    J = cost + reg;
    A=J;
        

end