function [S,P,H] = ComputeNetwork(X,W,B)

    % Get number of layers
    layers = size(W,1);
    
    % Define S and H
    S = cell(layers,1);
    H = cell(layers,1);
    H(1) = {X};
    
    for i = 1:layers
    
        w = cell2mat(W(i,:));
        b = cell2mat(B(i,:));
        h = cell2mat(H(i,:));
        
        s = w*h+b;
        S(i)={s};
        if i ~=layers
            H(i+1)={max(s,0)};
        end
    end
    
    % Determine final layer
    s = cell2mat(S(end));
    P = exp(s)./sum(exp(s));
    
    % Disregard initial layer (input)
    H = H(2:end);

end

% OLD FUNCITON
% s1 = W1*X + b1;
% H = max(s1,0);
% s = W2*H + b2;
% P = exp(s)./sum(exp(s));

