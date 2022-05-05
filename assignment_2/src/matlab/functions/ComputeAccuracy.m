function acc = ComputeAccuracy(X, y, W, b)

    % Get probabilities
    
    W1 = cell2mat(W(1));
    W2 = cell2mat(W(2));
    b1 = cell2mat(b(1));
    b2 = cell2mat(b(2));

    [~,~,P,~] = ComputeNetwork(X,W1,W2,b1,b2);
    
    % Get most likely answer, done by :
    % (A) get max values
    % (B) get indexes where value == max value and make the rest zero
    [max_values,~] = max (P,[],1);
    P = double((P'==max_values')');

    % Correct values
    correct = sum(sum(P.*y));
    score = correct/size(y,2);

    acc = score;

end