function acc = ComputeAccuracy(X, y, W, B)

    % Get probabilities
    [~,P,~] = ComputeNetwork(X,W,B);
    
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