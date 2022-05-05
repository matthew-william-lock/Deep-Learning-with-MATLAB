function acc = ComputeAccuracyBatchNorm(X, y, W, B,NetParams)

    % Get probabilities
    [~,~,P,~,~,~] = ComputeNetworkBatchNorm(X,W,B,NetParams);
    
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