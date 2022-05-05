function [correct,incorrect] = ComputeAccuracyHisogramMBCE(X, y, W, b)

    % Get probabilities
    P = EvaluateClassifierSigmoid(X, W, b);
    [max_values,~] = max (P,[],1);
    output = double((P'==max_values')');
    correct_matrix = output.*y;
    
    % Get number of classes
    C = size(P,1);
    correct = zeros(1,C);
    incorrect = zeros(1,C);

    for i = 1:C
        
        correct_in_class = sum(correct_matrix(i,:));
        true_correct_in_class = sum(y(i,:));
        correct(i) = correct_in_class/true_correct_in_class*100;
        incorrect(i) = 100 - correct(i);

%         correct(i) = correct_in_class/true_correct_in_class*100;
%         incorrect(i) = 100-correct(i);
    end

end