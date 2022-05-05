function [X, Y, y] = LoadBatch(filename)
    
    % Load batch
    A = load(filename);

    % Get data
    X = double(A.data')/255;

    % One-hot representation for labels
    Y = zeros(10,size(X,2));
    for i = 1:size(X,2)
        Y(A.labels(i)+1,i)=1;
    end

    y = A.labels+1;



end

