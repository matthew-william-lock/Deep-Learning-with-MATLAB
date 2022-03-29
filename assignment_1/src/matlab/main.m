%% Exercise One
clear;

%% Load the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validate, Y_validate, y_tvalidate] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

% Preprocess the data
mean_X_train = mean(X_train, 2);
std_X_train = std(X_train, 0, 2);
mean_X_validate = mean(X_validate, 2);
std_X_validate = std(X_validate, 0, 2);
mean_X_test = mean(X_test, 2);
std_X_test = std(X_test, 0, 2);

%% Normalise wrt mean and std
X_train = X_train - repmat(mean_X_train, [1, size(X_train, 2)]);
X_train = X_train ./ repmat(std_X_train, [1, size(X_train, 2)]);

X_validate = X_validate - repmat(mean_X_validate, [1, size(X_validate, 2)]);
X_validate = X_validate ./ repmat(std_X_validate, [1, size(X_validate, 2)]);

X_test = X_test - repmat(mean_X_test, [1, size(X_test, 2)]);
X_test = X_test ./ repmat(std_X_test, [1, size(X_test, 2)]);

%% Init W and B
W = normrnd(0,0.01,[size(Y_train,1),size(X_test,1)]);
b = normrnd(0,0.01,[size(Y_train,1),1]);

%% Evaluate classifier
P = EvaluateClassifier(X_train(:, 1:100), W,b);

%% Compute Cost
J = ComputeCost(X_train(:, 1:100),Y_train(:, 1:100), W,b,0.1);

%% Compute accuracy
acc = ComputeAccuracy(X_train(:, 1:100), Y_train(:, 1:100), W, b);

%% Calculate gradients

X = X_train(:, 1:2);
Y = Y_train(:, 1:2);
P = EvaluateClassifier(X, W,b);
lambda = 0;

[grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda);
[grad_b_num, grad_W_num] = ComputeGradsNumSlow(X, Y, W, b, lambda, 1e-6);


   
