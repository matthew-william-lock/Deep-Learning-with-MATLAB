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
P = EvaluateClassifierSigmoid(X_train(:, 1), W,b);

%% Compute Cost
J = ComputeCost(X_train(:, 1:100),Y_train(:, 1:100), W,b,0.1);
J = ComputeCostMBCE(X_train(:, 1:100),Y_train(:, 1:100), W,b,0.1);

%% Compute accuracy
acc = ComputeAccuracy(X_train(:, 1:100), Y_train(:, 1:100), W, b);

%% Calculate gradients
% X = X_train(:, 1:20);
% Y = Y_train(:, 1:20);
% P = EvaluateClassifier(X, W,b);
% 
% [~, ~] = ComputeGradients(X, Y, P, W, lambda);

lambda = 1;
X = X_train(:, 1);
Y = Y_train(:, 1);
P = EvaluateClassifier(X, W,b);

[~, ~] = ComputeGradients(X, Y, P, W, lambda);

%% Calculate gradients accuracy

% Data and parameter setup
X = X_train(:, 1:101);
Y = Y_train(:, 1:101);
P = EvaluateClassifier(X, W,b);
lambda = 1;

% Calculate analytical gradients
[grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda);

% Calculate numerical gradients
[b_num, W_num] = ComputeGradsNum(X, Y, W, b, lambda, 1e-6);
[b_num_s, W_num_s] = ComputeGradsNumSlow(X, Y, W, b, lambda, 1e-6);

% abs(grad_W-W_num)/max(10^-6,abs(grad_W+W_num));
% 
% % Computer difference between analytical and numerical
% dif_W = grad_W - W_num;
% dif_W_s = grad_W - W_num;
% dif_b = grad_b - b_num;
% dif_b_s = grad_b - b_num_s;
% 
% % Determine max difference
% max_diff_W = max(max((dif_W),[],'all'),max((dif_W_s),[],'all'));
% max_diff_b = max(max((dif_b),[],'all'),max((dif_b_s),[],'all'));

val= abs(grad_W-W_num)/max(10^-6,norm(grad_W)+norm(W_num));
val_max = max(max(val));

val_b= abs(grad_b-b_num)/max(10^-6,norm(grad_b)+norm(b_num));
val_b_max = max(max(val_b));

%% Calculate gradients with multiple cross ...

% Data and parameter setup
X = X_train(:, 1:20);
Y = Y_train(:, 1:20);
lambda = 1;

P_single = EvaluateClassifier(X, W,b);
P_multiple = EvaluateClassifierSigmoid(X, W,b);

% Calculate analytical gradients
[grad_W_single, grad_b_single] = ComputeGradients(X, Y, P_single, W, lambda);
[grad_W_mult, grad_b_mult] = ComputeGradientsMultiple(X, Y, P_multiple, W, lambda);

% Calculate numerical gradients
[b_num, W_num] = ComputeGradsNum(X, Y, W, b, lambda, 1e-6);
[b_num_mult, W_num_mult] = ComputeGradsNumMult(X, Y, W, b, lambda, 1e-6);

val= abs(grad_W_single-W_num)/max(10^-6,norm(grad_W_single)+norm(W_num));
val_max_single = max(max(val));

val= abs(grad_W_mult-W_num_mult)/max(10^-6,norm(grad_W_mult)+norm(W_num_mult));
val_max_double = max(max(val));




   
