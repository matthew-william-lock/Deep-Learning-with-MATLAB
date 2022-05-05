%% Load the data
clear;
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validate, Y_validate, y_validate] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
label_names = LoadLabelNames('batches.meta.mat');

%% Preprocess the data
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

%% Inital setup

% Setup paramater list
lambda_list = [0.01];
n_batch_list = [100];
eta_list = [0.001];
n_batch = 100;

% Setup cyclic learning
nmin = 1e-5;
nmax = 1e-1;
t = 0;
n_s = 500;
nt_hist = [nmin];
cycles = 1;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
loss_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_test = zeros(1,length(eta_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(eta_list));

% Loop through experiments
for experiment_no=1:length(eta_list)

    % Init seed
    rng(400);

    % Init W and B
    intermediate_layers = [50];
    d = size(X_train,1);
    [W,B] = InitParameters(X_train,Y_train,d,intermediate_layers);

    % Set paramts
    n_batch = n_batch_list(experiment_no);
    n_epochs = n_epochs_list(experiment_no);
    lambda=lambda_list(experiment_no);
    eta = eta_list(experiment_no);

    k = 2;
    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;

    % Compute loss before training
    loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,B,lambda);
    loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
    accuracy_train(experiment_no,1) = ComputeAccuracy(X_train, Y_train, W, B);
    accuracy_validate(experiment_no,1) = ComputeAccuracy(X_validate, Y_validate, W, B);
    cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, B,0);
    cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, B,0);

    % Learning steps
    for i=1:epochs
        
        % Minibatch GD
        [W,b,nt_hist] = MiniBatchGDCyclicLearning(X_train, Y_train, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,B,lambda);
        loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
        accuracy_train(experiment_no,i+1) = ComputeAccuracy(X_train, Y_train, W, B);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracy(X_validate, Y_validate, W, B);
        cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, B,0);
        cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, B,0);
    
        fprintf("Experiment %d - cycles %d completed ( %.2f) \n",experiment_no,i,length(nt_hist)/(cycles*2*n_s)*100);
    end

    %     accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

%% Gradient check

% reduce the number of inputs
batch_size = 1:100;
X = X_train(1:10, batch_size);
Y = Y_train(:, batch_size);

intermediate_layers = [50,25];
d = size(X,1);
lambda = 0.5;

% Init W and B
rng(400);
[W,B] = InitParameters(X,Y_train,d,intermediate_layers);

% Calculate numerical gradients
[grad_b_num, grad_W_num] = ComputeGradsNum(X, Y, W, B, lambda, 1e-5);
[b_num_s, W_num_s] = ComputeGradsNumSlow(X, Y, W, B, lambda, 1e-5);

% Calculate analytical gradients
[S,P,H] = ComputeNetwork(X,W,B);
[grad_W, grad_b] = ComputeGradientsKLayer(X,H, Y, P, W, lambda);

% Get number of layers
layers = length(grad_W);

max_rel_diff_W = -inf;
max_rel_diff_b = -inf;

% Iterate through layers and check max relatice difference
for i = 1:layers
    
    grad_W1 = cell2mat(grad_W(i));
    grad_b1 = cell2mat(grad_b(i));

    grad_W_num_1 = cell2mat(grad_W_num(i));
    grad_b_num_1 = cell2mat(grad_b_num(i));
    grad_W_num_1_slow = cell2mat(W_num_s(i));
    grad_b_num_1_slow = cell2mat(b_num_s(i));

    rel_dif_W = abs(grad_W1-grad_W_num_1)/max(10^-6,norm(grad_W1)+norm(grad_W_num_1));
    rel_dif_W_s = abs(grad_W1-grad_W_num_1_slow)/max(10^-6,norm(grad_W1)+norm(grad_W_num_1_slow));
    rel_dif_b = abs(grad_b1-grad_b_num_1)/max(10^-6,norm(grad_b1)+norm(grad_b_num_1));
    rel_dif_b_s = abs(grad_b1-grad_b_num_1_slow)/max(10^-6,norm(grad_b1)+norm(grad_b_num_1_slow));

    % Determine max difference
    max_rel_diff_W1 = max(max((rel_dif_W),[],'all'),max((rel_dif_W_s),[],'all'));
    max_rel_diff_b1 = max(max((rel_dif_b),[],'all'),max((rel_dif_b_s),[],'all'));
    
    max_rel_diff_W = max(max_rel_diff_W1,max_rel_diff_W);
    max_rel_diff_b = max(max_rel_diff_b1,max_rel_diff_b);

end

disp(max_rel_diff_W);
disp(max_rel_diff_b);

%% Replicating assignment two default settings

% Setup paramater list
lambda_list = [0.01];
n_batch_list = [100];
eta_list = [0.001];
n_batch = 100;

% Setup cyclic learning
nmin = 1e-5;
nmax = 1e-1;
t = 0;
n_s = 500;
nt_hist = [nmin];
cycles = 1;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
loss_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_test = zeros(1,length(eta_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(eta_list));

% Loop through experiments
for experiment_no=1:length(eta_list)

    % Init seed
    rng(400);

    % Init W and B
    intermediate_layers = [50];
    d = size(X_train,1);
    [W,B] = InitParameters(X_train,Y_train,d,intermediate_layers);

    % Set paramts
    n_batch = n_batch_list(experiment_no);
    n_epochs = n_epochs_list(experiment_no);
    lambda=lambda_list(experiment_no);
    eta = eta_list(experiment_no);

    k = 2;
    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;

    % Compute loss before training
    loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,B,lambda);
    loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
    accuracy_train(experiment_no,1) = ComputeAccuracy(X_train, Y_train, W, B);
    accuracy_validate(experiment_no,1) = ComputeAccuracy(X_validate, Y_validate, W, B);
    cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, B,0);
    cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, B,0);

    % Learning steps
    for i=1:epochs
        
        % Minibatch GD
        [W,b,nt_hist] = MiniBatchGDCyclicLearning(X_train, Y_train, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,B,lambda);
        loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
        accuracy_train(experiment_no,i+1) = ComputeAccuracy(X_train, Y_train, W, B);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracy(X_validate, Y_validate, W, B);
        cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, B,0);
        cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, B,0);
    
        fprintf("Experiment %d - cycles %d completed ( %.2f) \n",experiment_no,i,length(nt_hist)/(cycles*2*n_s)*100);
    end

    %     accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_loss_learning_plots

%% Ex 2 - Two layer

clear;

% Load all data
[X1, Y1, ~] = LoadBatch('data_batch_1.mat');
[X2, Y2, ~] = LoadBatch('data_batch_2.mat');
[X3, Y3, ~] = LoadBatch('data_batch_3.mat');
[X4, Y4, ~] = LoadBatch('data_batch_4.mat');
[X5, Y5, ~] = LoadBatch('data_batch_5.mat');

% Concat data
X = [X1 , X2, X3, X4, X5];
Y = [Y1 , Y2, Y3, Y4, Y5];

% Clear variables for memory
clear X1 X2 X3 X4 X5 Y1 Y2 Y3 Y4 Y5 ;

% Create training and validation data
n_validation = 5000;
X_train = X(:,1:end-n_validation); Y_train = Y(:,1:end-n_validation);
X_validate = X(:,end-n_validation+1:end); Y_validate = Y(:,end-n_validation+1:end);
n = size(X_train,2);

% Clear variables for memory
clear X Y ;

% Load test
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
fprintf("All data loaded \n");

% Preprocess the data
mean_X_train = mean(X_train, 2);
std_X_train = std(X_train, 0, 2);
mean_X_validate = mean(X_validate, 2);
std_X_validate = std(X_validate, 0, 2);
mean_X_test = mean(X_test, 2);
std_X_test = std(X_test, 0, 2);
fprintf("Data processed \n");

% Normalise wrt mean and std
X_train = X_train - repmat(mean_X_train, [1, size(X_train, 2)]);
X_train = X_train ./ repmat(std_X_train, [1, size(X_train, 2)]);
X_validate = X_validate - repmat(mean_X_validate, [1, size(X_validate, 2)]);
X_validate = X_validate ./ repmat(std_X_validate, [1, size(X_validate, 2)]);
X_test = X_test - repmat(mean_X_test, [1, size(X_test, 2)]);
X_test = X_test ./ repmat(std_X_test, [1, size(X_test, 2)]);
fprintf("Data normalised \n");

% Generate results
fprintf("Initalising network \n");

% Setup paramater list
lambda_list = [0.005];
n_batch_list = [100];
eta_list = [0.001];
n_batch = 100;

% Setup cyclic learning
nmin = 1e-5;
nmax = 1e-1;
t = 0;
% n_s = 5*45000/n_batch;
n_s = 2*floor(n/n_batch);
nt_hist = [nmin];
cycles = 2;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
loss_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_test = zeros(1,length(eta_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(eta_list));

% Loop through experiments
for experiment_no=1:length(eta_list)

    % Init seed
    rng(400);

    % Init W and B
    intermediate_layers = [50,50];
    d = size(X_train,1);
    [W,B] = InitParameters(X_train,Y_train,d,intermediate_layers);

    % Set paramts
    n_batch = n_batch_list(experiment_no);
    n_epochs = n_epochs_list(experiment_no);
    lambda=lambda_list(experiment_no);
    eta = eta_list(experiment_no);

    k = 2;
    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;

    % Compute loss before training
    loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,B,lambda);
    loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
    accuracy_train(experiment_no,1) = ComputeAccuracy(X_train, Y_train, W, B);
    accuracy_validate(experiment_no,1) = ComputeAccuracy(X_validate, Y_validate, W, B);
    cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, B,0);
    cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, B,0);

    % Learning steps
    for i=1:epochs
        
        % Minibatch GD
        [W,b,nt_hist] = MiniBatchGDCyclicLearning(X_train, Y_train, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,B,lambda);
        loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
        accuracy_train(experiment_no,i+1) = ComputeAccuracy(X_train, Y_train, W, B);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracy(X_validate, Y_validate, W, B);
        cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, B,0);
        cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, B,0);
    
        fprintf("Experiment %d - epochs %d completed ( %.2f) \n",experiment_no,i,length(nt_hist)/(cycles*2*n_s)*100);

        % Shuffle training data
        shuffle = randperm(size(X_train, 2));
        X_train = X_train(:, shuffle);
        Y_train = Y_train(:,shuffle);
    end

    %     accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_loss_learning_plots

%% Ex 2 - 9 Layer network

% Setup paramater list
lambda_list = [0.005];
n_batch_list = [100];
eta_list = [0.001];
n_batch = 100;

% Setup cyclic learning
n = size(X_train,2);
nmin = 1e-5;
nmax = 1e-1;
t = 0;
% n_s = 5*45000/n_batch;
n_s = 2*floor(n/n_batch);
nt_hist = [nmin];
cycles = 2;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
loss_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_test = zeros(1,length(eta_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(eta_list));

% Loop through experiments
for experiment_no=1:length(eta_list)

    % Init seed
    rng(400);

    % Init W and B
    intermediate_layers = [50,30,20,20,10,10,10,10];
    d = size(X_train,1);
    [W,B] = InitParameters(X_train,Y_train,d,intermediate_layers);

    % Set paramts
    n_batch = n_batch_list(experiment_no);
    n_epochs = n_epochs_list(experiment_no);
    lambda=lambda_list(experiment_no);
    eta = eta_list(experiment_no);

    k = 2;
    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;

    % Compute loss before training
    loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,B,lambda);
    loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
    accuracy_train(experiment_no,1) = ComputeAccuracy(X_train, Y_train, W, B);
    accuracy_validate(experiment_no,1) = ComputeAccuracy(X_validate, Y_validate, W, B);
    cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, B,0);
    cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, B,0);

    % Learning steps
    for i=1:epochs
        
        % Minibatch GD
        [W,b,nt_hist] = MiniBatchGDCyclicLearning(X_train, Y_train, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,B,lambda);
        loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
        accuracy_train(experiment_no,i+1) = ComputeAccuracy(X_train, Y_train, W, B);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracy(X_validate, Y_validate, W, B);
        cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, B,0);
        cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, B,0);
    
        fprintf("Experiment %d - epochs %d completed ( %.2f) \n",experiment_no,i,length(nt_hist)/(cycles*2*n_s)*100);

        % Shuffle training data
        shuffle = randperm(size(X_train, 2));
        X_train = X_train(:, shuffle);
        Y_train = Y_train(:,shuffle);
    end

    %     accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_loss_learning_plots

%% Gradient check batch norm

% reduce the number of inputs
clc;
batch_size = 1:101;
X = X_train(1:10, batch_size);
Y = Y_train(:, batch_size);

intermediate_layers = [50,50];
d = size(X,1);
lambda = 0.005;

% Init W and B
rng(400);
[W,B,gammas,betas] = InitParametersBatchNorm(X,Y_train,d,intermediate_layers);
NetParams.W = W;
NetParams.b = B;
NetParams.use_bn = (1==1);
NetParams.gammas = gammas;
NetParams.betas = betas;
NetParams.use_precomputed = (1==0);

% Calculate numerical gradients
Grads = ComputeGradsNumSlowBatchNorm(X, Y, NetParams, lambda, 1e-5);

% Calculate analytical gradients
[S,S_hat,P,H,MEW,V]= ComputeNetworkBatchNorm(X,W,B,NetParams);
[grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradientsKLayerBatchNorm(X,H, Y, P, W, lambda, S_hat, NetParams,V,MEW,S);

% Get number of layers
layers = length(grad_W);

max_rel_diff_W = -inf;
max_rel_diff_b = -inf;
max_rel_diff_gamma = -inf;
max_rel_diff_beta = -inf;


% Iterate through layers and check max relatice difference
for i = 1:layers
    
    grad_W1 = cell2mat(grad_W(i));
    grad_b1 = cell2mat(grad_b(i));

    grad_W_num_1 = cell2mat(Grads.W(i));
    grad_b_num_1 = cell2mat(Grads.b(i));

    rel_dif_W = abs(grad_W1-grad_W_num_1)/max(10^-6,norm(grad_W1)+norm(grad_W_num_1));
    rel_dif_b = abs(grad_b1-grad_b_num_1)/max(10^-6,norm(grad_b1)+norm(grad_b_num_1));

    % Determine max difference
    max_rel_diff_W1 = max((rel_dif_W),[],'all');
    max_rel_diff_b1 = max((rel_dif_b),[],'all');

    max_rel_diff_W = max(max_rel_diff_W1,max_rel_diff_W);
    max_rel_diff_b = max(max_rel_diff_b1,max_rel_diff_b);

    % Batch norm paramters
    if i ~=layers

        grad_gamma1 = cell2mat(grad_gamma(i));
        grad_beta1 = cell2mat(grad_beta(i));
        
        grad_gamma_num_1 = cell2mat(Grads.gammas(i));
        grad_beta_num_1 = cell2mat(Grads.betas(i));

        rel_dif_gamma = abs(grad_gamma1-grad_gamma_num_1)/max(10^-6,norm(grad_gamma1)+norm(grad_gamma_num_1));
        rel_dif_beta = abs(grad_beta1-grad_beta_num_1)/max(10^-6,norm(grad_beta1)+norm(grad_beta_num_1));

        max_rel_diff_gamma1 = max((rel_dif_gamma),[],'all');
        max_rel_diff_beta1 = max((rel_dif_beta),[],'all');

        max_rel_diff_gamma = max(max_rel_diff_gamma1,max_rel_diff_gamma);
        max_rel_diff_beta = max(max_rel_diff_beta1,max_rel_diff_beta);

    end

end

disp(max_rel_diff_W);
disp(max_rel_diff_b);
disp(max_rel_diff_gamma);
disp(max_rel_diff_beta);

%% Ex 3 - Batch norm (reduced dataset)

% Load the data
clear;
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validate, Y_validate, y_tvalidate] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
label_names = LoadLabelNames('batches.meta.mat');

% Preprocess the data
mean_X_train = mean(X_train, 2);
std_X_train = std(X_train, 0, 2);
mean_X_validate = mean(X_validate, 2);
std_X_validate = std(X_validate, 0, 2);
mean_X_test = mean(X_test, 2);
std_X_test = std(X_test, 0, 2);

% Normalise wrt mean and std
X_train = X_train - repmat(mean_X_train, [1, size(X_train, 2)]);
X_train = X_train ./ repmat(std_X_train, [1, size(X_train, 2)]);

X_validate = X_validate - repmat(mean_X_validate, [1, size(X_validate, 2)]);
X_validate = X_validate ./ repmat(std_X_validate, [1, size(X_validate, 2)]);

X_test = X_test - repmat(mean_X_test, [1, size(X_test, 2)]);
X_test = X_test ./ repmat(std_X_test, [1, size(X_test, 2)]);

% Setup paramater list
lambda_list = [0.005];
n_batch_list = [100];
eta_list = [0.001];
n_batch = 100;
alpha = 0.8;

% Setup cyclic learning
n = size(X_train,2);
nmin = 1e-5;
nmax = 1e-1;
t = 0;
% n_s = 5*45000/n_batch;
n_s = 2*floor(n/n_batch);
nt_hist = [nmin];
cycles = 2;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
loss_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_test = zeros(1,length(eta_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(eta_list));

% Loop through experiments
for experiment_no=1:length(eta_list)

    % Init seed
    rng(400);

    % Init W and B
    intermediate_layers = [50,50];
    d = size(X_train,1);
    [W,B,gammas,betas] = InitParametersBatchNorm(X_train,Y_train,d,intermediate_layers);
    NetParams.W = W;
    NetParams.b = B;
    NetParams.use_bn = (1==1);
    NetParams.gammas = gammas;
    NetParams.betas = betas;
    NetParams.use_precomputed = (1==0);

    % Set paramts
    n_batch = n_batch_list(experiment_no);
    n_epochs = n_epochs_list(experiment_no);
    lambda=lambda_list(experiment_no);
    eta = eta_list(experiment_no);

    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;

    % Compute loss before training
    %     loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,B,lambda);
    %     loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
    accuracy_train(experiment_no,1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
    accuracy_validate(experiment_no,1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);
    %     cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, B,0);
    %     cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, B,0);

    % Learning steps
    for i=1:epochs
        
        % Minibatch GD
        % [S,S_hat,P,H,MEW,V]= ComputeNetworkBatchNorm(X,W,B,NetParams);
        % [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradientsKLayerBatchNorm(X,H, Y, P, W, lambda, S_hat, NetParams,V,MEW,S);
        
        [W,b,NetParams.gammas,NetParams.betas,nt_hist] = MiniBatchGDCyclicLearningBatchNorm(X_train, Y_train, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax, alpha,NetParams);
    
        % Compute loss at the end of the epoch
        %         loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,B,lambda);
        %         loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
        accuracy_train(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);
        %         cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, B,0);
        %         cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, B,0);
    
        fprintf("Experiment %d - epochs %d completed ( %.2f) \n",experiment_no,i,i/epochs*100);

        % Shuffle training data
        shuffle = randperm(size(X_train, 2));
        X_train = X_train(:, shuffle);
        Y_train = Y_train(:,shuffle);
    end

    %     accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_loss_learning_plots

%% Ex 3 - full data set , 3 layer network

clear;

% Load all data
[X1, Y1, ~] = LoadBatch('data_batch_1.mat');
[X2, Y2, ~] = LoadBatch('data_batch_2.mat');
[X3, Y3, ~] = LoadBatch('data_batch_3.mat');
[X4, Y4, ~] = LoadBatch('data_batch_4.mat');
[X5, Y5, ~] = LoadBatch('data_batch_5.mat');

% Concat data
X = [X1 , X2, X3, X4, X5];
Y = [Y1 , Y2, Y3, Y4, Y5];

% Clear variables for memory
clear X1 X2 X3 X4 X5 Y1 Y2 Y3 Y4 Y5 ;

% Create training and validation data
n_validation = 5000;
X_train = X(:,1:end-n_validation); Y_train = Y(:,1:end-n_validation);
X_validate = X(:,end-n_validation+1:end); Y_validate = Y(:,end-n_validation+1:end);
n = size(X_train,2);

% Clear variables for memory
clear X Y ;

% Load test
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
fprintf("All data loaded \n");

% Preprocess the data
mean_X_train = mean(X_train, 2);
std_X_train = std(X_train, 0, 2);
mean_X_validate = mean(X_validate, 2);
std_X_validate = std(X_validate, 0, 2);
mean_X_test = mean(X_test, 2);
std_X_test = std(X_test, 0, 2);
fprintf("Data processed \n");

% Normalise wrt mean and std
X_train = X_train - repmat(mean_X_train, [1, size(X_train, 2)]);
X_train = X_train ./ repmat(std_X_train, [1, size(X_train, 2)]);
X_validate = X_validate - repmat(mean_X_validate, [1, size(X_validate, 2)]);
X_validate = X_validate ./ repmat(std_X_validate, [1, size(X_validate, 2)]);
X_test = X_test - repmat(mean_X_test, [1, size(X_test, 2)]);
X_test = X_test ./ repmat(std_X_test, [1, size(X_test, 2)]);
fprintf("Data normalised \n");

% Generate results
fprintf("Initalising network \n");

% Setup paramater list
lambda_list = [0.005];
n_batch_list = [100];
eta_list = [0.001];
n_batch = 100;
alpha = 0.8;

% Setup cyclic learning
n = size(X_train,2);
nmin = 1e-5;
nmax = 1e-1;
t = 0;
n_s = 5*45000/n_batch;
% n_s = 2*floor(n/n_batch);
nt_hist = [nmin];
cycles = 2;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
loss_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_train = zeros(length(eta_list),1,n_epochs_list(1)+1);
cost_validate = zeros(length(eta_list),1,n_epochs_list(1)+1);
accuracy_test = zeros(1,length(eta_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(eta_list));

% Loop through experiments
for experiment_no=1:length(eta_list)

    % Init seed
    rng(400);

    % Init W and B
    intermediate_layers = [50,50];
    d = size(X_train,1);
    [W,B,gammas,betas] = InitParametersBatchNorm(X_train,Y_train,d,intermediate_layers);
    NetParams.W = W;
    NetParams.b = B;
    NetParams.use_bn = (1==1);
    NetParams.gammas = gammas;
    NetParams.betas = betas;
    NetParams.use_precomputed = (1==0);

    % Set paramts
    n_batch = n_batch_list(experiment_no);
    n_epochs = n_epochs_list(experiment_no);
    lambda=lambda_list(experiment_no);
    eta = eta_list(experiment_no);

    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;

    % Compute loss before training
    %     loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,B,lambda);
    %     loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
    %     cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, B,0);
    %     cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, B,0);

    %     accuracy_train(experiment_no,1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
    %     accuracy_validate(experiment_no,1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);

    % Learning steps
    tic
    for i=1:epochs
        
        % Minibatch GD
        % [S,S_hat,P,H,MEW,V]= ComputeNetworkBatchNorm(X,W,B,NetParams);
        % [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradientsKLayerBatchNorm(X,H, Y, P, W, lambda, S_hat, NetParams,V,MEW,S);
        
        [W,b,NetParams.gammas,NetParams.betas,nt_hist,NetParams] = MiniBatchGDCyclicLearningBatchNorm(X_train, Y_train, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax, alpha,NetParams);
        time = toc;

        % Compute loss at the end of the epoch
        %         loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,B,lambda);
        %         loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
        %         cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, B,0);
        %         cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, B,0);

        accuracy_train(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);
        
        percent = i/epochs*100;
        time_per_pecent = time/percent;
        time_left = (100-percent)*time_per_pecent;
        fprintf("Experiment %d - epochs %d completed ( %.2f)  - time left : %.2f \n",experiment_no,i,percent,time_left);

        % Shuffle training data
        shuffle = randperm(size(X_train, 2));
        X_train = X_train(:, shuffle);
        Y_train = Y_train(:,shuffle);
    end
            
    NetParams.use_precomputed = (1==1);
    accuracy_test(experiment_no) = ComputeAccuracyBatchNorm(X_test, Y_test, W, b,NetParams)*100;
    NetParams.use_precomputed = (1==0);
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_loss_learning_plots

%% Ex 3 - course search and fine, find good value for lambda (3 layer)

clear;

% Load all data
[X1, Y1, ~] = LoadBatch('data_batch_1.mat');
[X2, Y2, ~] = LoadBatch('data_batch_2.mat');
[X3, Y3, ~] = LoadBatch('data_batch_3.mat');
[X4, Y4, ~] = LoadBatch('data_batch_4.mat');
[X5, Y5, ~] = LoadBatch('data_batch_5.mat');

% Concat data
X = [X1 , X2, X3, X4, X5];
Y = [Y1 , Y2, Y3, Y4, Y5];

% Clear variables for memory
clear X1 X2 X3 X4 X5 Y1 Y2 Y3 Y4 Y5 ;

% Create training and validation data
n_validation = 5000;
X_train = X(:,1:end-n_validation); Y_train = Y(:,1:end-n_validation);
X_validate = X(:,end-n_validation+1:end); Y_validate = Y(:,end-n_validation+1:end);
n = size(X_train,2);

% Clear variables for memory
clear X Y ;

% Load test
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
fprintf("All data loaded \n");

% Preprocess the data
mean_X_train = mean(X_train, 2);
std_X_train = std(X_train, 0, 2);
mean_X_validate = mean(X_validate, 2);
std_X_validate = std(X_validate, 0, 2);
mean_X_test = mean(X_test, 2);
std_X_test = std(X_test, 0, 2);
fprintf("Data processed \n");

% Normalise wrt mean and std
X_train = X_train - repmat(mean_X_train, [1, size(X_train, 2)]);
X_train = X_train ./ repmat(std_X_train, [1, size(X_train, 2)]);
X_validate = X_validate - repmat(mean_X_validate, [1, size(X_validate, 2)]);
X_validate = X_validate ./ repmat(std_X_validate, [1, size(X_validate, 2)]);
X_test = X_test - repmat(mean_X_test, [1, size(X_test, 2)]);
X_test = X_test ./ repmat(std_X_test, [1, size(X_test, 2)]);
fprintf("Data normalised \n");

% Generate results
fprintf("Initalising network \n");

% Lambda course search
% l_min = -5;
% l_max = -1;
l = [-2.77777777777778;-2.33333333333333;-1.88888888888889];
l_min = l(1);
l_max = l(3);
lambda_list = l_min:(l_max-l_min)/9:l_max;
    lambda_list = 10.^lambda_list;

% Setup paramater list
n_batch_list = [100];
eta_list = [0.001];
n_batch = 100;
alpha = 0.8;

% Setup cyclic learning
n = size(X_train,2);
nmin = 1e-5;
nmax = 1e-1;
t = 0;
n_s = 5*45000/n_batch;
% n_s = 2*floor(n/n_batch);
nt_hist = [nmin];
cycles = 2;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
loss_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
cost_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
cost_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_test = zeros(1,length(lambda_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(lambda_list));

% Loop through experiments
for experiment_no=1:length(lambda_list)

    % Reset parameters
    nt_hist = [nmin];

    % Init seed
    rng(400);

    % Init W and B
    intermediate_layers = [50,50];
    d = size(X_train,1);
    [W,B,gammas,betas] = InitParametersBatchNorm(X_train,Y_train,d,intermediate_layers);
    NetParams.W = W;
    NetParams.b = B;
    NetParams.use_bn = (1==1);
    NetParams.gammas = gammas;
    NetParams.betas = betas;
    NetParams.use_precomputed = (1==0);

    % Set paramts
    n_batch = n_batch_list(1);
    n_epochs = n_epochs_list(1);
    lambda=lambda_list(experiment_no);

    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;

    % Compute loss before training
    %     loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,B,lambda);
    %     loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
    %     cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, B,0);
    %     cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, B,0);

    %     accuracy_train(experiment_no,1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
    %     accuracy_validate(experiment_no,1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);

    % Learning steps
    tic
    for i=1:epochs
        
        % Minibatch GD
        % [S,S_hat,P,H,MEW,V]= ComputeNetworkBatchNorm(X,W,B,NetParams);
        % [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradientsKLayerBatchNorm(X,H, Y, P, W, lambda, S_hat, NetParams,V,MEW,S);
        
        [W,b,NetParams.gammas,NetParams.betas,nt_hist,NetParams] = MiniBatchGDCyclicLearningBatchNorm(X_train, Y_train, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax, alpha,NetParams);
        time = toc;

        % Compute loss at the end of the epoch
        %         loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,B,lambda);
        %         loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
        %         cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, B,0);
        %         cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, B,0);

        accuracy_train(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);
        
        percent = i/epochs*100;
        time_per_pecent = time/percent;
        time_left = (100-percent)*time_per_pecent;
        fprintf("Experiment %d - epochs %d completed ( %.2f)  - time left : %.2f \n",experiment_no,i,percent,time_left);

        % Shuffle training data
        shuffle = randperm(size(X_train, 2));
        X_train = X_train(:, shuffle);
        Y_train = Y_train(:,shuffle);
    end
            
    NetParams.use_precomputed = (1==1);
    accuracy_test(experiment_no) = ComputeAccuracyBatchNorm(X_test, Y_test, W, b,NetParams)*100;
    NetParams.use_precomputed = (1==0);
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_loss_learning_plots

%% Ex 3 - final train for 3 layer network

clear;

% Load all data
[X1, Y1, ~] = LoadBatch('data_batch_1.mat');
[X2, Y2, ~] = LoadBatch('data_batch_2.mat');
[X3, Y3, ~] = LoadBatch('data_batch_3.mat');
[X4, Y4, ~] = LoadBatch('data_batch_4.mat');
[X5, Y5, ~] = LoadBatch('data_batch_5.mat');

% Concat data
X = [X1 , X2, X3, X4, X5];
Y = [Y1 , Y2, Y3, Y4, Y5];

% Clear variables for memory
clear X1 X2 X3 X4 X5 Y1 Y2 Y3 Y4 Y5 ;

% Create training and validation data
n_validation = 5000;
X_train = X(:,1:end-n_validation); Y_train = Y(:,1:end-n_validation);
X_validate = X(:,end-n_validation+1:end); Y_validate = Y(:,end-n_validation+1:end);
n = size(X_train,2);

% Clear variables for memory
clear X Y ;

% Load test
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
fprintf("All data loaded \n");

% Preprocess the data
mean_X_train = mean(X_train, 2);
std_X_train = std(X_train, 0, 2);
mean_X_validate = mean(X_validate, 2);
std_X_validate = std(X_validate, 0, 2);
mean_X_test = mean(X_test, 2);
std_X_test = std(X_test, 0, 2);
fprintf("Data processed \n");

% Normalise wrt mean and std
X_train = X_train - repmat(mean_X_train, [1, size(X_train, 2)]);
X_train = X_train ./ repmat(std_X_train, [1, size(X_train, 2)]);
X_validate = X_validate - repmat(mean_X_validate, [1, size(X_validate, 2)]);
X_validate = X_validate ./ repmat(std_X_validate, [1, size(X_validate, 2)]);
X_test = X_test - repmat(mean_X_test, [1, size(X_test, 2)]);
X_test = X_test ./ repmat(std_X_test, [1, size(X_test, 2)]);
fprintf("Data normalised \n");

% Generate results
fprintf("Initalising network \n");

% Lambda course search
% l_min = -5;
% l_max = -1;
% l = -2.77777777777778;
lambda_list = [0.00330003479112527];

% Setup paramater list
n_batch_list = [100];
eta_list = [0.001];
n_batch = 100;
alpha = 0.8;

% Setup cyclic learning
n = size(X_train,2);
nmin = 1e-5;
nmax = 1e-1;
t = 0;
n_s = 5*45000/n_batch;
% n_s = 2*floor(n/n_batch);
nt_hist = [nmin];
cycles = 2;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
loss_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
cost_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
cost_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_test = zeros(1,length(lambda_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(lambda_list));

% Loop through experiments
for experiment_no=1:length(lambda_list)

    % Reset parameters
    nt_hist = [nmin];

    % Init seed
    rng(400);

    % Init W and B
%     intermediate_layers = [50,50];
    intermediate_layers = [50, 30, 20, 20, 10, 10, 10, 10];
    d = size(X_train,1);
    [W,B,gammas,betas] = InitParametersBatchNorm(X_train,Y_train,d,intermediate_layers);
    NetParams.W = W;
    NetParams.b = B;
    NetParams.use_bn = (1==1);
    NetParams.gammas = gammas;
    NetParams.betas = betas;
    NetParams.use_precomputed = (1==0);

    % Set paramts
    n_batch = n_batch_list(1);
    n_epochs = n_epochs_list(1);
    lambda=lambda_list(experiment_no);

    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;

    % Compute loss before training
    %     loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,B,lambda);
    %     loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
    %     cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, B,0);
    %     cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, B,0);

    %     accuracy_train(experiment_no,1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
    %     accuracy_validate(experiment_no,1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);

    % Learning steps
    tic
    for i=1:epochs
        
        % Minibatch GD
        % [S,S_hat,P,H,MEW,V]= ComputeNetworkBatchNorm(X,W,B,NetParams);
        % [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradientsKLayerBatchNorm(X,H, Y, P, W, lambda, S_hat, NetParams,V,MEW,S);
        
        [W,b,NetParams.gammas,NetParams.betas,nt_hist,NetParams] = MiniBatchGDCyclicLearningBatchNorm(X_train, Y_train, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax, alpha,NetParams);
        time = toc;

        % Compute loss at the end of the epoch
        %         loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,B,lambda);
        %         loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
        %         cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, B,0);
        %         cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, B,0);

        accuracy_train(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);
        
        percent = i/epochs*100;
        time_per_pecent = time/percent;
        time_left = (100-percent)*time_per_pecent;
        fprintf("Experiment %d - epochs %d completed ( %.2f)  - time left : %.2f \n",experiment_no,i,percent,time_left);

        % Shuffle training data
        shuffle = randperm(size(X_train, 2));
        X_train = X_train(:, shuffle);
        Y_train = Y_train(:,shuffle);
    end
            
    NetParams.use_precomputed = (1==1);
    accuracy_test(experiment_no) = ComputeAccuracyBatchNorm(X_test, Y_test, W, b,NetParams)*100;
    NetParams.use_precomputed = (1==0);
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_loss_learning_plots

%% Pros of batch norm

clear;

% Load all data
[X1, Y1, ~] = LoadBatch('data_batch_1.mat');
[X2, Y2, ~] = LoadBatch('data_batch_2.mat');
[X3, Y3, ~] = LoadBatch('data_batch_3.mat');
[X4, Y4, ~] = LoadBatch('data_batch_4.mat');
[X5, Y5, ~] = LoadBatch('data_batch_5.mat');

% Concat data
X = [X1 , X2, X3, X4, X5];
Y = [Y1 , Y2, Y3, Y4, Y5];

% Clear variables for memory
clear X1 X2 X3 X4 X5 Y1 Y2 Y3 Y4 Y5 ;

% Create training and validation data
n_validation = 5000;
X_train = X(:,1:end-n_validation); Y_train = Y(:,1:end-n_validation);
X_validate = X(:,end-n_validation+1:end); Y_validate = Y(:,end-n_validation+1:end);
n = size(X_train,2);

% Clear variables for memory
clear X Y ;

% Load test
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
fprintf("All data loaded \n");

% Preprocess the data
mean_X_train = mean(X_train, 2);
std_X_train = std(X_train, 0, 2);
mean_X_validate = mean(X_validate, 2);
std_X_validate = std(X_validate, 0, 2);
mean_X_test = mean(X_test, 2);
std_X_test = std(X_test, 0, 2);
fprintf("Data processed \n");

% Normalise wrt mean and std
X_train = X_train - repmat(mean_X_train, [1, size(X_train, 2)]);
X_train = X_train ./ repmat(std_X_train, [1, size(X_train, 2)]);
X_validate = X_validate - repmat(mean_X_validate, [1, size(X_validate, 2)]);
X_validate = X_validate ./ repmat(std_X_validate, [1, size(X_validate, 2)]);
X_test = X_test - repmat(mean_X_test, [1, size(X_test, 2)]);
X_test = X_test ./ repmat(std_X_test, [1, size(X_test, 2)]);
fprintf("Data normalised \n");

% Generate results
fprintf("Initalising network \n");

% Lambda course search
% l_min = -5;
% l_max = -1;
% l = -2.77777777777778;
lambda_list = [0.005];

% Setup paramater list
n_batch_list = [100];
eta_list = [0.001];
n_batch = 100;
alpha = 0.8;

% Setup cyclic learning
n = size(X_train,2);
nmin = 1e-5;
nmax = 1e-1;
t = 0;
n_s = 5*45000/n_batch;
% n_s = 2*floor(n/n_batch);
nt_hist = [nmin];
cycles = 2;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
loss_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
cost_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
cost_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_test = zeros(1,length(lambda_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(lambda_list));

% Loop through experiments
for experiment_no=1:length(lambda_list)

    % Reset parameters
    nt_hist = [nmin];

    % Init seed
    rng(400);

    % Init W and B
%     intermediate_layers = [50,50];
    intermediate_layers = [50, 30, 20, 20, 10, 10, 10, 10];
    d = size(X_train,1);
    [W,B,gammas,betas] = InitParametersBatchNorm(X_train,Y_train,d,intermediate_layers);
    NetParams.W = W;
    NetParams.b = B;
    NetParams.use_bn = (1==1);
    NetParams.gammas = gammas;
    NetParams.betas = betas;
    NetParams.use_precomputed = (1==0);

    % Set paramts
    n_batch = n_batch_list(1);
    n_epochs = n_epochs_list(1);
    lambda=lambda_list(experiment_no);

    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;

    % Compute loss before training
    %     loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,B,lambda);
    %     loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
    %     cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, B,0);
    %     cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, B,0);

    %     accuracy_train(experiment_no,1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
    %     accuracy_validate(experiment_no,1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);

    % Learning steps
    tic
    for i=1:epochs
        
        % Minibatch GD
        % [S,S_hat,P,H,MEW,V]= ComputeNetworkBatchNorm(X,W,B,NetParams);
        % [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradientsKLayerBatchNorm(X,H, Y, P, W, lambda, S_hat, NetParams,V,MEW,S);
        
        [W,b,NetParams.gammas,NetParams.betas,nt_hist,NetParams] = MiniBatchGDCyclicLearningBatchNorm(X_train, Y_train, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax, alpha,NetParams);
        time = toc;

        % Compute loss at the end of the epoch
        %         loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,B,lambda);
        %         loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,B,lambda);
        %         cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, B,0);
        %         cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, B,0);

        accuracy_train(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);
        
        percent = i/epochs*100;
        time_per_pecent = time/percent;
        time_left = (100-percent)*time_per_pecent;
        fprintf("Experiment %d - epochs %d completed ( %.2f)  - time left : %.2f \n",experiment_no,i,percent,time_left);

        % Shuffle training data
        shuffle = randperm(size(X_train, 2));
        X_train = X_train(:, shuffle);
        Y_train = Y_train(:,shuffle);
    end
            
    NetParams.use_precomputed = (1==1);
    accuracy_test(experiment_no) = ComputeAccuracyBatchNorm(X_test, Y_test, W, b,NetParams)*100;
    NetParams.use_precomputed = (1==0);
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_loss_learning_plots

%% Ex 3 - guassian weight

clear;

% Load all data
[X1, Y1, ~] = LoadBatch('data_batch_1.mat');
[X2, Y2, ~] = LoadBatch('data_batch_2.mat');
[X3, Y3, ~] = LoadBatch('data_batch_3.mat');
[X4, Y4, ~] = LoadBatch('data_batch_4.mat');
[X5, Y5, ~] = LoadBatch('data_batch_5.mat');

% Concat data
X = [X1 , X2, X3, X4, X5];
Y = [Y1 , Y2, Y3, Y4, Y5];

% Clear variables for memory
clear X1 X2 X3 X4 X5 Y1 Y2 Y3 Y4 Y5 ;

% Create training and validation data
n_validation = 5000;
X_train = X(:,1:end-n_validation); Y_train = Y(:,1:end-n_validation);
X_validate = X(:,end-n_validation+1:end); Y_validate = Y(:,end-n_validation+1:end);
n = size(X_train,2);

% Clear variables for memory
clear X Y ;

% Load test
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
fprintf("All data loaded \n");

% Preprocess the data
mean_X_train = mean(X_train, 2);
std_X_train = std(X_train, 0, 2);
mean_X_validate = mean(X_validate, 2);
std_X_validate = std(X_validate, 0, 2);
mean_X_test = mean(X_test, 2);
std_X_test = std(X_test, 0, 2);
fprintf("Data processed \n");

% Normalise wrt mean and std
X_train = X_train - repmat(mean_X_train, [1, size(X_train, 2)]);
X_train = X_train ./ repmat(std_X_train, [1, size(X_train, 2)]);
X_validate = X_validate - repmat(mean_X_validate, [1, size(X_validate, 2)]);
X_validate = X_validate ./ repmat(std_X_validate, [1, size(X_validate, 2)]);
X_test = X_test - repmat(mean_X_test, [1, size(X_test, 2)]);
X_test = X_test ./ repmat(std_X_test, [1, size(X_test, 2)]);
fprintf("Data normalised \n");

% Generate results
fprintf("Initalising network \n");

% Lambda course search
% l_min = -5;
% l_max = -1;
% l = -2.77777777777778;
lambda_list = [0.005];

% Setup paramater list
n_batch_list = [100];
eta_list = [0.001];
n_batch = 100;
alpha = 0.8;
sigma = 1e-4;   

% Setup cyclic learning
n = size(X_train,2);
nmin = 1e-5;
nmax = 1e-1;
t = 0;
n_s = 5*45000/n_batch;
% n_s = 2*floor(n/n_batch);
nt_hist = [nmin];
cycles = 2;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
loss_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
cost_train = zeros(length(lambda_list),1,n_epochs_list(1)+1);
cost_validate = zeros(length(lambda_list),1,n_epochs_list(1)+1);
accuracy_test = zeros(1,length(lambda_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(lambda_list));

% Loop through experiments
for experiment_no=1:length(lambda_list)

    % Reset parameters
    nt_hist = [nmin];

    % Init seed
    rng(400);

    % Init W and B
    intermediate_layers = [50,50];
    d = size(X_train,1);
    [W,B,gammas,betas] = InitParametersGuassian(X_train,Y_train,d,intermediate_layers,sigma);
    NetParams.W = W;
    NetParams.b = B;
    NetParams.use_bn = (1==1);
    NetParams.gammas = gammas;
    NetParams.betas = betas;
    NetParams.use_precomputed = (1==0);

    % Set paramts
    n_batch = n_batch_list(1);
    n_epochs = n_epochs_list(1);
    lambda=lambda_list(experiment_no);

    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;

    % Compute loss before training
    loss_train(experiment_no,1) = ComputeCostBatchNorm(X_train, Y_train, NetParams, lambda);
    loss_validate(experiment_no,1) = ComputeCostBatchNorm(X_validate, Y_validate, NetParams, lambda);
    %     cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, B,0);
    %     cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, B,0);

    %     accuracy_train(experiment_no,1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
    %     accuracy_validate(experiment_no,1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);

    % Learning steps
    tic
    for i=1:epochs
        
        % Minibatch GD
        % [S,S_hat,P,H,MEW,V]= ComputeNetworkBatchNorm(X,W,B,NetParams);
        % [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradientsKLayerBatchNorm(X,H, Y, P, W, lambda, S_hat, NetParams,V,MEW,S);
        
        [W,b,NetParams.gammas,NetParams.betas,nt_hist,NetParams] = MiniBatchGDCyclicLearningBatchNorm(X_train, Y_train, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax, alpha,NetParams);
        time = toc;
        NetParams.W = W;
        NetParams.b = b;

        % Compute loss at the end of the epoch
        loss_train(experiment_no,i+1) = ComputeCostBatchNorm(X_train, Y_train, NetParams, lambda);
        loss_validate(experiment_no,i+1) = ComputeCostBatchNorm(X_validate, Y_validate, NetParams, lambda);
        %         cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, B,0);
        %         cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, B,0);

        accuracy_train(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_train, Y_train, W, B, NetParams);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracyBatchNorm(X_validate, Y_validate, W, B, NetParams);
        
        percent = i/epochs*100;
        time_per_pecent = time/percent;
        time_left = (100-percent)*time_per_pecent;
        fprintf("Experiment %d - epochs %d completed ( %.2f)  - time left : %.2f \n",experiment_no,i,percent,time_left);

        % Shuffle training data
        shuffle = randperm(size(X_train, 2));
        X_train = X_train(:, shuffle);
        Y_train = Y_train(:,shuffle);
    end
            
    NetParams.use_precomputed = (1==1);
    accuracy_test(experiment_no) = ComputeAccuracyBatchNorm(X_test, Y_test, W, b,NetParams)*100;
    NetParams.use_precomputed = (1==0);
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_loss_learning_plots