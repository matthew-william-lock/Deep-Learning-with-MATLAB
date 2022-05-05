%% Load the data
clear;
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validate, Y_validate, y_tvalidate] = LoadBatch('data_batch_2.mat');
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

% %% Test gradients
% 
% X = X_train(:, 1:100);
% Y = Y_train(:, 1:100);
% 
% m = 50;
% d = size(X_train,1);
% lambda = 1;
% 
% % Init W and B
% [W1,b1,W2,b2] = InitParameters(X,Y,d,m);
% 
% W = {W1;W2};
% b = {b1;b2};
% 
% % Compute analytical gradients
% [s1,s,P,H] = ComputeNetwork(X,W1,W2,b1,b2);
% [grad_W1, grad_b1,grad_W2, grad_b2] = ComputeGradients2Layer(X,H, Y, P, W1, W2, lambda);
% 
% % Calculate numerical gradients
% [grad_b_num, grad_W_num] = ComputeGradsNum(X, Y, W, b, lambda, 1e-5);
% [b_num_s, W_num_s] = ComputeGradsNumSlow(X, Y, W, b, lambda, 1e-5);
% 
% % Split numerical results
% grad_W_num_1 = cell2mat(grad_W_num(1));
% grad_W_num_2 = cell2mat(grad_W_num(2));
% grad_b_num_1 = cell2mat(grad_b_num(1));
% grad_b_num_2 = cell2mat(grad_b_num(2));
% grad_W_num_1_slow = cell2mat(W_num_s(1));
% grad_W_num_2_slow = cell2mat(W_num_s(2));
% grad_b_num_1_slow = cell2mat(b_num_s(1));
% grad_b_num_2_slow = cell2mat(b_num_s(2));
% 
% % Computer difference between analytical and numerical
% rel_dif_W1 = abs(grad_W1-grad_W_num_1)/max(10^-6,norm(grad_W1)+norm(grad_W_num_1));
% rel_dif_W1_s = abs(grad_W1-grad_W_num_1_slow)/max(10^-6,norm(grad_W1)+norm(grad_W_num_1_slow));
% rel_dif_b1 = abs(grad_b1-grad_b_num_1)/max(10^-6,norm(grad_b1)+norm(grad_b_num_1));
% rel_dif_b1_s = abs(grad_b1-grad_b_num_1_slow)/max(10^-6,norm(grad_b1)+norm(grad_b_num_1_slow));
% rel_dif_W2 = abs(grad_W2-grad_W_num_2)/max(10^-6,norm(grad_W2)+norm(grad_W_num_2));
% rel_dif_W2_s = abs(grad_W2-grad_W_num_2_slow)/max(10^-6,norm(grad_W2)+norm(grad_W_num_2_slow));
% rel_dif_b2 = abs(grad_b2-grad_b_num_2)/max(10^-6,norm(grad_b2)+norm(grad_b_num_2));
% rel_dif_b2_s = abs(grad_b2-grad_b_num_2_slow)/max(10^-6,norm(grad_b2)+norm(grad_b_num_2_slow));
% 
% % Determine max difference
% max_rel_diff_W1 = max(max((rel_dif_W1),[],'all'),max((rel_dif_W1_s),[],'all'));
% max_rel_diff_b1 = max(max((rel_dif_b1),[],'all'),max((rel_dif_b1_s),[],'all'));
% max_rel_diff_W2 = max(max((rel_dif_W2),[],'all'),max((rel_dif_W2_s),[],'all'));
% max_rel_diff_b2 = max(max((rel_dif_b2),[],'all'),max((rel_dif_b2_s),[],'all'));
% 
% 
% % Check results
% % val= abs(grad_b1-grad_b_num_1)/max(10^-6,norm(grad_b1)+norm(grad_b_num_1));
% % val_max_single = max(max(val));

%% Exercise 3

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
    m = 50;
    d = size(X_train,1);
    [W1,b1,W2,b2] = InitParameters(X_train,Y_train,d,m);
    W = {W1;W2};
    b = {b1;b2};

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
    loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,b,lambda);
    loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,b,lambda);
    accuracy_train(experiment_no,1) = ComputeAccuracy(X_train, Y_train, W, b);
    accuracy_validate(experiment_no,1) = ComputeAccuracy(X_validate, Y_validate, W, b);
    cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, b,0);
    cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, b,0);

    % Learning steps
    for i=1:epochs
        
        % Minibatch GD
        [W,b,nt_hist] = MiniBatchGDCyclicLearning(X_train, Y_train, GDparams, W, b, lambda, nt_hist, n_s, nmin, nmax);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,b,lambda);
        loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,b,lambda);
        accuracy_train(experiment_no,i+1) = ComputeAccuracy(X_train, Y_train, W, b);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracy(X_validate, Y_validate, W, b);
        cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, b,0);
        cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, b,0);
    
        fprintf("Experiment %d - cycles %d completed ( %.2f) \n",experiment_no,i,length(nt_hist)/(cycles*2*n_s)*100);
    end

    %     accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

% make_loss_learning_plots

%% Exercise 4

% Setup paramater list
lambda_list = [0.01];
n_batch_list = [100];
eta_list = [0.001];

% Setup cyclic learning
nmin = 1e-5;
nmax = 1e-1;
t = 0;
n_s = 800;
nt_hist = [nmin];
cycles = 3;
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
    m = 50;
    d = size(X_train,1);
    [W1,b1,W2,b2] = InitParameters(X_train,Y_train,d,m);
    W = {W1;W2};
    b = {b1;b2};

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
    loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,b,lambda);
    loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,b,lambda);
    accuracy_train(experiment_no,1) = ComputeAccuracy(X_train, Y_train, W, b);
    accuracy_validate(experiment_no,1) = ComputeAccuracy(X_validate, Y_validate, W, b);
    cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, b,0);
    cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, b,0);

    % Learning steps
    for i=1:epochs
        
        % Minibatch GD
        [W,b,nt_hist] = MiniBatchGDCyclicLearning(X_train, Y_train, GDparams, W, b, lambda, nt_hist, n_s, nmin, nmax);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,b,lambda);
        loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,b,lambda);
        accuracy_train(experiment_no,i+1) = ComputeAccuracy(X_train, Y_train, W, b);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracy(X_validate, Y_validate, W, b);
        cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, b,0);
        cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, b,0);
    
        fprintf("Experiment %d - cycles %d completed ( %.2f) \n",experiment_no,i,length(nt_hist)/(cycles*2*n_s)*100);
    end

    %     accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

% make_loss_learning_plots

%% Exercise 4.b (course random search)
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
n = size(X,2);
n_validation = 5000;
X_train = X(:,1:end-n_validation); Y_train = Y(:,1:end-n_validation);
X_validate = X(:,end-n_validation+1:end); Y_validate = Y(:,end-n_validation+1:end);

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
% Course search for lambda
l_min = -3.8571;
l_max = -2.7143;
% l_min = 5.2e-4;
% l_max = 0.0072;
lambda_list = l_min:(l_max-l_min)/9:l_max;
lambda_list = 10.^lambda_list;
n_batch = 100;

% Setup cyclic learning
nmin = 1e-5;
nmax = 1e-1;
t = 0;
n_s = 2*floor(n/n_batch);
nt_hist = [nmin];
cycles = 2;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(lambda_list),1,round(n_epochs_list(1)+1));
loss_validate = zeros(length(lambda_list),1,round(n_epochs_list(1)+1));
accuracy_train = zeros(length(lambda_list),1,round(n_epochs_list(1)+1));
accuracy_validate = zeros(length(lambda_list),1,round(n_epochs_list(1)+1));
cost_train = zeros(length(lambda_list),1,round(n_epochs_list(1)+1));
cost_validate = zeros(length(lambda_list),1,round(n_epochs_list(1)+1));
accuracy_test = zeros(1,length(lambda_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(eta_list));

% Loop through experiments
for experiment_no=1:length(lambda_list)

    % Reset parameters
    nt_hist = [nmin];

    % Init seed
    rng(400);

    % Init W and B
    m = 50;
    d = size(X_train,1);
    [W1,b1,W2,b2] = InitParameters(X_train,Y_train,d,m);
    W = {W1;W2};
    b = {b1;b2};

    % Set paramts

    n_epochs = n_epochs_list(1);
    lambda=lambda_list(experiment_no);

    k = 2;
    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;

    % Compute loss before training
    loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,b,lambda);
    loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,b,lambda);
    accuracy_train(experiment_no,1) = ComputeAccuracy(X_train, Y_train, W, b);
    accuracy_validate(experiment_no,1) = ComputeAccuracy(X_validate, Y_validate, W, b);
    cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, b,0);
    cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, b,0);

    % Learning steps
    for i=1:ceil(epochs)
        
        % Minibatch GD
        [W,b,nt_hist] = MiniBatchGDCyclicLearning(X_train, Y_train, GDparams, W, b, lambda, nt_hist, n_s, nmin, nmax);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,b,lambda);
        loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,b,lambda);
        accuracy_train(experiment_no,i+1) = ComputeAccuracy(X_train, Y_train, W, b);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracy(X_validate, Y_validate, W, b);
        cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, b,0);
        cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, b,0);
    
        fprintf("Experiment %d - cycles %d completed ( %.2f) \n",experiment_no,i,length(nt_hist)/(cycles*2*n_s)*100);
    end

        accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

% make_many_plots

%% Final training
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
n_validation = 1000;
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
% Course search for lambda
lambda_list = [0.00107583196648000];
n_batch = 100;
eta_list = [0.001];

% Setup cyclic learning
nmin = 1e-5;
nmax = 1e-1;
t = 0;
n_s = 2*floor(n/n_batch);
nt_hist = [nmin];
cycles = 3;
epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
n_epochs_list = [epochs];

% Setup matrices for datastorage
loss_train = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));
loss_validate = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));
accuracy_train = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));
accuracy_validate = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));
cost_train = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));
cost_validate = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));
accuracy_test = zeros(1,length(lambda_list)+1);
% W_history = zeros(size(W,1),size(W,2),length(eta_list));

% Loop through experiments
for experiment_no=1:length(lambda_list)

    % Reset parameters
    nt_hist = [nmin];

    % Init seed
    rng(400);

    % Init W and B
    m = 50;
    d = size(X_train,1);
    [W1,b1,W2,b2] = InitParameters(X_train,Y_train,d,m);
    W = {W1;W2};
    b = {b1;b2};

    % Set paramts

    n_epochs = n_epochs_list(1);
    lambda=lambda_list(experiment_no);

    k = 2;
    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;

    % Compute loss before training
    loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,b,lambda);
    loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,b,lambda);
    accuracy_train(experiment_no,1) = ComputeAccuracy(X_train, Y_train, W, b);
    accuracy_validate(experiment_no,1) = ComputeAccuracy(X_validate, Y_validate, W, b);
    cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, b,0);
    cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, b,0);

    % Learning steps
    for i=1:ceil(epochs)
        
        % Minibatch GD
        [W,b,nt_hist] = MiniBatchGDCyclicLearning(X_train, Y_train, GDparams, W, b, lambda, nt_hist, n_s, nmin, nmax);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,b,lambda);
        loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,b,lambda);
        accuracy_train(experiment_no,i+1) = ComputeAccuracy(X_train, Y_train, W, b);
        accuracy_validate(experiment_no,i+1) = ComputeAccuracy(X_validate, Y_validate, W, b);
        cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, b,0);
        cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, b,0);
    
        fprintf("Experiment %d - cycles %d completed ( %.2f) \n",experiment_no,i,length(nt_hist)/(cycles*2*n_s)*100);
    end

        accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_loss_learning_plots

%% Functions 
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

function names = LoadLabelNames(filename)
    
    % Load batch
    A = load(filename);
    names =  A.label_names;

end

function [W1,b1,W2,b2] = InitParameters(X_test,Y_train,d,m)
%INITPARAMETERS Summary of this function goes here
%   Detailed explanation goes here
    W1 = normrnd(0,1/sqrt(d),[m,size(X_test,1)]);
    b1 = zeros([m,1]);
    W2 = normrnd(0,1/sqrt(m),[size(Y_train,1),m]);
    b2 = zeros([size(Y_train,1),1]);
end

function [s1,s,P,H] = ComputeNetwork(X,W1,W2,b1,b2)
%COMPUTENETWORK Summary of this function goes here
%   Detailed explanation goes here

s1 = W1*X + b1;
H = max(s1,0);
s = W2*H + b2;
P = exp(s)./sum(exp(s));

end

function [grad_W1, grad_b1,grad_W2, grad_b2] = ComputeGradients2Layer(X,H, Y, P, W1, W2, lambda)

% grad W is gradient matrix of the cost J relative to W and has size K×d.
% grad b is gradient vector of the cost J relative to b and has size K×1.
% J => cost


Pbatch =  P;
HBatch = H;

% Determine Gbatch
Gbatch = -(Y-Pbatch);

% Get number of examples
nb = size(X,2);

% Determine gradients
dLdW2 = (1/nb)* Gbatch * HBatch';
dLdB2 = (1/nb)*Gbatch*ones(size(X,2),1);

% Set Gbatch
Gbatch = W2'*Gbatch;
Gbatch = Gbatch .* (HBatch > 0)*1;

% Determine gradients
dLdW1 = (1/nb)* Gbatch * X';
dLdB1 = (1/nb)*Gbatch*ones(size(X,2),1);

% return gradients
grad_W2 = dLdW2 + 2*lambda*W2;
grad_b2 = dLdB2;
grad_W1 = dLdW1 + 2*lambda*W1;
grad_b1 = dLdB1;

end

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

[c, ~] = ComputeCost(X, Y, W, b, lambda);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        fprintf(' %d/%d %d/%d \n ',j, length(W),i,length(b{j}));
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})   
        fprintf(' %d/%d %d/%d \n ',j, length(W),i,numel(W{j}));
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-c) / h;
    end
end

end

function [J,A] = ComputeCost(X, Y, W, b, lambda)
    
    % Get probabilities
    [~,~,P,~] = ComputeNetwork(X,cell2mat(W(1)),cell2mat(W(2)),cell2mat(b(1)),cell2mat(b(2)));
    
    % Calculate cost (same as taking loss on the diagonal?)
    % For single input, i.e. size(Y) = 10,1 => same as cost = Y'*-log(P)
    cost = sum(-log(P(Y==1))) / size(P,2);

    % Calculate regularisation cost
    reg = lambda* (norm(cell2mat(W(1)),"fro")^2 + norm(cell2mat(W(2)),"fro")^2);

    J = cost + reg;
    A=J;
        

end
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})

        fprintf(' %d/%d %d/%d \n ',j, length(W),i,length(b{j}));
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        fprintf(' %d/%d %d/%d \n ',j, length(W),i,numel(W{j}));
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end

end

function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)

    % Create minibatches and do learning
    for j=1:size(X,2)/GDparams.n_batch
        
        j_start = (j-1)*GDparams.n_batch + 1;
        j_end = j*GDparams.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        W1 = cell2mat(W(1));
        W2 = cell2mat(W(2));
        b1 = cell2mat(b(1));
        b2 = cell2mat(b(2));

        [~,~,P,H] = ComputeNetwork(Xbatch,W1,W2,b1,b2);
        [grad_W1, grad_b1,grad_W2, grad_b2] = ComputeGradients2Layer(Xbatch,H, Ybatch, P, W1, W2, lambda);

        W(1) = {W1 - GDparams.eta*grad_W1};
        W(2) = {W2 - GDparams.eta*grad_W2};
        b(1) = {b1 - GDparams.eta*grad_b1};
        b(2) = {b2 - GDparams.eta*grad_b2};

    end

    Wstar = W;
    bstar = b;

end

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

function [Wstar, bstar, nt_hist] = MiniBatchGDCyclicLearning(X, Y, GDparams, W, b, lambda, nt_hist, n_s, nmin, nmax)

    % Create minibatches and do learning
    for j=1:size(X,2)/GDparams.n_batch
        
        j_start = (j-1)*GDparams.n_batch + 1;
        j_end = j*GDparams.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        W1 = cell2mat(W(1));
        W2 = cell2mat(W(2));
        b1 = cell2mat(b(1));
        b2 = cell2mat(b(2));

        [~,~,P,H] = ComputeNetwork(Xbatch,W1,W2,b1,b2);
        [grad_W1, grad_b1,grad_W2, grad_b2] = ComputeGradients2Layer(Xbatch,H, Ybatch, P, W1, W2, lambda);

        t = length(nt_hist)-1;
        l = floor(t/(2*n_s));

        if (2*l*n_s) <= t && t<= ((2*l+1)*n_s)
            nt = nmin + (t-2*l*n_s)/n_s * (nmax-nmin);
        else
            nt = nmax - (t-(2*l+1)*n_s)/n_s * (nmax-nmin);
        end

        nt_hist(end+1) = nt;

        W(1) = {W1 - nt*grad_W1};
        W(2) = {W2 - nt*grad_W2};
        b(1) = {b1 - nt*grad_b1};
        b(2) = {b2 - nt*grad_b2};

    end

    Wstar = W;
    bstar = b;

end





