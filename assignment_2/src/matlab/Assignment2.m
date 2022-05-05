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

%% Test gradients

X = X_train(:, 1:100);
Y = Y_train(:, 1:100);

m = 50;
d = size(X_train,1);
lambda = 1;

% Init W and B
[W1,b1,W2,b2] = InitParameters(X,Y,d,m);

W = {W1;W2};
b = {b1;b2};

% Compute analytical gradients
[s1,s,P,H] = ComputeNetwork(X,W1,W2,b1,b2);
[grad_W1, grad_b1,grad_W2, grad_b2] = ComputeGradients2Layer(X,H, Y, P, W1, W2, lambda);

% Calculate numerical gradients
[grad_b_num, grad_W_num] = ComputeGradsNum(X, Y, W, b, lambda, 1e-5);
[b_num_s, W_num_s] = ComputeGradsNumSlow(X, Y, W, b, lambda, 1e-5);

% Split numerical results
grad_W_num_1 = cell2mat(grad_W_num(1));
grad_W_num_2 = cell2mat(grad_W_num(2));
grad_b_num_1 = cell2mat(grad_b_num(1));
grad_b_num_2 = cell2mat(grad_b_num(2));
grad_W_num_1_slow = cell2mat(W_num_s(1));
grad_W_num_2_slow = cell2mat(W_num_s(2));
grad_b_num_1_slow = cell2mat(b_num_s(1));
grad_b_num_2_slow = cell2mat(b_num_s(2));

% Computer difference between analytical and numerical
rel_dif_W1 = abs(grad_W1-grad_W_num_1)/max(10^-6,norm(grad_W1)+norm(grad_W_num_1));
rel_dif_W1_s = abs(grad_W1-grad_W_num_1_slow)/max(10^-6,norm(grad_W1)+norm(grad_W_num_1_slow));
rel_dif_b1 = abs(grad_b1-grad_b_num_1)/max(10^-6,norm(grad_b1)+norm(grad_b_num_1));
rel_dif_b1_s = abs(grad_b1-grad_b_num_1_slow)/max(10^-6,norm(grad_b1)+norm(grad_b_num_1_slow));
rel_dif_W2 = abs(grad_W2-grad_W_num_2)/max(10^-6,norm(grad_W2)+norm(grad_W_num_2));
rel_dif_W2_s = abs(grad_W2-grad_W_num_2_slow)/max(10^-6,norm(grad_W2)+norm(grad_W_num_2_slow));
rel_dif_b2 = abs(grad_b2-grad_b_num_2)/max(10^-6,norm(grad_b2)+norm(grad_b_num_2));
rel_dif_b2_s = abs(grad_b2-grad_b_num_2_slow)/max(10^-6,norm(grad_b2)+norm(grad_b_num_2_slow));

% Determine max difference
max_rel_diff_W1 = max(max((rel_dif_W1),[],'all'),max((rel_dif_W1_s),[],'all'));
max_rel_diff_b1 = max(max((rel_dif_b1),[],'all'),max((rel_dif_b1_s),[],'all'));
max_rel_diff_W2 = max(max((rel_dif_W2),[],'all'),max((rel_dif_W2_s),[],'all'));
max_rel_diff_b2 = max(max((rel_dif_b2),[],'all'),max((rel_dif_b2_s),[],'all'));


% Check results
% val= abs(grad_b1-grad_b_num_1)/max(10^-6,norm(grad_b1)+norm(grad_b_num_1));
% val_max_single = max(max(val));

%% Exercise 2 (sanity chec)

X_train = X_train(:,1:100);
Y_train = Y_train(:,1:100);

% Setup paramater list
lambda_list = [0];
n_epochs_list = [200];
n_batch_list = [100];
eta_list = [0.01];

% Setup matrices for datastorage
loss_train = zeros(length(eta_list),1,n_epochs_list(1));
loss_validate = zeros(length(eta_list),1,n_epochs_list(1));
accuracy_train = zeros(length(eta_list),1,n_epochs_list(1));
accuracy_validate = zeros(length(eta_list),1,n_epochs_list(1));
cost_train = zeros(length(eta_list),1,n_epochs_list(1));
cost_validate = zeros(length(eta_list),1,n_epochs_list(1));
accuracy_test = zeros(1,length(eta_list));
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

    cycles = 6;

    % Learning steps
    for i=1:200
        
        % Minibatch GD
        [W,b] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i) = ComputeCost(X_train,Y_train, W,b,lambda);
        loss_validate(experiment_no,i) = ComputeCost(X_validate,Y_validate, W,b,lambda);
        accuracy_train(experiment_no,i) = ComputeAccuracy(X_train, Y_train, W, b);
        accuracy_validate(experiment_no,i) = ComputeAccuracy(X_validate, Y_validate, W, b);
        %         cost_train(experiment_no,i) = ComputeCost(X_train, Y_train, W, b,0);
        %         cost_validate(experiment_no,i) = ComputeCost(X_validate, Y_validate, W, b,0);
    
        fprintf("Experiment %d - cycles %d completed (%0.2f ) \n",experiment_no,i,i/(cycles*2*cyclic.n_s));
    end

    %     accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_plots

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

make_loss_learning_plots

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

make_loss_learning_plots

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

