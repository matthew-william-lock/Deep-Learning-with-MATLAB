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
    intermediate_layers = [50, 50];
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
        
        [W,b,NetParams.gammas,NetParams.betas,nt_hist,NetParams] = MiniBatchGDCyclicLearningPreciseBatchNorm(X_train, Y_train, GDparams, W, B, lambda, nt_hist, n_s, nmin, nmax, alpha,NetParams);
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