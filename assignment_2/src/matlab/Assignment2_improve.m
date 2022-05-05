%% Increase number of hidden nodes
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
l1 = 0.00107583196648000;
l2 = l1*1.1;
l3 = l1*1.2;
lambda_list = [0.01];
m_list = [300];
n_batch = 100;
% eta_list = [0.001];

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
    m = m_list(experiment_no);
    d = size(X_train,1);
    [W1,b1,W2,b2] = InitParameters(X_train,Y_train,d,m);
    W = {W1;W2};
    b = {b1;b2};

    % Set paramts

    n_epochs = n_epochs_list(1);
    lambda=lambda_list(experiment_no);

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

%% Dropout
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
lambda_list = [0.002];
m_list = [150];
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
    m = m_list(experiment_no);
    d = size(X_train,1);
    [W1,b1,W2,b2] = InitParameters(X_train,Y_train,d,m);
    W = {W1;W2};
    b = {b1;b2};

    % Set paramts
    n_epochs = n_epochs_list(1);
    lambda=lambda_list(experiment_no);

    % Set dropout
    p = 0.3;

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
        [W,b,nt_hist] = MiniBatchGDCyclicLearningDropout(X_train, Y_train, GDparams, W, b, lambda, nt_hist, n_s, nmin, nmax,p);
    
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

%% Data augmentation
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

images = size(X_train,2);
flipped_images = rand(1,images) > 0.5;

dx = randi([0,10],1,images);
dy = randi([0,10],1,images);

% Flip images
for i = 1:images
    if flipped_images(i)
         X_train(:,i) = FlipImage(X_train(:,i));
    end
end

% Shift images
for i = 1:images
    X_train(:,i) = ShiftImage(X_train(:,i),dx(i), dy(i));
end

% get image of horse
image = permute(reshape(X_train(:,8)',32, 32, 3),[2,1,3]);
imshow(image);

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
cycles = 10;
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

%% Extensive testing
clear;

% Grid search parameters
grid_n = 3;

% lambda_min = 0.000599499587887344;
% lambda_max = 0.00107583196648000;
% lambda = lambda_min:(lambda_max-lambda_min)/(grid_n-1):lambda_max;
% 
% min = 0.3;
% max = 0.7;
% p = min:(max-min)/(grid_n-1):max;
% 
% min = 0;
% max = 0.2;
% anneal = min:(max-min)/(grid_n-1):max;
% 
% cycles = [3:2:7];

lambda = [0.000599499587887344];
p = [0.7];
cycles = [3];
anneal = [0,0.1,0.2];


parameters = GridSearch(lambda,p,cycles,anneal);

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

% Clear variables for memory
clear X Y ;

% Load test
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
fprintf("All data loaded \n");

% Data augmentation
images = size(X_train,2);
flipped_images = rand(1,images) > 0.5;

shift_image = rand(1,images) < 0.25;
dx = randi([0,3],1,images);
dy = randi([0,3],1,images);
dx = dx.*shift_image;
dy = dy.*shift_image;

% Flip and shift images
for i = 1:images
    if flipped_images(i)
         X_train(:,end+1) = FlipImage(X_train(:,i));
         Y_train(:,end+1) = Y_train(:,i);
         if dx(i) ~=0 ||  dy(i) ~=0 
            X_train(:,end) = ShiftImage(X_train(:,end),dx(i), dy(i));
        end
    end
    if dx(i) ~=0 ||  dy(i) ~=0 
        X_train(:,end+1) = ShiftImage(X_train(:,i),dx(i), dy(i));
        Y_train(:,end+1) = Y_train(:,i);
    end
end

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
m_list = [150];
n_batch = 100;

% Setup matrices for datastorage

% most turned off for sake of speed
% loss_train = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));
% loss_validate = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));
% accuracy_train = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));
% accuracy_validate = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));
% cost_train = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));
% cost_validate = zeros(length(lambda_list),1,ceil(n_epochs_list(1)+1));

accuracy_test = zeros(1,length(parameters)+1);
% W_history = zeros(size(W,1),size(W,2),length(eta_list));

% Loop through experiments
for experiment_no=1:length(parameters)

    % Get parameters
    lambda = parameters(experiment_no,1);
    p = parameters(experiment_no,2);
    cycles = parameters(experiment_no,3);
    anneal = parameters(experiment_no,4);

    % Setup cyclic learning
    n = size(X_train,2);
    nmin = 1e-5;
    nmax = 1e-1;
    t = 0;
    n_s = 2*floor(n/n_batch);
    nt_hist = [nmin];
    epochs = cycles*2*n_s/(size(X_train,2)/n_batch);
    n_epochs_list = [epochs];

    % Reset parameters
    nt_hist = [nmin];

    % Init seed
    rng(400);

    % Init W and B
    m = 150;
    d = size(X_train,1);
    [W1,b1,W2,b2] = InitParameters(X_train,Y_train,d,m);
    W = {W1;W2};
    b = {b1;b2};

    % Set paramts
    n_epochs = n_epochs_list(1);
    n = size(X_train,2);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;

    % Compute loss before training

    %     loss_train(experiment_no,1) = ComputeCost(X_train,Y_train, W,b,lambda);
    %     loss_validate(experiment_no,1) = ComputeCost(X_validate,Y_validate, W,b,lambda);
    %     accuracy_train(experiment_no,1) = ComputeAccuracy(X_train, Y_train, W, b);
    %     accuracy_validate(experiment_no,1) = ComputeAccuracy(X_validate, Y_validate, W, b);
    %     cost_train(experiment_no,1) = ComputeCost(X_train, Y_train, W, b,0);
    %     cost_validate(experiment_no,1) = ComputeCost(X_validate, Y_validate, W, b,0);

    % Learning steps
    for i=1:ceil(epochs)
        
        % Minibatch GD
        [W,b,nt_hist] = MiniBatchGDCyclicLearningDropout(X_train, Y_train, GDparams, W, b, lambda, nt_hist, n_s, nmin, nmax,anneal,p);
    
        % Compute loss at the end of the epoch

        %         loss_train(experiment_no,i+1) = ComputeCost(X_train,Y_train, W,b,lambda);
        %         loss_validate(experiment_no,i+1) = ComputeCost(X_validate,Y_validate, W,b,lambda);
        %         accuracy_train(experiment_no,i+1) = ComputeAccuracy(X_train, Y_train, W, b);
        %         accuracy_validate(experiment_no,i+1) = ComputeAccuracy(X_validate, Y_validate, W, b);
        %         cost_train(experiment_no,i+1) = ComputeCost(X_train, Y_train, W, b,0);
        %         cost_validate(experiment_no,i+1) = ComputeCost(X_validate, Y_validate, W, b,0);
    
        fprintf("Experiment %d - cycles %d completed ( %.2f) \n",experiment_no,i,length(nt_hist)/(cycles*2*n_s)*100);
    end

        accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    %     W_history(:,:,experiment_no) = W;
    %     [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end

make_loss_learning_plots


