% Parameters
% n_batch       the size of the mini-batch
% eta           the learning rate
% n_epcohs      the number of runs through the whole training set
% n_epcohs      the number of runs through the whole training set
% lambda        normalisation factor
% shuffle       shuffle the minibatches after each epoch (not implemented)

n_batch = 100;
n_epochs = 40;
shuffle = 0;
lambda=0.5;
eta = 0.001;

% Init seed
rng(400);

% Create GDparams;
GDparams.n_batch = n_batch;
GDparams.n_epochs = n_epochs;
GDparams.eta = eta;

% TO DO 
% - implement randperm
% - 

%% Load the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validate, Y_validate, y_tvalidate] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

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

%% Init W and B
W = normrnd(0,0.01,[size(Y_train,1),size(X_test,1)]);
b = normrnd(0,0.01,[size(Y_train,1),1]);

%% Init loss function
loss_train = zeros(1,n_epochs);
loss_validate = zeros(1,n_epochs);
accuracy_train = zeros(1,n_epochs);
accuracy_validate = zeros(1,n_epochs);
accuracy_test = 0;

%% Learn

% Mini batches
for i=1:GDparams.n_epochs
    
    % Create minibatches and do learning
    for j=1:size(X_train,2)/GDparams.n_batch
        
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X_train(:, j_start:j_end);
        Ybatch = Y_train(:, j_start:j_end);

        [W,b] = MiniBatchGD(Xbatch, Ybatch, GDparams, W, b, lambda);
    end

    % Compute loss at the end of the epoch
    loss_train(i) = ComputeCost(X_train,Y_train, W,b,lambda);
    loss_validate(i) = ComputeCost(X_validate,Y_validate, W,b,lambda);
    accuracy_train(i) = ComputeAccuracy(X_train, Y_train, W, b);
    accuracy_validate(i) = ComputeAccuracy(X_validate, Y_validate, W, b);

    fprintf("Epoch %d completed\n",i);
end

accuracy_test = ComputeAccuracy(X_test, Y_test, W, b)*100;

%% Show results

fprintf("Accuracy on Test Batch %0.2f\n",accuracy_test);

% Loss graph
figure;
plot(loss_train);
hold on;
plot(loss_validate);
legend('Training set','Validation set');
ylabel('Epoch');
xlabel('Loss');
title(sprintf('Loss over time (lambda = %0.3f,eta=%0.3f)',lambda,GDparams.eta));
grid;

% Accuracy graph
figure;
plot(accuracy_train);
hold on;
plot(accuracy_validate);
legend('Training set','Validation set');
ylabel('Epoch');
xlabel('Accuracy (%)');
title(sprintf('Accuracy over time (lambda = %0.3f,eta=%0.3f)',lambda,GDparams.eta));
grid;gt

%% Show templates
for i=1:10
    im = reshape(W(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
montage(s_im, 'Size', [2,5]);
