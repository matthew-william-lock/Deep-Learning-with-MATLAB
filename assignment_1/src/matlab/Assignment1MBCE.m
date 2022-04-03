% Parameters
% n_batch       the size of the mini-batch
% eta           the learning rate
% n_epcohs      the number of runs through the whole training set
% n_epcohs      the number of runs through the whole training set
% lambda        normalisation factor
% shuffle       shuffle the minibatches after each epoch (not implemented)

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

%% Init loss function

%% Generate results

% Setup paramater list
lambda_list = [0,0,0.1,1];
n_epochs_list = [40,40,40,40];
n_batch_list = [100,100,100,100];
eta_list = [0.1,0.001,0.001,0.001];
% lambda_list = [0.5];
% n_epochs_list = [40];
% n_batch_list = [100];
% eta_list = [0.001];

% Setup matrices for datastorage
W = normrnd(0,0.01,[size(Y_train,1),size(X_test,1)]);
loss_train = zeros(length(eta_list),1,40);
loss_validate = zeros(length(eta_list),1,40);
accuracy_train = zeros(length(eta_list),1,40);
accuracy_validate = zeros(length(eta_list),1,40);
cost_train = zeros(length(eta_list),1,40);
cost_validate = zeros(length(eta_list),1,40);
accuracy_test = zeros(1,length(eta_list));
W_history = zeros(size(W,1),size(W,2),length(eta_list));

% Loop through experiments
for experiment_no=1:length(eta_list)

    % Init seed
    rng(400);

    % Init W and B
    W = normrnd(0,0.01,[size(Y_train,1),size(X_test,1)]);
    b = normrnd(0,0.01,[size(Y_train,1),1]);

    % Set paramts
    n_batch = n_batch_list(experiment_no);
    n_epochs = n_epochs_list(experiment_no);
    lambda=lambda_list(experiment_no);
    eta = eta_list(experiment_no);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;

    % Epochs
    for i=1:GDparams.n_epochs
        
        % Minibatch GD
        [W,b] = MiniBatchGDMBCE(X_train, Y_train, GDparams, W, b, lambda);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i) = ComputeCostMBCE(X_train,Y_train, W,b,lambda);
        loss_validate(experiment_no,i) = ComputeCostMBCE(X_validate,Y_validate, W,b,lambda);
        accuracy_train(experiment_no,i) = ComputeAccuracyMBCE(X_train, Y_train, W, b);
        accuracy_validate(experiment_no,i) = ComputeAccuracyMBCE(X_validate, Y_validate, W, b);
        cost_train(experiment_no,i) = ComputeCostMBCE(X_train, Y_train, W, b,0);
        cost_validate(experiment_no,i) = ComputeCostMBCE(X_validate, Y_validate, W, b,0);
    
        fprintf("Experiment %d - Epoch %d completed\n",experiment_no,i);
    end

    accuracy_test(experiment_no) = ComputeAccuracyMBCE(X_test, Y_test, W, b)*100;
    W_history(:,:,experiment_no) = W;
    [correct, incorrect] = ComputeAccuracyHisogramMBCE(X_test,Y_test,W,b);

end



%% Show results

n = length(eta_list);

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:n
    
    a = ceil(n/2);

    h = subplot(a,a,experiment_no);

    % Set paramts
    n_batch = n_batch_list(experiment_no);
    n_epochs = n_epochs_list(experiment_no);
    lambda=lambda_list(experiment_no);
    eta = eta_list(experiment_no);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;

    % Loss graph
    plot(loss_train(experiment_no,:));
    hold on;
    plot(loss_validate(experiment_no,:));
    legend('Training set','Validation set');
    xlabel('Epoch');
    ylabel('Loss');
    title(sprintf('Loss (lambda = %0.1f,eta=%0.3f)',lambda,GDparams.eta));
    grid;
    
    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('loss_standard_%d.pdf',experiment_no));
   
end

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:n

    h = subplot(a,a,experiment_no);

    % Set paramts
    n_batch = n_batch_list(experiment_no);
    n_epochs = n_epochs_list(experiment_no);
    lambda=lambda_list(experiment_no);
    eta = eta_list(experiment_no);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;
    
    % Accuracy graph
    plot(accuracy_train(experiment_no,:));
    hold on;
    plot(accuracy_validate(experiment_no,:));
    legend('Training set','Validation set');
    xlabel('Epoch');
    ylabel('Accuracy (%)');
    title(sprintf('Accuracy (lambda = %0.1f,eta=%0.3f)',lambda,GDparams.eta));
    grid;
    
    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('accuracy_standard_%d.pdf',experiment_no));

end

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:n

    h = subplot(a,a,experiment_no);

    % Set paramts
    n_batch = n_batch_list(experiment_no);
    n_epochs = n_epochs_list(experiment_no);
    lambda=lambda_list(experiment_no);
    eta = eta_list(experiment_no);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;
    
    % Accuracy graph
    plot(cost_train(experiment_no,:));
    hold on;
    plot(cost_validate(experiment_no,:));
    legend('Training set','Validation set');
    xlabel('Epoch');
    ylabel('Cost');
    title(sprintf('Cost (eta=%0.3f)',GDparams.eta));
    grid;
    
    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('cost_standard_%d.pdf',experiment_no));

end

% Histograms
figure;
set(gcf, 'Position', get(0, 'Screensize'));

h = subplot(2,2,1);
bar(correct);
title('Correctly identified images per class');
xlabel('Class');
ylabel('Probability (%)');
grid;
set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
myAxes=findobj(h,'Type','Axes');
exportgraphics(myAxes,sprintf('hist_correct.pdf'));

h = subplot(2,2,2);
bar(incorrect);
title('Incorrectly identified images per class');
xlabel('Class');
ylabel('Probability (%)');
grid;
set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
myAxes=findobj(h,'Type','Axes');
exportgraphics(myAxes,sprintf('hist_incorrect.pdf'));



% fprintf("Accuracy on Test Batch : %0.2f %\n",accuracy_test);


%% Show templates
a = figure;
s_im{1,10} = [];
for experiment_no=1:n
    W = W_history(:,:,experiment_no);
    for i=1:10
        im = reshape(W(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
        imagesc(cell2mat(s_im(i)));
        grid off;
        axis off;
        title(label_names(i));
        ax = gca; 
        ax.FontSize = 18;
        myAxes=findobj(a,'Type','Axes');
        exportgraphics(myAxes,sprintf('weight_exp_%i_image%d.pdf',experiment_no,i));
    end
end
% montage(s_im, 'Size', [2,5]);

