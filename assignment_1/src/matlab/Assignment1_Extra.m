%% Parameters
% n_batch       the size of the mini-batch
% eta           the learning rate
% n_epcohs      the number of runs through the whole training set
% n_epcohs      the number of runs through the whole training set
% lambda        normalisation factor
% shuffle       shuffle the minibatches after each epoch (not implemented)

%% General setup

%% Use all available data
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
X_train = X(:,1:end-1000); Y_train = Y(:,1:end-1000);
X_validate = X(:,end-1000+1:end); Y_validate = Y(:,end-1000+1:end);

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
lambda_list = [0,0,0.1,1];
n_epochs_list = [40,40,40,40];
n_batch_list = [100,100,100,100];
eta_list = [0.1,0.001,0.001,0.001];

% Setup matrices for datastorage
loss_train = zeros(length(eta_list),1,40);
loss_validate = zeros(length(eta_list),1,40);
accuracy_train = zeros(length(eta_list),1,40);
accuracy_validate = zeros(length(eta_list),1,40);
cost_train = zeros(length(eta_list),1,40);
cost_validate = zeros(length(eta_list),1,40);
accuracy_test = zeros(1,length(eta_list));

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
        [W,b] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i) = ComputeCost(X_train,Y_train, W,b,lambda);
        loss_validate(experiment_no,i) = ComputeCost(X_validate,Y_validate, W,b,lambda);
        accuracy_train(experiment_no,i) = ComputeAccuracy(X_train, Y_train, W, b);
        accuracy_validate(experiment_no,i) = ComputeAccuracy(X_validate, Y_validate, W, b);
        cost_train(experiment_no,i) = ComputeCost(X_train, Y_train, W, b,0);
        cost_validate(experiment_no,i) = ComputeCost(X_validate, Y_validate, W, b,0);
    
        fprintf("Experiment %d - Epoch %d completed\n",experiment_no,i);
    end

    accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;

end

%% Data augmentation (flipping the image)
clear;

% Load all data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validate, Y_validate, y_tvalidate] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
label_names = LoadLabelNames('batches.meta.mat');

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

% Determine which images to flip
images_to_flip = rand(1,size(X_train,2));
images_to_flip = images_to_flip > 0.5;

% Flip images
for i =1:size(X_train,2)
    if images_to_flip(i)
        X_train(:,i) = FlipImage(X_train(:,i));
    end
end
fprintf("Images Flipped \n");

% Generate results
fprintf("Initalising network \n");

% Setup paramater list
lambda_list = [0,0.1,0.2,0.3];
n_epochs_list = [40,40,40,40];
n_batch_list = [100,100,100,100];
eta_list = [0.1,0.001,0.001,0.001];

% Setup matrices for datastorage
loss_train = zeros(length(eta_list),1,40);
loss_validate = zeros(length(eta_list),1,40);
accuracy_train = zeros(length(eta_list),1,40);
accuracy_validate = zeros(length(eta_list),1,40);
cost_train = zeros(length(eta_list),1,40);
cost_validate = zeros(length(eta_list),1,40);
accuracy_test = zeros(1,length(eta_list));

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
        [W,b] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i) = ComputeCost(X_train,Y_train, W,b,lambda);
        loss_validate(experiment_no,i) = ComputeCost(X_validate,Y_validate, W,b,lambda);
        accuracy_train(experiment_no,i) = ComputeAccuracy(X_train, Y_train, W, b);
        accuracy_validate(experiment_no,i) = ComputeAccuracy(X_validate, Y_validate, W, b);
        cost_train(experiment_no,i) = ComputeCost(X_train, Y_train, W, b,0);
        cost_validate(experiment_no,i) = ComputeCost(X_validate, Y_validate, W, b,0);
    
        fprintf("Experiment %d - Epoch %d completed\n",experiment_no,i);
    end

    accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;

end
%% Dynamic learning rate
clear;

% Load all data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validate, Y_validate, y_tvalidate] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
label_names = LoadLabelNames('batches.meta.mat');

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
n_list = [1,2,5,10];

% Setup matrices for datastorage
loss_train = zeros(length(n_list),1,40);
loss_validate = zeros(length(n_list),1,40);
accuracy_train = zeros(length(n_list),1,40);
accuracy_validate = zeros(length(n_list),1,40);
cost_train = zeros(length(n_list),1,40);
cost_validate = zeros(length(n_list),1,40);
accuracy_test = zeros(1,length(n_list));
eta_hist = zeros(length(n_list),1,40);

% Loop through experiments
for experiment_no=1:length(n_list)

    % Init seed
    rng(400);

    % Init W and B
    W = normrnd(0,0.01,[size(Y_train,1),size(X_test,1)]);
    b = normrnd(0,0.01,[size(Y_train,1),1]);

    % Set paramts
    n_batch = 100;
    n_epochs = 40;
    lambda=0.5;
    eta = 0.005;

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;

    % Epochs
    for i=1:GDparams.n_epochs

        eta_hist(experiment_no,i) = eta;
        
        % Minibatch GD
        [W,b] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda);
    
        % Compute loss at the end of the epoch
        loss_train(experiment_no,i) = ComputeCost(X_train,Y_train, W,b,lambda);
        loss_validate(experiment_no,i) = ComputeCost(X_validate,Y_validate, W,b,lambda);
        accuracy_train(experiment_no,i) = ComputeAccuracy(X_train, Y_train, W, b);
        accuracy_validate(experiment_no,i) = ComputeAccuracy(X_validate, Y_validate, W, b);
%         cost_train(experiment_no,i) = ComputeCost(X_train, Y_train, W, b,0);
%         cost_validate(experiment_no,i) = ComputeCost(X_validate, Y_validate, W, b,0);

        % decrease learnign rate
        if mod(i,n_list(experiment_no))==0
            eta = eta*0.9;
            GDparams.eta = eta;
        end
    
        fprintf("Experiment %d - Epoch %d completed\n",experiment_no,i);
    end

    accuracy_test(experiment_no) = ComputeAccuracy(X_test, Y_test, W, b)*100;
    
end

%% Show results

n = length(n_list);

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:n

    h = subplot(n/2,n/2,experiment_no);

    % Set paramts
    nth = n_list(experiment_no);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;

    % Loss graph
    plot(eta_hist(experiment_no,:));
    xlabel('Epoch');
    ylabel('Training rate (eta)');
    title(sprintf('Training rate (n=%d)',nth));
    
    grid;
    
    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('eta_%d.pdf',experiment_no));
   
end

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:n

    h = subplot(n/2,n/2,experiment_no);

    % Set paramts
    nth = n_list(experiment_no);

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
    title(sprintf('Accuracy (lambda = %0.3f, n=%d)',lambda,nth));
    grid;
    
    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('accuracy_%d.pdf',experiment_no));

end

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:n

    h = subplot(n/2,n/2,experiment_no);

    % Set paramts
    nth = n_list(experiment_no);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    GDparams.eta = eta;
    
    % Accuracy graph
    plot(loss_train(experiment_no,:));
    hold on;
    plot(loss_validate(experiment_no,:));
    legend('Training set','Validation set');
    xlabel('Epoch');
    ylabel('Loss');
    title(sprintf('Loss (lambda = %0.3f, n=%d)',lambda,nth));
    grid;
    
    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('loss_%d.pdf',experiment_no));

end
% 
% % Loop through experiments
% figure;
% set(gcf, 'Position', get(0, 'Screensize'));
% for experiment_no=1:n
% 
%     h = subplot(n/2,n/2,experiment_no);
% 
%     % Set paramts
%     n_batch = n_batch_list(experiment_no);
%     n_epochs = n_epochs_list(experiment_no);
%     lambda=lambda_list(experiment_no);
%     eta = eta_list(experiment_no);
% 
%     % Create GDparams;
%     GDparams.n_batch = n_batch;
%     GDparams.n_epochs = n_epochs;
%     GDparams.eta = eta;
%     
%     % Accuracy graph
%     plot(cost_train(experiment_no,:));
%     hold on;
%     plot(cost_validate(experiment_no,:));
%     legend('Training set','Validation set');
%     xlabel('Epoch');
%     ylabel('Cost');
%     title(sprintf('Cost (eta=%0.3f)',GDparams.eta));
%     grid;
%     
%     set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
%     myAxes=findobj(h,'Type','Axes');
%     exportgraphics(myAxes,sprintf('cost_%d.pdf',experiment_no));
% 
% end

%% Show results

n = length(eta_list);

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:n

    h = subplot(n/2,n/2,experiment_no);

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
    exportgraphics(myAxes,sprintf('loss_%d.pdf',experiment_no));
   
end

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:n

    h = subplot(n/2,n/2,experiment_no);

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
    exportgraphics(myAxes,sprintf('accuracy_%d.pdf',experiment_no));

end

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:n

    h = subplot(n/2,n/2,experiment_no);

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
    exportgraphics(myAxes,sprintf('cost_%d.pdf',experiment_no));

end


% fprintf("Accuracy on Test Batch : %0.2f %\n",accuracy_test);