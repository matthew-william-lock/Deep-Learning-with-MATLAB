%% Assignmnet
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
W = normrnd(0,0.01,[size(Y_train,1),size(X_test,1)]);

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
    W_history(:,:,experiment_no) = W;
    [correct, incorrect] = ComputeAccuracyHisogram(X_test,Y_test,W,b);

end



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
    exportgraphics(myAxes,sprintf('loss_standard_%d.pdf',experiment_no));
   
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
    exportgraphics(myAxes,sprintf('accuracy_standard_%d.pdf',experiment_no));

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

function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
    % Create minibatches and do learning
    for j=1:size(X,2)/GDparams.n_batch  
        j_start = (j-1)*GDparams.n_batch + 1;
        j_end = j*GDparams.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        P = EvaluateClassifier(Xbatch, W,b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
        W = W - GDparams.eta*grad_W;
        b = b - GDparams.eta*grad_b;
    end

    Wstar = W;
    bstar = b;
end

function P = EvaluateClassifier(X, W, b)
    intermediate = W*X + b;
    P = exp(intermediate)./sum(exp(intermediate));
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)

% grad W is gradient matrix of the cost J relative to W and has size K×d.
% grad b is gradient vector of the cost J relative to b and has size K×1.
% J => cost

    % Determine Pbatch and Gbatch
    Pbatch =  P;
    Gbatch = -(Y-Pbatch);
    
    % Get number of examples
    nb = size(X,2);
    
    % Determine gradients
    dLdW = (1/nb)* Gbatch * X';
    dLdB = (1/nb)*Gbatch*ones(size(X,2),1);
    
    % return gradients
    grad_W = dLdW + 2*lambda*W;
    grad_b = dLdB;

end

function J = ComputeCost(X, Y, W, b, lambda)   
    % Get probabilities
    P = EvaluateClassifier(X, W, b);
    
    % Calculate cost (same as taking loss on the diagonal?)
    % For single input, i.e. size(Y) = 10,1 => same as cost = Y'*-log(P)
    cost = sum(-log(P(Y==1))) / size(P,2);

    % Calculate regularisation cost
    reg = lambda* norm(W,"fro")^2;

    J = cost + reg;
end

function acc = ComputeAccuracy(X, y, W, b)

    % Get probabilities
    P = EvaluateClassifier(X, W, b);
    
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

function [correct,incorrect] = ComputeAccuracyHisogram(X, y, W, b)

    % Get probabilities
    P = EvaluateClassifier(X, W, b);
    [max_values,~] = max (P,[],1);
    output = double((P'==max_values')');
    correct_matrix = output.*y;
    
    % Get number of classes
    C = size(P,1);
    correct = zeros(1,C);
    incorrect = zeros(1,C);

    for i = 1:C
        
        correct_in_class = sum(correct_matrix(i,:));
        true_correct_in_class = sum(y(i,:));
        correct(i) = correct_in_class/true_correct_in_class*100;
        incorrect(i) = 100 - correct(i);

%         correct(i) = correct_in_class/true_correct_in_class*100;
%         incorrect(i) = 100-correct(i);
    end

end

