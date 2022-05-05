function [W,B,gammas,betas] = InitParametersGuassian(X_test,Y_train,d,intermediate_layers,sigma)

    % Number of nodes at each layer
    intermediate_layers = [intermediate_layers,size(Y_train,1)];

    % Determine number of layers
    layers = length(intermediate_layers);
    
    % Paramters
    W = cell(layers,1);
    B = cell(layers,1);
    gammas = cell(layers-1,1);
    betas = cell(layers-1,1);
    
    for i = 1: length(intermediate_layers)    
        % Account for first hidden layer
        if i==1
            W(i) = {normrnd(0,sigma,[intermediate_layers(i),size(X_test,1)])};
            B(i) = {zeros([intermediate_layers(i),1])};
        else
            W(i) = {normrnd(0,sigma,[intermediate_layers(i),intermediate_layers(i-1)])};
            B(i) = {zeros([intermediate_layers(i),1])};
        end    

        if i ~=layers
            gammas(i) = {ones([intermediate_layers(i),1])};
            betas(i) = {zeros([intermediate_layers(i),1])};
        end
    end

end

% OLD FUNCTION
%     W1 = normrnd(0,1/sqrt(d),[m,size(X_test,1)]);
%     b1 = zeros([m,1]);
%     W2 = normrnd(0,1/sqrt(m),[size(Y_train,1),m]);
%     b2 = zeros([size(Y_train,1),1]);

