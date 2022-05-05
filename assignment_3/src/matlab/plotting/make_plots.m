%% Show results

n = length(lambda_list);

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:n

    h = subplot(m,m,experiment_no);

    % Set paramts
    n_epochs = epochs;
    lambda=lambda_list(experiment_no);
    eta = eta_list(experiment_no);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;

    % Loss graph
    plot(loss_train(experiment_no,:));
    hold on;
    plot(loss_validate(experiment_no,:));
    legend('Training set','Validation set');
    xlabel('Epoch');
    ylabel('Loss');
    title(sprintf('Loss (lambda = %0.2f, eta =  %0.2f)',lambda,eta));
    grid;
    
    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('loss_standard_%d.pdf',experiment_no));
   
end

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:n

    h = subplot(m,m,experiment_no);

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
%     exportgraphics(myAxes,sprintf('cost_standard_%d.pdf',experiment_no));
% 
% end
% 
% % Histograms
% figure;
% set(gcf, 'Position', get(0, 'Screensize'));
% 
% h = subplot(2,2,1);
% bar(correct);
% title('Correctly identified images per class');
% xlabel('Class');
% ylabel('Probability (%)');
% grid;
% set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
% myAxes=findobj(h,'Type','Axes');
% exportgraphics(myAxes,sprintf('hist_correct.pdf'));
% 
% h = subplot(2,2,2);
% bar(incorrect);
% title('Incorrectly identified images per class');
% xlabel('Class');
% ylabel('Probability (%)');
% grid;
% set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
% myAxes=findobj(h,'Type','Axes');
% exportgraphics(myAxes,sprintf('hist_incorrect.pdf'));