%% Show results

update_steps_numbering = 0:(size(X_train,2)/n_batch):((length(nt_hist)-1)/450)*(size(X_train,2)/n_batch);
update_steps_numbering = update_steps_numbering / (2*n_s);

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:8

     h = subplot(4,2,experiment_no);

    % Set paramts
    n_batch = 100;
    n_epochs = epochs;
    lambda=lambda_list(experiment_no);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;

    % Loss graph
    plot(update_steps_numbering,loss_train(experiment_no,:));
    hold on;
    plot(update_steps_numbering,loss_validate(experiment_no,:));
    legend('Training set','Validation set');
    xlabel('Cycles');
    ylabel('Loss');
    title(sprintf('Loss (lambda = %0.2e)',lambda));
    grid;
    
    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('loss_standard_%d.pdf',experiment_no));
   
end

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
for experiment_no=1:8

     h = subplot(4,2,experiment_no);

    % Set paramts
    n_batch = 100;
    n_epochs = epochs;
    lambda=lambda_list(experiment_no);

    % Create GDparams;
    GDparams.n_batch = n_batch;
    GDparams.n_epochs = n_epochs;
    
    % Accuracy graph
    plot(update_steps_numbering,accuracy_train(experiment_no,:));
    hold on;
    plot(update_steps_numbering,accuracy_validate(experiment_no,:));
    legend('Training set','Validation set');
    xlabel('Epoch');
    ylabel('Accuracy (%)');
    title(sprintf('Accuracy (lambda = %0.2e)',lambda));
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