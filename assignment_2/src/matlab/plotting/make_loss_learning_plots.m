%% Show results

n = length(eta_list);
m = round(n/2);

% Loop through experiments
figure;
set(gcf, 'Position', get(0, 'Screensize'));
update_steps_numbering = 0:(size(X_train,2)/n_batch):(1+epochs)*(size(X_train,2)/n_batch);

for experiment_no=1:n

    h = subplot(2,2,1);

    % Set paramts
    %     n_batch = n_batch_list(experiment_no);
    %     n_epochs = n_epochs_list(experiment_no);
    %     lambda=lambda_list(experiment_no);
    %     eta = eta_list(experiment_no);
    % 
    %     % Create GDparams;
    %     GDparams.n_batch = n_batch;
    %     GDparams.n_epochs = n_epochs;
    %     GDparams.eta = eta;

    % Cost graph
    plot(update_steps_numbering,cost_train(experiment_no,:));
    hold on;
    plot(update_steps_numbering,cost_validate(experiment_no,:));
    legend('Training set','Validation set');
    xlabel('Training steps');
    ylabel('Cost');
    title(sprintf('Cost (cycles = %d)',cycles));
    grid;

    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('cost_%d.pdf',experiment_no));

    % Loss graph
    h = subplot(2,2,2);
    plot(update_steps_numbering,loss_train(experiment_no,:));
    hold on;
    plot(update_steps_numbering,loss_validate(experiment_no,:));
    legend('Training set','Validation set');
    xlabel('Training steps');
    ylabel('Loss');
    title(sprintf('Loss (cycles = %d)',cycles));
    grid;

    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('loss_%d.pdf',experiment_no));

    % Accuracy graph
    h = subplot(2,2,3);
    plot(update_steps_numbering,accuracy_train(experiment_no,:));
    hold on;
    plot(update_steps_numbering,accuracy_validate(experiment_no,:));
    legend('Training set','Validation set');
    xlabel('Training steps');
    ylabel('Accuracy (%)');
    title(sprintf('Accuracy (cycles = %d)',cycles));
    grid;

    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('accuracy_%d.pdf',experiment_no));

    % Learning rate
    h = subplot(2,2,4);
    plot(nt_hist);
    xlim([0 length(nt_hist)-1])
    xlabel('Training steps');
    ylabel('Learning rate (\eta)');
    title(sprintf('Learning rate (cycles = %d)',cycles));
    grid;
    
    set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',18);
    myAxes=findobj(h,'Type','Axes');
    exportgraphics(myAxes,sprintf('learning_Rate__%d.pdf',experiment_no));
   
end
