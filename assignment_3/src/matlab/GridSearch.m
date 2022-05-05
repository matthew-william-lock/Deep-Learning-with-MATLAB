% grid_n = 3;
% 
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

function parameters = GridSearch(lambda,p,cycles,anneal)

    values = {lambda;p;cycles;anneal};
    parameters = [];
    
    for i = 1:size(values,1)-1
        p1 = cell2mat(values(i));
        p2 = cell2mat(values(i+1));
    
        p_1 = repmat(p1,size(p2,2),1);
        p_1 = reshape(p_1,[],1);
    
        p_2 = reshape(p2,[],1);
        if i ==1
            p_2 = repmat(p_2,size(p1,2),1);
            p_n = [p_1,p_2];
        end
    
        
        
        if i >1
            temp = [];
            for j = 1:size(parameters,1)
                temp = [temp;repmat(parameters(j,:),size(p_2,1),1)]; 
            end
            parameters = temp;
            p_n = repmat(p_2,size(parameters,1)/length(p_2),1);
        end    
        if i ==1
            parameters(:,[end+1,end+2])=p_n;
        else
            parameters = [parameters,p_n];
        end
        %     parameters(:,end+1,:)=p_2;
    
    end

end




