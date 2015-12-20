addpath(genpath('/n/regal/TSC/Ensemble_method3/randomforest-matlab/'))
addpath(genpath('/n/regal/TSC/Ensemble_method3/bnt/'))

display('hola')

load('OUTL_ENSEMBLE') %('OUTL_YS_10NoKmeans2.mat')
    
bins=OUTL.bins;
N_classes=size(OUTL.CPT3,2);

% data =  vertcat(training_set, X_tst);
N_data = size(data,1);

[Y_hat, votes_test, prediction_pre_tree] = classRF_predict(data,OUTL.model,OUTL.test_options);


prob_data=zeros(N_data, N_classes);

for i=1:N_data
    for j=1:N_classes
        prob_data(i,j)=votes_test(i,j)/sum(votes_test(i,:));
    end
end


range=1/bins; range1=range;

for i=1:N_data
    for j=1:N_classes
        if prob_data(i,j)<range
            prob_data(i,j)=-1;
        end
    end
end

for k=2:bins-1
    for i=1:N_data
        for j=1:N_classes
            if prob_data(i,j)<range1+range && prob_data(i,j)>=range1
                prob_data(i,j)=-k;
            end
        end
    end
    range1=range1+range;
end

for i=1:N_data
    for j=1:N_classes
        if prob_data(i,j)>=range1 && prob_data(i,j)<=1 
            prob_data(i,j)=-bins;
        end
    end
end


clear i j k 

prob_data=-prob_data';

factors=zeros(N_data,N_classes);

for k=1:N_data
    
    for i=1:N_classes
        
        idx=[];
        parent_node=parents(OUTL.bnet2.dag, i);
        N_parents=size(parent_node);
        
        for j=1:N_parents(2)
            idx=vertcat(idx,prob_data(parent_node(j),k));
        end
        idx=idx';
        
        if N_parents(2)==1
            
            factors(k,i)=OUTL.CPT3{i}(idx(1),prob_data(i,k));
                       
            
        elseif N_parents(2)==2
            
            factors(k,i)=OUTL.CPT3{i}(idx(1),idx(2),prob_data(i,k));
            
            
        elseif N_parents(2)==3
            
            factors(k,i)=OUTL.CPT3{i}(idx(1),idx(2),idx(3),prob_data(i,k));
        elseif N_parents(2)==4
            
            factors(k,i)=OUTL.CPT3{i}(idx(1),idx(2),idx(3),idx(4),prob_data(i,k));
        
        elseif N_parents(2)==5
            
            factors(k,i)=OUTL.CPT3{i}(idx(1),idx(2),idx(3),idx(4),idx(5),prob_data(i,k));
        
        elseif N_parents(2)==6
            
            factors(k,i)=OUTL.CPT3{i}(idx(1),idx(2),idx(3),idx(4),idx(5),idx(6),prob_data(i,k));
        
        elseif N_parents(2)==7
            
            factors(k,i)=OUTL.CPT3{i}(idx(1),idx(2),idx(3),idx(4),idx(5),idx(6),idx(7),prob_data(i,k));
        
        elseif N_parents(2)==8
            
            factors(k,i)=OUTL.CPT3{i}(idx(1),idx(2),idx(3),idx(4),idx(5),idx(6),idx(7),idx(8),prob_data(i,k));    
               
            
        elseif N_parents(2)==0
            
            factors(k,i)=OUTL.CPT3{i}(prob_data(i,k));
        end
        
    end
end

joint=zeros(N_data,1); 

for i=1:N_data
    for j=1:N_classes
        if factors(i,j)~=0
            joint(i,1)=joint(i,1)+log(factors(i,j));
            %            conjunta(i)=log(factores(i,2))+log(factores(i,6))+log(factores(i,1));
        elseif factors(i,j)==0
            disp('wrong')
            joint(i)=joint(i)+log(min(nonzeros(min(factors))));
            
        end
        %         suma(i)=sum(factores(i,:));
        %         fminimo(i)=min(factores(i,:));
    end
end


% joint_ts = joint(1:size(training_set,1),1);
% joint_test = joint(size(training_set,1)+1:end,1);
