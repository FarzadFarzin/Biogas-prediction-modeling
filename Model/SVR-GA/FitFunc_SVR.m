%% Fitness function
function [FitVal] = FitFunc_SVR(pop)
global data y iter solution fitModel fitlist

solution{iter} = round(pop);

FeatIndex = find(round(pop(1:9))==1); 
s = size(FeatIndex);
if s(2) ==0
    FeatIndex=randi([1 9]);
end

Mytitle = {'VS-in1','VS-in2','VS-out','Ds-in','DS-out','pH','T','ALK',...
    'FA','Biogas'};

SelectedTitle = Mytitle(FeatIndex);

%% Input feature selection
x = data(:,FeatIndex);

%% Hyper Parameters of the Objective Function
C=pop(10);
e = pop(11);
sigma =pop(12); %gussion parameter % 'auto'
kernel= 'rbf';

%% K-fold CV 

numFolds = 5;
c = cvpartition(length(y),'k',numFolds);

% table to store the results 
netAry = {numFolds,1};
perfAry = zeros(numFolds,1);


for i = 1:numFolds
    
    %get Train and Test data for this fold
     trIdx = c.training(i);
     teIdx = c.test(i);
     xTrain = x(trIdx,:);
     yTrain = y(trIdx,:);
     xTest = x(teIdx,:);
     yTest = y(teIdx,:);
     
     %transform data to columns as expected by neural nets
%      xTrain = xTrain';
%      xTest = xTest';
%      yTrain = yTrain';
%      yTest = yTest';
     %% train Model
     Mdl = fitrsvm(xTrain,yTrain,'KernelFunction',kernel,...
     'KernelScale',sigma,...
     'Solver','L1QP',...
     'Epsilon',e,'BoxConstraint',C,...
     'Verbose',0,'NumPrint',100); % (,'KFold',5) ('polynomial','PolynomialOrder',5)

     %% Test Model
     yPred = predict(Mdl,xTest);
     perf = mse(yTest,yPred);
     perf = sqrt(perf);
%      disp(perf);
     
     %store results     
     netAry{i} = {Mdl xTrain xTest yTrain yTest};
     perfAry(i) = perf;
     
end


%take the network with min Loss value
[minKfold,minPerfId] = min(perfAry);
bestNet = netAry{minPerfId};

% CV MSE
FitVal = sum(perfAry)/numFolds; % CV of k-fold trained model

% Send Model Info Out of GA
fitlist(iter)=FitVal;
fitModel{iter} = {bestNet,SelectedTitle,FitVal, minKfold};

iter = iter +1;
end



