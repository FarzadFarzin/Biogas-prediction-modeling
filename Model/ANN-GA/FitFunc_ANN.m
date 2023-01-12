%% Fitness function
function [FitVal] = FitFunc_ANN(pop)
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
hidden_layersize=round(pop(11));
epochs=round(pop(10));
trainopts= {'trainscg','trainbfg','trainrp','traincgb','traincgf','traincgp','traingdx','trainoss'};
opt = round(pop(12));
transferopts= {'tansig','logsig','poslin','radbas'};
opt2=round(pop(13));

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
     xTrain = xTrain';
     xTest = xTest';
     yTrain = yTrain';
     yTest = yTest';

     %% Assignment of Hyperparameters & built model

     hiddenLayerSize = hidden_layersize ; %Hyperparameter1; hidden Layer Size
     trainFcn = trainopts{opt}; %Hyperparameter3; Train Fcn
     % Create net 
     net = fitnet(hiddenLayerSize,trainFcn);
     % Set Test and Validation to zero in the input data
     net.divideParam.trainRatio = 1;
     net.divideParam.testRatio = 0;
     net.divideParam.valRatio = 0;
     % Set model Parameters
     net.trainParam.epochs=epochs; %Hyperparameter2; Epochs number
     net.layers{1}.transferFcn = transferopts{opt2}; %Hyperparameter4; Transfer Fcn
     net.trainParam.showWindow = false;
     %% train network
     [net, tr] = train(net,xTrain,yTrain);
     yPred = net(xTest);
     perf = perform(net,yTest,yPred);
     perf = sqrt(perf);
%      disp(perf);
     
     %store results     
     netAry{i} = {net tr xTrain xTest yTrain yTest};
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