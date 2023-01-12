function [time,best_idv1] = ANN_PSO(i,ii)

%Define Global Varibales
global data y iter solution fitModel fitlist best_idv
%Read DATA
data = xlsread('..\DATA.xlsx',...
    'DATA', 'A2:j297');
% Create Save Dir.
currDate = strrep(datestr(datetime), ':', '_');
mkdir([pwd,'\ANN+PSO-k-fold\'],[currDate '_ANN-PSO_',num2str(i)]);


direc = [pwd,'\ANN+PSO-k-fold\',currDate '_ANN-PSO_',num2str(i)];

%% Pre-processing
 % Z-score
[Z,DataMean,DataStd] = zscore(data); %Z-score pre-processing
%Z-score Trim
Ztrim = find(abs(Z)>3);
Z(Ztrim) = NaN;
Z(any(isnan(Z), 2), :) = []; %Trim outliers

% Input-Output
data=Z(:,1:9); %Input AD varibale
y =Z(:,10); %Output Biogas Produced

%% Collect GA output Models
solution ={};
iter=1;
fitModel ={};
fitlist=[];
best_idv =[];

%% PSO setup
SwarmSize = i;
MaxIterations = ii;
options = optimoptions(@particleswarm,'SwarmSize',SwarmSize,...
                     'MaxIterations',MaxIterations,...
                     'PlotFcns',{@pswplotbestf}, ...
                     'OutputFcns',@pso_save_each_gen,...  
                     'Display', 'iter'); 
                 %'CrossoverFcn', {@crossoverarithmetic,0.8},...

nVars = 13; %GenomeLength; % This is the number of features in the dataset 
Fcn = @FitFunc_ANN; 
% Boundray condition
lb = [0 0 0 0 0 0 0 0 0 1 1 1 1];
ub = [1 1 1 1 1 1 1 1 1 100 30 8 3];
tic
[chromosome,fval] = particleswarm(Fcn,nVars,lb,ub,options);
time = toc
Feat_Index = find(chromosome(1:9)==1); % Index of Chromosome
Best_chromosome = [Feat_Index,chromosome(10:13)]; % Best Chromosome


[f1,idx]=sort(fitlist);
Bestnet = fitModel(idx(1));
best_idv1 =best_idv;
%% Work with best net
Bnet = Bestnet{1,1}{1,1}{1,1};
Btr =Bestnet{1,1}{1,1}{1,2};
Bx =Bestnet{1,1}{1,1}{1,3};
By = Bnet(Bx);

% %train data
trainX = Bestnet{1,1}{1,1}{1,3};
trainT = Bestnet{1,1}{1,1}{1,5};
trainY = Bnet(trainX);
%test data
testX = Bestnet{1,1}{1,1}{1,4};
testT = Bestnet{1,1}{1,1}{1,6};
testY = Bnet(testX);

%best input
Selected_input = Bestnet{1, 1}{1,2}

BTrainingFcn = Bnet.trainFcn
BActiveFcn = Bnet.layers{1}.transferFcn
BHiddenlayerSize = Bnet.layers{1}.dimensions
Bepochs = Bnet.trainParam.epochs
KfoldFval = fval


%Plot Regression
figure;
rtr=corrcoef(trainT,trainY);
r2tr=rtr(1,2)^2;

rte=corrcoef(testT,testY);
r2te=rte(1,2)^2;

plotregression(trainT,trainY,['Train Data, R^2= ' ,num2str(r2tr),','], ...
    testT,testY,['Test Data, R^2= ' ,num2str(r2te)]);

saveas (gcf,[direc,'\Reg_plot.m']);
close (gcf)

% Plot Fit
allT=[trainT,testT];
allY=[trainY,testY];

Rall = corrcoef(allT,allY);
RMSE = sqrt(mse(allT,allY));

figure;
rall=corrcoef(allT,allY);
r2all=rall(1,2)^2;

plotregression(allT,allY,['ALL Data, R^2= ' ,num2str(r2all),','])
a = get(gca,'ylabel');
eq = a.String;
xlabel(['Target',newline,newline,eq],"FontWeight","bold")
ylabel('Predicted',"FontWeight","bold")
str=['Model: ANN-PSO',newline,'Dataset: All Data',newline,'R = ',...
    num2str(Rall(1,2),3),newline,'RMSE = ',num2str(RMSE,3)];
annotation('textbox',[.65 .25 .1 .1],String=str)

saveas (gcf,[direc,'\Regall_plot.m']);



% Reverse Z-score
allT= allT.*DataStd(:,10)+DataMean(:,10);
allY = allY.*DataStd(:,10)+DataMean(:,10);

PlotFit(allT,allY,testT)

saveas (gcf,[direc,'\Fit_Plot.m']);


% Revised Reg plot

allT=[trainT,testT];
allY=[trainY,testY];

Rall = corrcoef(allT,allY);
RMSE = sqrt(mse(allT,allY));

figure;
rall=corrcoef(allT,allY);
r2all=rall(1,2)^2;

allT= allT.*DataStd(:,10)+DataMean(:,10);
allY = allY.*DataStd(:,10)+DataMean(:,10);
allT_act = allT/24;
allY_act = allY/24;


plotregression(allT_act,allY_act,['ALL Data, R^2= ' ,num2str(r2all),','])
a = get(gca,'ylabel');
eq = a.String;
xlabel(['Actual Biogas Yield (m^3/hr)',newline,newline,eq],"FontWeight","bold")
ylabel('Predicted (m^3/hr)',"FontWeight","bold")
str=['Model: ANN-PSO',newline,'Dataset: All Data',newline,'R = ',...
    num2str(Rall(1,2),3),newline,'RMSE = ',num2str(RMSE,3)];
annotation('textbox',[.65 .25 .1 .1],String=str)



%% Save Project Output
save ([direc,'\final.m']);
clc;close all;
end
