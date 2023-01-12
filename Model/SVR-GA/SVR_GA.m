function [time,best_idv3] = SVR_GA(i,ii)

%Define Global Varibales
global data y iter solution fitModel fitlist best_idv
%Read DATA
data = xlsread('..\DATA.xlsx',...
    'DATA', 'A2:j295');
% Create Save Dir.
currDate = strrep(datestr(datetime), ':', '_');
mkdir([pwd,'\SVR+GA-k-fold\'],[currDate '_SVR-GA_',num2str(i)]);

direc = [pwd,'\SVR+GA-k-fold\',currDate '_SVR-GA_',num2str(i)];

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
%% GA setup
popsize = i;
Gensize = ii;
options = optimoptions(@ga,'PopulationSize',popsize,...
                     'Generations',Gensize,...
                     'SelectionFcn',@selectionroulette,...
                     'EliteCount',ceil(0.05*popsize),...
                     'MutationFcn', @mutationadaptfeasible,...
                     'CrossoverFcn',@crossoverscattered,...
                     'StallGenLimit',100,...
                     'PlotFcns',{@gaplotbestf},...  
                     'OutputFcns',@ga_save_each_gen,... 
                     'Display', 'iter'); 

nVars = 12; %GenomeLength; % This is the number of features in the dataset 
Fcn = @FitFunc_SVR; 
% Boundray condition
lb = [0 0 0 0 0 0 0 0 0 0.01 0.001 1e-6 ];
ub = [1 1 1 1 1 1 1 1 1 100 5 8 ];
tic
[chromosome,fval,~,~,~,~] = ga(Fcn,nVars,[],[],[],[],lb,ub,[],options);
time = toc
Best_chromosome = [round(chromosome(1:9)),chromosome(10:12)]; % Best Chromosome
Feat_Index = find(Best_chromosome(1:9)==1); % Index of Chromosome
best_idv3= best_idv;
%% Find best Solution From GA optimization 
lastGen = fitlist(:,length(fitlist)-popsize+1:length(fitlist));
lastGenMdl = fitModel(:,length(fitModel)-popsize+1:length(fitModel));

[f1,idx]=sort(lastGen);
Bestmdl = lastGenMdl(idx(1));

%% Work with best net
Bmdl = Bestmdl{1,1}{1,1}{1,1};
Bx =Bestmdl{1,1}{1,1}{1,2};
By = predict(Bmdl,Bx);

% %train data
trainX = Bestmdl{1,1}{1,1}{1,2};
trainT = Bestmdl{1,1}{1,1}{1,4};
trainY = predict(Bmdl,trainX);
%test data
testX = Bestmdl{1,1}{1,1}{1,3};
testT = Bestmdl{1,1}{1,1}{1,5};
testY = predict(Bmdl,testX);

selectedinput = Bestmdl{1,1}{1,2}
Ep = Bmdl.Epsilon
gamma = Bmdl.KernelParameters
C = Bmdl.BoxConstraints(1)
KfoldFval = fval

%Plot Regression
figure;
rtr=corrcoef(trainT,trainY);
r2tr=rtr(1,2)^2;

rte=corrcoef(testT,testY);
r2te=rte(1,2)^2;

plotregression(trainT,trainY,['Test Data, R^2= ' ,num2str(r2tr),','], ...
    testT,testY,['Test Data, R^2= ' ,num2str(r2te)]);

saveas (gcf,[direc,'\Reg_plot.m']);
close (gcf)

% Plot Fit
allT=[trainT',testT'];
allY=[trainY',testY'];

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
str=['Model: SVR-GA',newline,'Dataset: All Data',newline,'R = ',...
    num2str(Rall(1,2),3),newline,'RMSE = ',num2str(RMSE,3)];
annotation('textbox',[.65 .25 .1 .1],String=str)

saveas (gcf,[direc,'\Regall_plot.m']);

% Reverse Z-score
allT= allT.*DataStd(:,10)+DataMean(:,10);
allY = allY.*DataStd(:,10)+DataMean(:,10);

PlotFit(allT,allY,testT)

saveas (gcf,[direc,'\Fit_Plot.m']);


% Revised Reg plot (Actual Value)

allT=[trainT',testT'];
allY=[trainY',testY'];

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
str=['Model: SVR-GA',newline,'Dataset: All Data',newline,'R = ',...
    num2str(Rall(1,2),3),newline,'RMSE = ',num2str(RMSE,3)];
annotation('textbox',[.65 .25 .1 .1],String=str)

%% Save Project Output
save ([direc,'\final.m']);
clc;close all;
end
