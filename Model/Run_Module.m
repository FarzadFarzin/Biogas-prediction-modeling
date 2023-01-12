clc;clear;close all;

addpath([pwd,'\ANN+GA-k-fold\']);
addpath([pwd,'\ANN+PSO-k-fold\']);
addpath([pwd,'\SVR+GA-k-fold\']);
addpath([pwd,'\SVR+PSO-k-fold\']);

popsize = [20 50 100 150 200];

timeANNGA = zeros(1,length(popsize));
timeANNPSO = zeros(1,length(popsize));
timeSVRGA = zeros(1,length(popsize));
timeSVRPSO = zeros(1,length(popsize));

ii  = 30;
bestIdvANNGA = zeros(ii,length(popsize));
bestIdvANNPSO = zeros(ii,length(popsize));
bestIdvSVRGA = zeros(ii,length(popsize));
bestIdvSVRPSO = zeros(ii,length(popsize));
for i = 1:length(popsize)

    [time1,best_idv1] = ANN_GA(popsize(i),ii);
    timeANNGA(1,i) = time1;
    bestIdvANNGA(:,i) =best_idv1(:,2);

    [time2,best_idv2] = ANN_PSO(popsize(i),ii);
    timeANNPSO(1,i) = time2;
    bestIdvANNPSO(:,i) = best_idv2(:,2);

    [time3,best_idv3] = SVR_GA (popsize(i),ii);
    timeSVRPSO(1,i) = time3;
    bestIdvSVRGA(:,i) = best_idv3(:,2);

    [time4,best_idv4] = SVR_PSO (popsize(i),ii);
    timeSVRGA(1,i) = time4;
    bestIdvSVRPSO(:,i) = best_idv4(:,2);

end

%% plot output

data = {bestIdvANNGA,bestIdvANNPSO,bestIdvSVRGA,bestIdvSVRPSO};
title = {'ANN-GA','ANN-PSO','SVR-GA','SVR-PSO'};
for it = 1:length(data)
    plotpop(data{it},popsize,title{it})
end


bartitle =categorical(title);
ybar = [timeANNGA;timeANNPSO;timeSVRGA;timeSVRPSO];
figure
bar (bartitle,ybar)
ylabel('sec')
listofname ={};
for i = 1:length(popsize)
    listofname{i} = (['Population ',num2str(popsize(i))]);
end
legend(listofname)

save final.mat

function plotpop(data,popsize,name)
figure
plot (data)
ylabel('RMSE','FontWeight','bold')
xlabel('Generation','FontWeight','bold')
title (name)
listofname ={};
for i = 1:length(popsize)
    listofname{i} = (['Population ',num2str(popsize(i))]);
end
legend(listofname)
end






