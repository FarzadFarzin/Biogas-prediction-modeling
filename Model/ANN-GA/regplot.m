function regplot (t,y,r,rmse,modelname,datalabel)

figure;
format long
b1 = r;

yCalc1 = b1.*t;
scatter(t,y)
hold on
plot(t,yCalc1)
xlabel('Target (m^3/Day)')
ylabel('Predicted (m^3/Day)')
colormap autumn
grid on
dim =[.2 .6 .3 .3];
str= ['Model: ',modelname,'Dataset: ',datalabel,'R = ',r,'RMSE = ',rmse];
annotation("textbox",dim,String=str,FitBoxToText="on",EdgeColor='w')





end
