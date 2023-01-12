function PlotFit(t,y,testdata)
    trNum = length(t)-length(testdata);
    font = 12;
    markercolor = '#FF5A44';

    markercolor1 = '#687EBA';


    figure();
    plot(y,'-s','Color',markercolor,'MarkerSize',6,...
    'MarkerEdgeColor',markercolor);
    
    hold on;
    plot(t,'-o','Color',markercolor1,'MarkerSize',6,...
    'MarkerEdgeColor',markercolor1);

    A= [zeros(1,trNum)+trNum;linspace(0,max(y)*1.15,trNum)]';
    hold on
    plot(A(:,1),A(:,2),'k-','LineWidth',2)
    ylim ([0 max(y)*1.15])
    xlim ([0 length(y)*1.05])
    hold off
    legend ('Predicted', 'Measured','Location','southwest','FontSize',font);
    xlabel('Number of Data')
    ylabel('Biogas Flow (m^3 /Day)')

    dim = [.2 .6 .3 .3];
    str = ' Training';
    annotation('textbox',dim,'String',str,'FitBoxToText','on',...
        'FontSize',14,'EdgeColor','none');
    dim = [.8 .6 .3 .3];
    str = ' Testing';
    annotation('textbox',dim,'String',str,'FitBoxToText','on',...
        'FontSize',14,'EdgeColor','none');
end