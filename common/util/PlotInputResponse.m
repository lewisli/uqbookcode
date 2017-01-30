function [ h ] = PlotInputResponse(DataStruct,TruthRealization,FontSize,SavePath)
%PlotInputResponse: Plots input response for historical and forecasts
%
% Inputs:
%   HistoricalStruct: Struct containing historical data
%   ForecastStruct: Struct containing forecast data
%   TruthRealization: Index of realization taken to be the truth
%   FontSize: Font size in plots
%
% Author: Lewis Li (lewisli@stanford.edu)
% Original Date: March 4th 2016
% Last Updated: September 26th 2016

if (nargin < 3)
    FontSize=12;
end

if (nargin < 4)
    SaveOn = false;
else
    SaveOn = true;
end

NumRealizations = size(DataStruct.data,1);
NumResponses = size(DataStruct.data,3);

MaxCols = min(3,NumResponses);
NumRows = 1;

MaxFiguresPerPage = 3;

for i = 1:NumResponses
    if mod(i,MaxFiguresPerPage) == 1
<<<<<<< HEAD
        FigID =fix(i/MaxFiguresPerPage) + 1;
=======
        FigID =fix(i/MaxFiguresPerPage) + 1; 
>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b
        h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
        NumCols = min(NumResponses - MaxFiguresPerPage*(FigID-1),MaxCols);
    end
    
<<<<<<< HEAD
    subplot(NumRows,MaxCols,mod(i-1,MaxFiguresPerPage)+1);
    
    
    hold on; title([ DataStruct.type ': ' DataStruct.ObjNames{i}],...
        'FontSize',FontSize); axis square;
=======
    subplot(NumRows,MaxCols,mod(i-1,MaxFiguresPerPage)+1); 
    
    
    hold on; title([ DataStruct.type ': ' DataStruct.ObjNames{i}],...
         'FontSize',FontSize); axis square;
>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b
    
    for j = 1:NumRealizations
        h1 = plot(DataStruct.time, DataStruct.data(j,:,i)',...
            'color',[0.5 0.5 0.5]);
    end
    xlabel('Time (days)','FontSize',FontSize);
    ylabel(DataStruct.name,'FontSize',FontSize);
<<<<<<< HEAD
    set(gcf,'color','w');
    set(gca,'FontSize',FontSize);
    axis tight;
    
    hlegend = legend([h1],'Prior');
    set(hlegend,'Location','southwest');
=======
>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b
    
    if (TruthRealization>0)
        h2 = plot(DataStruct.time, ...
            DataStruct.data(TruthRealization,:,i)','r','LineWidth',3);
        hlegend = legend([h1,h2],'Prior','Observed');
<<<<<<< HEAD
        set(hlegend,'Location','southwest');
    end
    
    if (mod(i,MaxFiguresPerPage) == 0 || i == NumResponses)
        FigID =ceil(i/MaxFiguresPerPage);
        if SaveOn == true
            export_fig([SavePath 'Prior_' DataStruct.type '_' ...
                num2str(FigID)],'-png','-m3');
=======
        axis tight;
        set(gcf,'color','w');
        set(gca,'FontSize',FontSize);
        set(hlegend,'Location','northwest');
    end
    
    if (mod(i,MaxFiguresPerPage) == 0 || i == NumResponses)
        FigID =ceil(i/MaxFiguresPerPage); 
        if SaveOn == true
            export_fig([SavePath 'Prior_' DataStruct.type '_' num2str(FigID)],...
                '-png','-m3');
>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b
        end
        
    end
    
end

end

