% OilCase1.m
%
% Generate figures for Chapter 9 Oil Case 1: Forecasting Injector
% Efficiencies
%
% Author: Lewis Li (lewisli@stanford.edu)
% Original Date: December 24th 2016
% Last Modified: Janurary 18th 2017
%
% ToDo: Save HistoricalStruct and load that instead of parsing raw 3DSL
% outputs

%% Paths
case_name = 'OilCase1';
input_data_path = 'C:\Users\Lewis Li\ResearchData\LibyanCase\';
figure_output_path = 'C:\Users\Lewis Li\Documents\Github\uqbookcode\Chapter9\figures\';

addpath('../../common/cfca');
addpath('../../common/util');
addpath('../../common/export_fig');
addpath('../../common/fda_matlab/');
addpath('../../common/likelihood_continuous/');
% Create folder for saving figures
save_path = [figure_output_path case_name '\'];
if ~exist(save_path, 'dir')
    mkdir(save_path)
end

fontsize = 22;



%% 1a) Load Production Profile (Data Variable)
truth_realization = 12;
load([input_data_path 'Situation1\results\SimulationResults.mat']);

ForecastColumn = 4;
HistoricalColumn = 4;

% This is the time step that we divide the forecast and history
TimeColumn = 2;
NumTimeSteps = 400;
EndTime = 4000;
ForecastObjName={'P5'}; % Not used
HistoricalObjName = {'P1','P2','P3','P4','P5'};
ForecastSpline = [4 20];
HistoricalSpline = [4 20];
ForecastStart = 160;
HistoricalEnd = 80;

[HistoricalStruct, ~] = GenerateDataStructsWithInterpolation(Data, ...
    PropertyNames, ForecastColumn, HistoricalColumn, TimeColumn, ...
    HistoricalEnd,ForecastStart, NumTimeSteps, ForecastSpline, ...
    HistoricalSpline, ForecastObjName, HistoricalObjName, EndTime);

% Plot Production Profiles
h1  = PlotInputResponse(HistoricalStruct,truth_realization,...
    fontsize);

%% 1b) Load in Injector Efficiency (Prediction Variable)
num_realizations=length(HistoricalStruct.RunNames);

% Name of injectors who's efficiencies we want to load
inj_name = {'I1','I2','I3'};
num_injectors = length(inj_name);

% Path to Efficiencies
inj_dir = [input_data_path 'Situation1\results\waf\'];
forecast_var = zeros(num_realizations,num_injectors);

h = waitbar(0,'Loading efficiencies...');
for i = 1:num_realizations-1
    index = HistoricalStruct.RunNames(i);
    run_name = ['Run' num2str(index)];
    forecast_struct = load([inj_dir run_name '.mat']);
    
    for j = 1:num_injectors
        raw_time = forecast_struct.(run_name).(inj_name{j}).time;
        raw_inj = forecast_struct.(run_name).(inj_name{j}).inj_eff;
        
        % Interpolate the injector efficiencies onto same time axis as
        % historical time and set extrapolated values to 0 (assumed
        % injector is off)
        inj = interp1(raw_time,raw_inj,HistoricalStruct.time,'pchip',0);
        forecast_var(i,j) = inj(end);
    end
    waitbar(i/num_realizations);
end
close(h);

% Plot prior forecast variables
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
edges = [0:0.05:1.05];
for i = 1:num_injectors
    subplot(1,num_injectors,i);
    histogram(forecast_var(:,i),edges);
    xlabel([inj_name{i} ' Efficiency']);
    ylabel('Number of Realizations');
    axis tight; axis square;
    set(gcf,'color','w');
    set(gca,'fontsize',fontsize);
end

export_fig([save_path 'Prior_Inj_Efficiency'],'-png','-m3');

close(h);

%% 2. Dimension Reduction on Data Variable
%map_estimate = zeros(num_realizations,num_injectors);
warning('off','all');

%Define basis function for FDA
HistoricalStruct.spline=[3 40]; % 3rd order B-Spline with 40 knots
    
parfor truth_realization = 454:num_realizations-1
    tStart = tic;  % TIC, pair 2  

    avail_real = setdiff(1:num_realizations,truth_realization);
   
    %FPCA
    addpath('../../common/fda_matlab/');
    histPCA = ComputeHarmonicScores(HistoricalStruct,0);
    
    %Mixed PCA
    rmpath('../../common/fda_matlab/');
    [mpca_scores,mpca_obs] = MixedPCA(histPCA,truth_realization);
    
    %Kernel Density Estimation to Obtain Posterior Forecast
    HarmonicScores = [mpca_scores;mpca_obs];
    
    %3. Regression
    PriorInjectorEfficiency = forecast_var(avail_real,:);
    
    num_pdf_pts = 100;
    injValues = zeros(num_pdf_pts,num_injectors);
    injPDF = zeros(num_pdf_pts,num_injectors);
    
    for i = 1:num_injectors
        [injValues(:,i),injPDF(:,i)] = ...
            UpdateProbabilityContinuous(HarmonicScores,...
            PriorInjectorEfficiency(:,i));
        
        [~,idx] = max(injPDF(:,i));
        map_estimate(truth_realization,i) = injValues(idx,i);
    end
    
    tElapsed = toc(tStart);  % TOC, pair 2  
    display(['Realization ' num2str(truth_realization) ...
        ' Took ' num2str(tElapsed) ' seconds to execute']);
end

%%

truth_realization = randi(num_realizations);


% 4. Plot posterior vs prior
% h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
% for i = 1:num_injectors
%     subplot(1,num_injectors,i);
%     [f,xi] = ksdensity(PriorInjectorEfficiency(:,i));
%
%     hold on;
%     plot(injValues(:,i),injPDF(:,i),'linewidth',2);
%     plot(xi,f,'color',[0.5 0.5 0.5],'linewidth',2);
%     plot([forecast_var(truth_realization,i),...
%         forecast_var(truth_realization,i)],...
%         [0,max(injPDF(:,i))],'r','linewidth',2);
%     xlabel(['h'],'fontsize',fontsize);
%     ylabel(['PDF'],'fontsize',fontsize);
%     set(gca,'fontsize',fontsize);
%     axis tight; axis square;
%     hlegend = legend('Posterior','Prior','Truth');
%     set(hlegend,'FontSize',fontsize-6);
%     %set(hlegend,'location','northwest');
%     set(gcf,'color','w');
% end


%export_fig -m3 Efficiency_Regression_Results.png