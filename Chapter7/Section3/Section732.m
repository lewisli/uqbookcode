% Libyan_Injector_Efficiency.m
%
% Generate figures to illustrate Scenario 1 of Chapter 9.2: When to abandon
% the oil field.
%
% Author: Lewis Li (lewisli@stanford.edu)
% Original Date: December 24th 2016
% Last Updated: December 27th 2016
clear all; clc; close all;
addpath('../../common/cfca');
addpath('../../common/likelihood_continuous');
addpath('../../common/util');
addpath('../../common/export_fig/');

%% Load in producer data
results_dir='C:\Users\Lewis Li\ResearchData\LibyanCase\Situation1\results\';
load([results_dir 'SimulationResults.mat']);
SaveFolder='../figures/Section7_3_2/';

FontSize=24;
%% Generate Structure for data
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

h1  = PlotInputResponse(HistoricalStruct,1,FontSize,SaveFolder);

%% Load in forecast variable (Injector Efficiency)
num_realizations=length(HistoricalStruct.RunNames);
inj_name = 'I1';
inj_dir = [results_dir '/waf/'];

forecast_var = zeros(num_realizations,1);
full_forecast = zeros(num_realizations,HistoricalEnd);
for i = 1:num_realizations-1
    
    index = HistoricalStruct.RunNames(i);
    run_name = ['Run' num2str(index)];
    forecast_struct = load([inj_dir run_name '.mat']);
    
    raw_time = forecast_struct.(run_name).(inj_name).time;
    raw_inj = forecast_struct.(run_name).(inj_name).inj_eff;
    
    % Interpolate the injector efficiencies onto same time axis as
    % historical time and set extrapolated values to 0 (assumed injector is
    % off)
    inj = interp1(raw_time,raw_inj,HistoricalStruct.time,'pchip',0);
    forecast_var(i) = inj(end);
    full_forecast(i,:) = inj;
end

%% Plot prior forecasts
h=figure('Units', 'normalized', 'Position', [0,0,0.5,0.5]);
histogram(forecast_var,25);
xlabel('Injector 1 Efficency', 'FontSize', FontSize);
ylabel('Number of Realizations', 'FontSize', FontSize);
set(gca,'fontsize',FontSize-4);
set(gcf,'color','w');

export_fig('-m3',[SaveFolder '/prior_inj.png']);

% hold on;
% for i = 1:num_realizations
%     plot(HistoricalStruct.time,full_forecast(i,:),'color',[0.5 0.5 0.5]);
% end
% set(gcf,'color','w');
% xlabel('Time (days)','FontSize',22);
% ylabel('Injector 1 Efficiency','FontSize',22);
% set(gca,'FontSize',19);


%% Perform dimension reduction on historical data
addpath('../../common/fda_matlab/');
truth_real = 258;
%truth_real = 199;

avail_real = setdiff(1:num_realizations,truth_real);

% Define basis function for FDA
HistoricalStruct.spline=[3 40]; % 3rd order B-Spline with 40 knots

% FPCA
histPCA = ComputeHarmonicScores(HistoricalStruct,0);

% Mixed PCA
[mpca_scores,mpca_obs] = MixedPCA(histPCA,truth_real);

% Kernel Density Estimation to Obtain Posterior Forecast
HarmonicScores = [mpca_scores;mpca_obs];
PriorInjectorEfficiency = forecast_var(avail_real);

[injValues,injPDF] = UpdateProbabilityContinuous(HarmonicScores,...
    PriorInjectorEfficiency);

%% Plot scatter
ObservedLineThickness = 3;
ScatterSize = 100;
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
for i = 1:3
    subplot(1,3,i);
    hold on;
    scatter(mpca_scores(:,i),forecast_var(avail_real),ScatterSize,'filled');
    xlabel(['d_' num2str(i)],'fontsize',FontSize);
    ylabel(['h'],'fontsize',FontSize);
    set(gca,'fontsize',FontSize);
    plot([mpca_obs(i),mpca_obs(i)],[min(forecast_var),max(forecast_var)],'r-',...
        'LineWidth',ObservedLineThickness);
    text(mpca_obs(i) + abs(mpca_obs(i))*0.25,min(forecast_var) + ...
        abs(min(forecast_var))+0.05,'d_{obs}','Fontweight','b','FontSize',FontSize);
    axis square;
end

set(gcf,'color','w');
export_fig('-m3',[SaveFolder 'Efficiency_Scatter.png']);
%% Plot prior vs posterior
h=figure('Units', 'normalized', 'Position', [0,0,0.5,0.5]);
[f,xi] = ksdensity(PriorInjectorEfficiency);
hold on;
plot(injValues,injPDF,'linewidth',2);
plot(xi,f,'color',[0.5 0.5 0.5],'linewidth',2);
plot([forecast_var(truth_real),forecast_var(truth_real)],[0,max(injPDF)],'r','linewidth',2);
xlabel(['h'],'fontsize',FontSize);
ylabel(['PDF'],'fontsize',FontSize);
set(gca,'fontsize',FontSize);
axis tight;
hlegend = legend('Posterior','Prior','Truth');
set(hlegend,'FontSize',FontSize);
set(gcf,'color','w');
grid on;
export_fig('-m3',[SaveFolder 'Efficiency_Regression_Results.png']);


%% Maximum a posteriori probability (MAP) estimate
[~,idx] = max(injPDF);
map_estimate = injValues(idx)
mean_estimate = injValues*injPDF/length(injPDF)


%% Perform multiple linear regression
[Mdl] = fitlm(mpca_scores,forecast_var(avail_real));
[y_mlr,~] = predict(Mdl,mpca_obs,'prediction','observation')

Color=[0    0.4470    0.7410];
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
subplot(121);
plotResiduals(Mdl);
xlabel('Residuals','Fontsize',FontSize);
ylabel('Number of Realizations','FontSize',FontSize);
axis tight; axis square;
set(gca,'fontsize',FontSize);
set(gcf,'color','w');
subplot(122);
plotResiduals(Mdl,'fitted','color',Color,'MarkerSize',25,'Marker','.');
xlabel('Fitted Values','Fontsize',FontSize);
ylabel('Residuals','FontSize',FontSize);
axis tight; axis square;
set(gca,'fontsize',FontSize);
set(gcf,'color','w');
export_fig('-m3',[SaveFolder 'residual_histo.png']);





