% OilCase2.m
% Generate figures for Chapter 9 Oil Case 3: Forecasting Future Rates
%
% Author: Lewis Li (lewisli@stanford.edu)
% Original Date: December 30th 2016
% Last Modified: Janurary 30th 2017
close all; clear all; clc;
results_path = 'C:\Users\Lewis Li\ResearchData\LibyanCase\Situation3\results\Situation3.mat';
load(results_path);
SaveFolder = ['../figures/OilCase3/'];
NumRealizations = 500;

%% Generate Data Structs
ForecastColumn = 4;
HistoricalColumn = 4;

% This is the time step that we divide the forecast and history
TimeColumn = 2;
NumTimeSteps = 450;
EndTime = 9000;
ForecastObjName={'Field'};
HistoricalObjName = {'P1','P2','P3','P4','P5'};
ForecastSpline = [4 20];
HistoricalSpline = [4 20];
ForecastStart = 250;
HistoricalEnd = 200;

[HistoricalStruct, ForecastStruct] = GenerateDataStructsWithInterpolation(Data, ...
    PropertyNames, ForecastColumn, HistoricalColumn, TimeColumn, ...
    HistoricalEnd,ForecastStart, NumTimeSteps, ForecastSpline, ...
    HistoricalSpline, ForecastObjName, HistoricalObjName, EndTime);

TruthRealization = 464;
FontSize = 20;
close all;

h1=PlotInputResponse(HistoricalStruct,TruthRealization,FontSize,SaveFolder);
%h2=PlotInputResponse(ForecastStruct,0,FontSize+16,SaveFolder);
%

addpath('../../common/cfca');
addpath('../../common/util');
addpath('../../common/export_fig');
addpath('../../common/fda_matlab/');

%
addpath('../../common/fda_matlab/');
HistoricalStruct.spline=[3 40]; % 6th order B-Spline with 20 knots
histPCA = ComputeHarmonicScores(HistoricalStruct,4);
SaveFolder = '../figures/OilCase3/';
%ForecastStruct.spline = [6 20]; % 6th order B-Spline with 20 knots
predPCA = ComputeHarmonicScores(ForecastStruct,3);

% Perform CFCAa
% The eigenvalue tolerance sets the number of eigenvalues we will keep
% after FPCA. Keeping too many eigenvalues may need to highly oscillating
% posterior times series; while keeping too little results in oversmoothed
% and unrealistic models
EigenvalueTolerance = 0.99;
OutlierPercentile = 95;
epsilon = 0;

% Run CFCA: The results are the mean/covariance of h*(in Gaussian space)
% and Dc,Hc are the prior model coefficients in the canonical space
PlotLevel = 1;
FontSize = 24;
[ mu_posterior, C_posterior, Dc, Df, Hc, Hf, B, dobs_c] = ComputeCFCAPosterior(...
    HistoricalStruct, ForecastStruct, TruthRealization, EigenvalueTolerance,...
    OutlierPercentile,PlotLevel,FontSize,SaveFolder,epsilon);

%
close all;
addpath('../../common/fda_matlab/');
NumPosteriorSamples = 100;

[SampledPosteriorRealizations,Hf_post]= SampleCanonicalPosterior(...
    mu_posterior,C_posterior,NumPosteriorSamples,Hc,B,Hf,...
    ForecastStruct.time,predPCA,0,0,0,SaveFolder);

% Compute quantiles
[PriorQuantiles, PosteriorQuantiles] = ComputeQuantiles(...
    ForecastStruct.data, SampledPosteriorRealizations);
close all;
% Plot sampled responses and quantiles
PlotPosteriorSamplesAndQuantiles(ForecastStruct,TruthRealization, ...
    SampledPosteriorRealizations,PriorQuantiles,PosteriorQuantiles,SaveFolder);

%%
% Plot quantiles
EconomicalViable = 600;

FigHandle = figure('Position', [100, 100, 1049, 895]);
hold on;
h0 = plot(ForecastStruct.time,PriorQuantiles','color',[0.5 0.5 0.5],...
    'LineWidth',3);
h2 = plot(ForecastStruct.time,ForecastStruct.data(TruthRealization,:),...
    'r','LineWidth',3);
h3 = plot(ForecastStruct.time,PosteriorQuantiles,'b--','LineWidth',3);
h4 = plot(ForecastStruct.time,...
    repmat(EconomicalViable,length(ForecastStruct.time)),'g-.',...
    'linewidth',4);
legend([h0(1), h3(1),h2(1),h4(1)],'Prior','Posterior','Reference','Min Viable Rate');
xlabel('t(days)');ylabel(['Forecasted: ' ForecastStruct.name]);axis square;
title('Abandonment Decision');
set(gca,'FontSize',24);
axis tight;
set(gcf,'color','w');
% export_fig -m2 AbandonmentDecision.png