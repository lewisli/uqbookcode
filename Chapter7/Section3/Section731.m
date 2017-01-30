% Chapter7Scenario1.m
%
% Generate figures to illustrate Scenario 3 of Chapter 9.2: When to abandon
% the oil field.
%
% Author: Lewis Li (lewisli@stanford.edu)
% Original Date: December 27th 2016
% Last Updated: December 27th 2016


addpath('../../common/util');
addpath('../../common/cfca');
addpath('../../common/thirdparty/export_fig/');


% Load in simulation results
results_dir = 'C:\Users\Lewis Li\ResearchData\LibyanCase\COMG\Prior\';
load([results_dir 'Prior.mat']);

CaseName = 'Section7_3_1';
SaveFolder = ['../figures/' CaseName '/'];
%% Generate Data Structs
ForecastColumn = 4;
HistoricalColumn = 4;

% This is the time step that we divide the forecast and history
TimeColumn = 2;
NumTimeSteps = 240;
EndTime = 12000;
ForecastObjName={'Field'};
HistoricalObjName = {'P1','P2','P3','P4','P5'};
ForecastSpline = [4 20];
HistoricalSpline = [4 20];
ForecastStart = 170;
HistoricalEnd = 60;

[HistoricalStruct, ForecastStruct] = GenerateDataStructsWithInterpolation(Data, ...
    PropertyNames, ForecastColumn, HistoricalColumn, TimeColumn, ...
    HistoricalEnd,ForecastStart, NumTimeSteps, ForecastSpline, ...
    HistoricalSpline, ForecastObjName, HistoricalObjName, EndTime);
ForecastStruct.time = linspace(3500,11000,length(ForecastStruct.time));
%% Set aside one realization that we will deem the "reference";
TruthRealization = 460;
FontSize = 22;

% Plot to verify data structures/choice of input/output
h1  = PlotInputResponse( HistoricalStruct,TruthRealization,FontSize,SaveFolder);
h2  = PlotInputResponse( ForecastStruct,0,34,SaveFolder);

%%
HistoricalStruct.spline=[3 40]; % 6th order B-Spline with 20 knots
histPCA = ComputeHarmonicScores(HistoricalStruct,0);

ForecastStruct.spline = [3 20]; % 6th order B-Spline with 20 knots
predPCA = ComputeHarmonicScores(ForecastStruct,4);

%%
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


%% Sample from CFCA posterior and transform forecasts back into time domain
close all;
NumPosteriorSamples = 100;
addpath('../../common/fda_matlab/');
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