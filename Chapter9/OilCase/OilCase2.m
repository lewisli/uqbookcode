% OilCase2.m
%
% Generate figures for Chapter 9 Oil Case 1: Forecasting Saturation Maps
%
% Author: Lewis Li (lewisli@stanford.edu)
% Original Date: December 28th 2016
% Last Modified: Janurary 18th 2017

%% Paths
case_name = 'OilCase2';
input_data_path = '/Volumes/ScratchExternal/ResearchData/LibyanCase/';
figure_output_path = '/Volumes/ScratchExternal/Github/uqbookcode/Chapter9/figures/';

addpath('../../common/cfca');
addpath('../../common/util');
addpath('../../common/export_fig');
addpath('../../common/fda_matlab/');
addpath('../../common/likelihood_continuous/');

% Create folder for saving figures
save_path = [figure_output_path case_name '/'];
if ~exist(save_path, 'dir')
    mkdir(save_path)
end

fontsize = 22;
truth_realization = 1;

%% 1a) Load Production Profile (Data Variable)
load([input_data_path 'Situation1/results/SimulationResults.mat']);

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
HistoricalEnd = 160;

[HistoricalStruct, ~] = GenerateDataStructsWithInterpolation(Data, ...
    PropertyNames, ForecastColumn, HistoricalColumn, TimeColumn, ...
    HistoricalEnd,ForecastStart, NumTimeSteps, ForecastSpline, ...
    HistoricalSpline, ForecastObjName, HistoricalObjName, EndTime);

% Plot Production Profiles
h1  = PlotInputResponse(HistoricalStruct,truth_realization,...
    fontsize);
