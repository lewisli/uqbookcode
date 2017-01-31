% Section72.m
% Generates figures for Section 7.2
% Author: Lewis Li
% Original Date: Nov 22nd 2016

addpath('../../common/plot/');
addpath('../../common/export_fig/');
save_path = '../figures/';
FontSize = 24;
%% Figure 7.4: Example of extrapolation problems
close all;

% Generate random samples
s = RandStream('mt19937ar','seed',1971);

% Simulation domain
num_pts = 250;
x_start = 1;
x_end = 5.5;
X = linspace(x_start,x_end,num_pts);
X = X';

% Specify the parameters for a second order Fourier series
w = .6067;
a0 = 1.6345;
a1 = -0.6235;
b1 = -1.3501;
a2 = -2.1622;
b2 = -.9443;

% Fourier2 is the true (unknown) relationship between X and Y
Y = a0 + a1*cos(X*w) + b1*sin(X*w) + a2*cos(2*X*w) + b2*sin(2*X*w);

% Add in a noise vector
K = max(Y) - min(Y);
noisy = Y +  .2*K*randn(num_pts,1);

% Cut off index
cut_off_index = 50;

% Cut off the second half of the data
noisy_trunc = noisy(1:cut_off_index);
x_trunc = X(1:cut_off_index);

% Generate a scatter plot of first half of points
scatter_size = 150;
h1 = figure('Position', [100, 100, 1049, 895]);
h_scatter = scatter(x_trunc,noisy_trunc,scatter_size,'filled');

% Fit a linear regression and plot
P = polyfit(x_trunc,noisy_trunc,1);
yfit = P(1)*X+P(2);
hold on;
h_linear = plot(X,yfit,'k-.','linewidth',3);

% Get true regression function
foo = fit(X, noisy, 'fourier2');
h_true_fit = plot(foo);
set(h_true_fit,'linewidth',3);

% New input
x_new = 5;
h_new_input = plot([x_new,x_new],[min(noisy);,P(1)*x_end+P(2);],...
    'b--','linewidth',3);

% Predicted value with linear regression
MarkerSize = 30;
pred_val = P(1)*x_new+P(2);
h_pred = plot(x_new,pred_val,'kp','Markersize',MarkerSize,...
    'MarkerFaceColor','k');

% True val
true_val = foo(x_new);
h_true_val = plot(x_new,true_val,'rp','Markersize',MarkerSize,...
    'MarkerFaceColor','r');

h_legend = legend([h_scatter(1),h_linear(1),h_true_fit(1),...
    h_new_input(1),h_pred(1),h_true_val(1)],'Samples',...
    'Regression function','True function','New Data Obs',...
    'Predicted Value','True Value');

set(h_legend,'location','northwest');
ylabel('Prediction Variable');
xlabel('Data Variable');
set(gcf,'color','w');
set(gca,'fontsize',FontSize);
axis tight;
grid on;
export_fig('-m3',[save_path 'Extrapolation_danger.png']);

%% Kernel Density Estimate Figures
% Generate some fake data for illustration. On purpose i will use a
% Gaussian Mixture model
num_mixtures = 10;
num_dim = 2;
rng(1); % For reproducibility
mu = zeros(num_mixtures,num_dim);
sigma = zeros(num_dim,num_dim,num_mixtures);

% Populate each Gaussian with random mean and covariance
for i = 1:num_mixtures
    mu(i,:) = randn(num_dim,1)*5;
    A = rand(num_dim);
    A = 0.5*(A+A');
    sigma(:,:,i) = A + num_dim*eye(num_dim);
end

% Uniform mixture weights
p = ones(1,num_mixtures)/num_mixtures;

% Create GM object and sample from it
obj = gmdistribution(mu,sigma,p);
Y = random(obj,1000);
h = Y(:,1);
d = Y(:,2);
x_range = [min(d) max(d)];
y_range = [min(h) max(h)];

h_fig=figure('Units', 'normalized', 'Position', [0,0,0.5,0.5]);
hold on;
h1 = scatter(d,h,50,'filled');
dobs = 5;
plot([dobs dobs],y_range,'r','LineWidth',3)
text(dobs+0.25,y_range(1)*0.85,'d_{obs}','FontSize',20)

grid on;
ax = gca;
FormatPlot(h1,'d','h',FontSize);
export_fig('-m3',[save_path 'KDE_Raw_Data.png']);
%
% generate a Gaussian mixture with distant modes
addpath('../../common/kde2d');
MIN_XY = [min(d) min(h)];
MAX_XY = [max(d) max(h)];
[bandwidth,density,ycoord,xcoord]=kde2d(Y,512,MIN_XY,MAX_XY);

%%
FontSize = 34;
h_fig=figure('Units', 'normalized', 'Position', [0,0,1,1]);
subplot(1,2,1);
h2=surf(xcoord,ycoord,density,'LineStyle','none'), view([0,90]);
grid on;
hold on;
plot3([dobs dobs],x_range,[1 1],'r','LineWidth',3)
text(dobs+0.25,x_range(1)*0.85,1,'d_{obs}','FontSize',FontSize,'color','r')
FormatPlot(h2,'d','h',FontSize);

subplot(1,2,2);
[c,cindex] = min(abs(xcoord(:,1)-dobs))
conditional = density(cindex,:);
forecast = ycoord(cindex,:);

Area=trapz(forecast,conditional);

h3=plot(forecast,conditional./Area,'LineWidth',2);
FormatPlot(h3,'h','f(h|d_{obs})',FontSize');
export_fig('-m3',[save_path 'KDE_Results.png']);
close all;

%% Figure 7.3
FontSize = 24;
addpath('../../common/util');
addpath('../../common/cfca');
addpath('../../common/fda_matlab/');
input_data_path = 'C:\Users\Lewis Li\ResearchData\LibyanCase\';

load([input_data_path 'Situation1\results\SimulationResults.mat']);

ForecastColumn = 4;
HistoricalColumn = 4;

% This is the time step that we divide the forecast and history
TimeColumn = 2;
NumTimeSteps = 400;
EndTime = 4000;
ForecastObjName={'P5'}; % Not used
HistoricalObjName = {'P3'};
ForecastSpline = [4 20];
HistoricalSpline = [4 20];
ForecastStart = 160;
HistoricalEnd = 80;

[HistoricalStruct, ~] = GenerateDataStructsWithInterpolation(Data, ...
    PropertyNames, ForecastColumn, HistoricalColumn, TimeColumn, ...
    HistoricalEnd,ForecastStart, NumTimeSteps, ForecastSpline, ...
    HistoricalSpline, ForecastObjName, HistoricalObjName, EndTime);

% Plot Production Profiles
%h1  = PlotInputResponse(HistoricalStruct,0,FontSize);
histPCA = ComputeHarmonicScores(HistoricalStruct,0);

%%
close all;
outlier_hist = [-350,-800,300];

numPredCoeffs = 3;

PhiH = eval_fd(HistoricalStruct.time,histPCA{1}.harmfd);

% Re-construct time series
h_reconstructed = eval_fd(HistoricalStruct.time,histPCA{1}.meanfd) + ...
     PhiH(:,1:numPredCoeffs)*outlier_hist';

HistoricalStructOutlier = HistoricalStruct;
HistoricalStructOutlier.data = [HistoricalStructOutlier.data;h_reconstructed'];

h1  = PlotInputResponse(HistoricalStructOutlier,...
    size(HistoricalStructOutlier.data,1),FontSize,'../figures/');

