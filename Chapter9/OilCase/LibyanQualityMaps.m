%% LibyanCaseDemo.m
%
% Using direct forecasting to predict quality maps
%
% Author: Lewis Li
% Date: Feb 2nd 2017

close all; clear all;
addpath('../cfca');
addpath('../util');

input_data_path='/media/lewisli/ScratchExternal/ResearchData/LibyanCase/Situation4/results/';

load([input_data_path 'production_data.mat']);
load([input_data_path 'quality_maps.mat']);

FontSize=24;

%% Generate Data Structs for production data
HistoricalColumn = 4;
ForecastColumn = HistoricalColumn;

% This is the time step that we divide the forecast and history
TimeColumn = 2;
NumTimeSteps = 150;
EndTime = 3000;
ForecastObjName={'P1'};
HistoricalObjName = {'P1','P2','P3','P4','P5'};
ForecastSpline = [4 20];
HistoricalSpline = [4 20];
ForecastStart = 100;
HistoricalEnd = 150;

[HistoricalStruct, ~] = GenerateDataStructsWithInterpolation(Data, ...
    PropertyNames, ForecastColumn, HistoricalColumn, TimeColumn, ...
    HistoricalEnd,ForecastStart, NumTimeSteps, ForecastSpline, ...
    HistoricalSpline, ForecastObjName, HistoricalObjName, EndTime);

%%

TruthRealization=120;
%h1  = PlotInputResponse( HistoricalStruct,TruthRealization,FontSize);
HistoricalStruct.spline=[3 40]; % 6th order B-Spline with 20 knots
histPCA = ComputeHarmonicScores(HistoricalStruct,0);

%% Eigenimage analysis
map_size=size(quality_maps);
num_realizations = map_size(3);
map_size = map_size(1:2);

% Flatten data matrix
quality_map_flat = reshape(quality_maps,[prod(map_size),num_realizations]);

norm_q_map = quality_map_flat-repmat(mean(quality_map_flat,2)...
    ,1,num_realizations);
mean_image = mean(quality_map_flat,2);

% Plot some of the prior forecast maps
num_real_to_plot = 2;
close all;
h=figure('Units', 'normalized', 'Position', [0,0,0.85,1]);

for i = 1:num_real_to_plot
    real_no = randi(num_realizations);
    ax(i)=subplot(1,2,i);
    imagesc(reshape(quality_map_flat(:,real_no),map_size));
    set(gca,'fontsize',FontSize);
    title(['Real : ' num2str(real_no)],'fontsize',FontSize);
    axis image;
end

set(gcf,'color','w');
hp3 = get(subplot(1,2,1),'Position');
hp4 = get(subplot(1,2,2),'Position');
c = colorbar('Position', [hp3(1)  hp3(2)  hp4(4)-0.04  0.05],'location',...
    'southoutside');
c.Label.String = 'Reservoir Quality';

%% 2. Perform eigenimage decomposition on the saturation map
S = norm_q_map'*norm_q_map;
[V,D] = eig(S);

Eigenimages = normc(norm_q_map*V);

num_weights = 25;
weights = zeros(num_realizations,num_weights);

for i = 1:num_realizations
    weight = Eigenimages'*(quality_map_flat(:,i)-mean_image);
    weights(i,:)  = weight(end-num_weights+1:end);
end

weightsFlipped = fliplr(weights);

%% 2a) Plot what image looks like with num_weights eigenvalues
realization = 183;
EvStartIndex = num_realizations-num_weights + 1;
EvEndIndex = num_realizations;
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
reconstructed = Eigenimages(:,EvStartIndex:EvEndIndex)*weights(...
    realization,:)';
subplot(121);
imagesc(reshape(reconstructed+mean_image,map_size));
set(gcf,'color','w');
set(gca,'fontsize',FontSize);
title(['Reconstructed With ' num2str(num_weights) ' Eigenimages'],...
    'fontsize',FontSize);
axis tight;
axis equal;
subplot(122);
imagesc(reshape(quality_map_flat(:,realization),map_size));
title(['Original'],'fontsize',FontSize);
set(gcf,'color','w');
set(gca,'fontsize',FontSize);
axis tight;
axis equal;
hp3 = get(subplot(1,2,1),'Position');
hp4 = get(subplot(1,2,2),'Position');
c = colorbar('Position', [hp3(1)  hp3(2)  hp4(4)-0.04  0.05],'location',...
    'southoutside');
c.Label.String = 'Reservoir Quality';

%% Perform regression between weightsFlipped and Df
%TruthRealization = 27;
TruthRealization=120;

%TruthRealization = 120;
%TruthRealization = 295;
%TruthRealization = 124;


AvailableRealizations = setdiff(1:length(HistoricalStruct.RunNames),...
    TruthRealization);
eigentolerance=0.9;

% Not all realizations executed successfully
Hf = weightsFlipped(HistoricalStruct.RunNames(AvailableRealizations),:);

[Df, Df_obs]=MixedPCA(histPCA,TruthRealization,eigentolerance);

% Maximize correlation between
[A,B,r,Dc,Hc] = canoncorr(Df,Hf);

% Project dobs_f into canonical space
dobs_c=(Df_obs-mean(Df))*A;

% Plot what things look like in canonical space
%PlotLowDimModels(Dc,Hc,dobs_c,'c',FontSize);

% Compute Posterior
% Perform a normal score transform
epsilon = 0;
Hc_gauss = NormalScoreTransform(Hc,0);
C_H = cov(Hc_gauss);
H_CG_Mean = mean(Hc_gauss)';

% Find best linear bit between Dc and Hc_gauss
G = Dc'/Hc_gauss';
DDiff= Dc'-G*Hc_gauss';
C_T = DDiff*DDiff'/length(Dc);
C_Dc = zeros(size(C_T));

% Perform Gaussian Regression
mu_posterior = H_CG_Mean + C_H*G'*pinv(G*C_H*G' + C_T+C_Dc)*...
    (dobs_c'-G*H_CG_Mean);
C_posterior = C_H - C_H*G'*inv(G*C_H*G' + C_T+C_Dc)*G*C_H;

% 5. Sample posterior and reconstruct saturation maps
CycleSize = 100;
PosteriorSamples = mvnrnd(mu_posterior',C_posterior,CycleSize)';

% The h_c are not Gaussian, hence we need to backtransform
PosteriorSamplesTransformed = BackTransform(PosteriorSamples,Hc);

% We need to undo the canonical transform...
HpostCoef = PosteriorSamplesTransformed'*pinv(B)+repmat(mean(Hf,1)',...
    1,CycleSize)'; % H_f_posterior

% Need to flip to match the eigenimage ordering
sampledScores = fliplr(HpostCoef);

% Verify dimension reduction on saturation maps (reconstruct with only 90%)
EigenvaluesToKeep = num_weights;
EvStartIndex = num_realizations-EigenvaluesToKeep + 1;
EvEndIndex = num_realizations;

% take mean of reconstructed images
reconstructed_posterior = zeros(prod(map_size),CycleSize);

for i = 1:CycleSize
    reconstructed_posterior(:,i) = ...
        Eigenimages(:,EvStartIndex:EvEndIndex)*sampledScores(i,:)';
end

reconstructed = Eigenimages(:,EvStartIndex:EvEndIndex)*...
    weights(TruthRealization,:)';
reconstructed = reshape(reconstructed+mean_image,map_size);

posterior_mean = reshape(mean(reconstructed_posterior,2)+mean_image,map_size);
posterior_var = reshape(var(reconstructed_posterior,0,2),map_size);
posterior_real = reshape(reconstructed_posterior(:,randi(CycleSize))++mean_image,...
    map_size);
truth = reshape(quality_map_flat(:,TruthRealization),map_size);

min_sat =  min(quality_map_flat(:,TruthRealization));;
max_sat = max(quality_map_flat(:,TruthRealization));
fontsize = 28;

% Plot the mean of the posterior vs truth
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
subplot(221);
imagesc(posterior_mean);
%caxis([min_sat,max_sat])
set(gca,'fontsize',fontsize);
axis tight;
axis equal; axis tight;
title('Posterior Mean','fontsize',fontsize);
hold on;
colorbar;


subplot(222);
imagesc(reconstructed);
caxis([min_sat,max_sat])
title('Truth','fontsize',fontsize);
set(gcf,'color','w');
set(gca,'fontsize',fontsize);
axis equal; axis tight;
hold on;
colorbar;

subplot(223);
imagesc(posterior_var);

set(gca,'fontsize',fontsize);
axis tight;
axis equal; axis tight;
title('Posterior Variance','fontsize',fontsize);
hold on;
colorbar;

subplot(224);
imagesc(posterior_real);
set(gca,'fontsize',fontsize);
axis tight;
axis equal; axis tight;
title('Posterior Realization','fontsize',fontsize);
hold on;
colorbar;




% hp3 = get(subplot(1,2,1),'Position');
% hp4 = get(subplot(1,2,2),'Position');
% c = colorbar('Position', [hp3(1)  hp3(2)  hp4(4)-0.04  0.05],'location',...
%      'southoutside');
% c.Label.String = 'Reservoir Quality';