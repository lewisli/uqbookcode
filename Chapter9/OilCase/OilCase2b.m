% OilCase2b.m
% Generate figures for Chapter 9 Oil Case 2: Forecasting Quality Maps
%
% Author: Lewis Li (lewisli@stanford.edu)
% Original Date: December 28th 2016
% Last Modified: Janurary 25th 2017

close all; clear all;
addpath('../cfca');
addpath('../util');

case_name = 'OilCase2';
figure_output_path = '../figures/';

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

FontSize=24;
%% 1a) Load Production Profile (Data Variable) and
% Quality Maps (Prediction Variable)
input_data_path='/media/lewisli/ScratchExternal/ResearchData/LibyanCase/Situation4/results/';
load([input_data_path 'production_data.mat']);
load([input_data_path 'quality_maps.mat']);

%% 1b) Generate Data Structs for production data
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

% Plot Production Profiles

TruthRealization=120;
h1  = PlotInputResponse( HistoricalStruct,TruthRealization,FontSize,save_path);
HistoricalStruct.spline=[3 40]; % 6th order B-Spline with 20 knots
histPCA = ComputeHarmonicScores(HistoricalStruct,0);

%% Eigenimage analysis
map_size=size(quality_maps);
map_size = map_size(1:2);

% Quality map contains failed runs, we need to remove those invalid maps
num_realizations = length(HistoricalStruct.RunNames);

quality_maps = quality_maps(:,:,HistoricalStruct.RunNames);

% Flatten quality maps and remove failed runs
quality_map_flat = reshape(quality_maps,...
    [prod(map_size),num_realizations]);

% Normalize flattened quality maps
norm_q_map = quality_map_flat-repmat(mean(quality_map_flat,2)...
    ,1,num_realizations);

% Compute mean of quality maps
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
%export_fig('-m2',[save_path 'Prior_forecasts.png']);
%% 2. Perform eigenimage decomposition on the saturation map
S = norm_q_map'*norm_q_map;
[V,D] = eig(S);

Eigenimages = normc(norm_q_map*V);

num_weights = 50;
coefficients = zeros(num_realizations,num_weights);

for i = 1:num_realizations
    coefficient = Eigenimages'*(quality_map_flat(:,i)-mean_image);
    coefficients(i,:)  = coefficient(end-num_weights+1:end);
end

% We need to flip the weights around because the eigenvalues are generated
% from smallest to largest and we only want to perform regression on the largest
quality_map_coefficients = fliplr(coefficients);

%% 2a) Plot what image looks like with num_weights eigenvalues
TruthRealization=139;
EvStartIndex = num_realizations-num_weights + 1;
EvEndIndex = num_realizations;
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
reconstructed_truth = Eigenimages(:,EvStartIndex:EvEndIndex)*coefficients(...
    TruthRealization,:)';
subplot(121);
imagesc(reshape(reconstructed_truth+mean_image,map_size));
set(gcf,'color','w');
set(gca,'fontsize',FontSize);
title(['Reconstructed With ' num2str(num_weights) ' Eigenimages'],...
    'fontsize',FontSize);
axis tight;
axis equal;
subplot(122);
imagesc(reshape(quality_map_flat(:,TruthRealization),map_size));
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
%TruthRealization=120;

%TruthRealization = 120;
%TruthRealization = 295;
%TruthRealization =    330;

%% Get true posterior in canonical space
close all;

%TruthRealization=255;
%TruthRealization=175;
%TruthRealization=139;
%TruthRealization=422;
TruthRealization=36;


% Number of quality maps coefficients we will use for regression
num_image_coeff=10;
eigentolerance=0.99;

Hf = quality_map_coefficients(:,1:num_image_coeff);
w = warning ('off','all');
[Df, ~]=MixedPCA(histPCA,0,eigentolerance);


[A,B,r,Dc,Hc] = canoncorr(Df,Hf);

% Get d_obs in canonical space
Df_obs = Df(TruthRealization,:);
dobs_c=(Df_obs-mean(Df))*A;

% Find best linear bit between Dc and Hc
C_H = cov(Hc);
H_CG_Mean = mean(Hc)';

G = Dc'/Hc';
DDiff= Dc'-G*Hc';
C_T = DDiff*DDiff'/length(Dc);
C_Dc = zeros(size(C_T));

% Perform Gaussian Regression
mu_posterior = H_CG_Mean + C_H*G'*pinv(G*C_H*G' + C_T+C_Dc)*...
    (dobs_c'-G*H_CG_Mean);
C_posterior = C_H - C_H*G'*inv(G*C_H*G' + C_T+C_Dc)*G*C_H;

%mu_posterior = Hc(TruthRealization,:);
% Sample from posterior and unproject
NumPosteriorSamples = 5000;
H_c_posterior = mvnrnd(mu_posterior',C_posterior,NumPosteriorSamples)';

% We need to undo the canonical transform...
H_f_posterior = H_c_posterior'*pinv(B); % H_f_posterior

mean_hf_post = mean(H_f_posterior)
true_hf = Hf(TruthRealization,:)

% Reconstruct posterior quality maps
% Need to flip the weights to match the eigenimage ordering
posterior_coefficients = fliplr(H_f_posterior);

% Verify dimension reduction on saturation maps (reconstruct with only 90%)
EigenvaluesToKeep = num_image_coeff;
EvStartIndex = num_realizations-EigenvaluesToKeep + 1;
EvEndIndex = num_realizations;

% Reconstruct posterior maps by multiplying posterior coefficients with
% appropiate eigenimages
reconstructed_posterior = zeros(prod(map_size),NumPosteriorSamples);
for i = 1:NumPosteriorSamples
    reconstructed_posterior(:,i) = ...
        Eigenimages(:,EvStartIndex:EvEndIndex)*posterior_coefficients(i,:)';
end

% Reconstruct truth using the same number of eigenimage as a comparison
reconstructed_truth_flat = Eigenimages(:,EvStartIndex:EvEndIndex)*...
    coefficients(TruthRealization,1:num_image_coeff)';

% Compute mean of posterior samples
posterior_mean = reshape(mean(reconstructed_posterior,2),map_size);

% Compute prior mean image (need to add back onto posterior samples)
prior_mean=reshape(mean_image,map_size);

% Reshape the reconstructed truth into an image
reconstructed_truth = reshape(reconstructed_truth_flat,map_size);

% Compute posterior
posterior_var = reshape(var(reconstructed_posterior,0,2),map_size);

% Get color limits
min_sat = min(quality_map_flat(:,TruthRealization));
max_sat = max(quality_map_flat(:,TruthRealization));

%
x = linspace(1,map_size(2),map_size(2));
y = linspace(1,map_size(1),map_size(1));
[xx,yy]=meshgrid(x,y);


% Plot reconstructed truth
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
subplot(131);
PlotContouredImage(quality_maps(:,:,TruthRealization),map_size);
title('Truth','fontsize',fontsize);
set(gcf,'color','w');
set(gca,'fontsize',fontsize);
axis equal; axis tight;
hold on;
%colorbar;

subplot(133);
posterior_mean_normalized = (posterior_mean+prior_mean);
PlotContouredImage(posterior_mean_normalized,map_size);
title('Posterior Mean','fontsize',fontsize);
set(gcf,'color','w');
set(gca,'fontsize',fontsize);
axis equal; axis tight;
hold on;

subplot(132);
PlotContouredImage(prior_mean,map_size);
caxis([min_sat,max_sat])
title('Prior Mean','fontsize',fontsize);
set(gcf,'color','w');
set(gca,'fontsize',fontsize);
axis equal; axis tight;
hold on;

hp3 = get(subplot(1,3,1),'Position');
hp4 = get(subplot(1,3,3),'Position');
c = colorbar('Position', [hp3(1)  hp3(2)  hp4(4)-0.04  0.05],'location',...
    'southoutside');
c.Label.String = 'Reservoir Quality';

figure;
imagesc(posterior_var);
title('Prior Variance','fontsize',fontsize);
set(gcf,'color','w');
set(gca,'fontsize',fontsize);
axis equal; axis tight;
hold on;
%% Decision Model

[X,Y] = meshgrid(1:map_size(1),1:map_size(2));
Truth=quality_maps(:,:,TruthRealization);
Q = zeros(map_size);
QTrue = zeros(map_size);
h=waitbar(0,'Please wait');
for x=1:map_size(1)
    waitbar(x/map_size(1));
    for y = 1:map_size(2)
        
        distance = (sqrt((x-X).^2+(y-Y).^2)+1);
   
        invdist = 1./distance;
        invdist(distance>10) = 0;
        Q(x,y) = sum(sum(posterior_mean_normalized'.*invdist));
        QTrue(x,y) = sum(sum(Truth'.*invdist));
    end
end
close(h);

%%
[M,I] = max(QTrue(:));
[I_row, I_col] = ind2sub(size(Q),I);

hold on;
imagesc(flipud(posterior_mean_normalized));
plot(I_col,map_size(1)-I_row,'rx','markersize',24,'linewidth',7);
axis tight;
set(gcf,'color','w');
set(gca,'fontsize',fontsize-12);


