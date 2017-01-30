% OilCase2.m
%
<<<<<<< HEAD
% Generate figures for Chapter 9 Oil Case 2: Forecasting Saturation Maps
%
% Author: Lewis Li (lewisli@stanford.edu)
% Original Date: December 28th 2016
% Last Modified: Janurary 25th 2017

%% Paths
case_name = 'OilCase2';
input_data_path = 'C:\Users\Lewis Li\ResearchData\LibyanCase\Situation3\';
figure_output_path = 'C:\Users\Lewis Li\Documents\Github\uqbookcode/Chapter9\figures\';

=======
% Generate figures for Chapter 9 Oil Case 1: Forecasting Saturation Maps
%
% Author: Lewis Li (lewisli@stanford.edu)
% Original Date: December 28th 2016
% Last Modified: Janurary 18th 2017

%% Paths
case_name = 'OilCase2';
input_data_path = '/Volumes/ScratchExternal/ResearchData/LibyanCase/';
figure_output_path = '/Volumes/ScratchExternal/Github/uqbookcode/Chapter9/figures/';
>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b

addpath('../../common/cfca');
addpath('../../common/util');
addpath('../../common/export_fig');
addpath('../../common/fda_matlab/');
<<<<<<< HEAD
=======
addpath('../../common/likelihood_continuous/');
>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b

% Create folder for saving figures
save_path = [figure_output_path case_name '/'];
if ~exist(save_path, 'dir')
    mkdir(save_path)
end

fontsize = 22;
<<<<<<< HEAD

%% 1a) Load Production Profile (Data Variable)
load([input_data_path 'results\Situation3.mat']);
=======
truth_realization = 1;

%% 1a) Load Production Profile (Data Variable)
load([input_data_path 'Situation1/results/SimulationResults.mat']);
>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b

ForecastColumn = 4;
HistoricalColumn = 4;

% This is the time step that we divide the forecast and history
TimeColumn = 2;
<<<<<<< HEAD
ForecastObjName={'Field'}; % Not used

% We will use the production rates for the 5 existing producers as the data
% variable (up to 4000 days)
HistoricalObjName = {'P1','P2','P3','P4','P5'};
EndTime = 4000;

ForecastSpline = [4 20];
HistoricalSpline = [4 20];
ForecastStart = 160;
HistoricalEnd = 400;
NumTimeSteps = 400;

% Interpolate the production rates from the 5 wells to be on a linearly
% spaced time domain
=======
NumTimeSteps = 400;
EndTime = 4000;
ForecastObjName={'P5'}; % Not used
HistoricalObjName = {'P1','P2','P3','P4','P5'};
ForecastSpline = [4 20];
HistoricalSpline = [4 20];
ForecastStart = 160;
HistoricalEnd = 160;

>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b
[HistoricalStruct, ~] = GenerateDataStructsWithInterpolation(Data, ...
    PropertyNames, ForecastColumn, HistoricalColumn, TimeColumn, ...
    HistoricalEnd,ForecastStart, NumTimeSteps, ForecastSpline, ...
    HistoricalSpline, ForecastObjName, HistoricalObjName, EndTime);

% Plot Production Profiles
<<<<<<< HEAD
TruthRealization = 145;
h1  = PlotInputResponse(HistoricalStruct,TruthRealization,fontsize,save_path);

%% 1b) Load saturation data
sat_map_path = [input_data_path 'results\sat_maps\avg_saturation.mat'];
load(sat_map_path);
map_size=size(avgZ);
map_size=map_size(1:2);
num_reals=length(HistoricalStruct.RunNames);

saturation_maps = reshape(avgZ(:,:,HistoricalStruct.RunNames),...
    [prod(map_size),num_reals]);

norm_saturation_maps = saturation_maps-repmat(mean(saturation_maps,2)...
    ,1,num_reals);
mean_image = mean(saturation_maps,2);

% Plot some of the prior forecast maps
num_real_to_plot = 2;
h=figure('Units', 'normalized', 'Position', [0,0,0.85,1]);
for i = 1:num_real_to_plot
    real_no = randi(num_reals);
    ax(i)=subplot(1,2,i);
    imagesc(reshape(saturation_maps(:,real_no),map_size));
    set(gca,'fontsize',fontsize);
    title(['Real : ' num2str(i)],'fontsize',fontsize);
    axis image;
end
set(gcf,'color','w');
hp3 = get(subplot(1,2,1),'Position');
hp4 = get(subplot(1,2,2),'Position');
c = colorbar('Position', [hp3(1)  hp3(2)  hp4(4)-0.04  0.05],'location',...
    'southoutside');
c.Label.String = 'Reservoir Quality';
%export_fig('-m3',[save_path 'prior_forecasts.png']);
%% 2. Perform eigenimage decomposition on the saturation map
S = norm_saturation_maps'*norm_saturation_maps;
[V,D] = eig(S);
Eigenimages = normc(norm_saturation_maps*V);

num_weights = 10;
weights = zeros(num_reals,num_weights);

for i = 1:num_reals
    weight = Eigenimages'*(saturation_maps(:,i)-mean_image);
    weights(i,:)  = weight(end-num_weights+1:end);
end

weightsFlipped = fliplr(weights);


%% 2a) Plot what image looks like with num_weights eigenvalues
realization = TruthRealization;
EvStartIndex = num_reals-num_weights + 1;
EvEndIndex = num_reals;
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
reconstructed = Eigenimages(:,EvStartIndex:EvEndIndex)*weights(...
    realization,:)';
subplot(121);
imagesc(reshape(reconstructed+mean_image,map_size));
set(gcf,'color','w');
set(gca,'fontsize',fontsize);
title(['Reconstructed With ' num2str(num_weights) ' Eigenimages'],...
    'fontsize',fontsize);
axis tight;
axis equal;
subplot(122);
imagesc(reshape(saturation_maps(:,realization),map_size));
title(['Original'],'fontsize',fontsize);
set(gcf,'color','w');
set(gca,'fontsize',fontsize);
axis tight;
axis equal;
hp3 = get(subplot(1,2,1),'Position');
hp4 = get(subplot(1,2,2),'Position');
c = colorbar('Position', [hp3(1)  hp3(2)  hp4(4)-0.04  0.05],'location',...
    'southoutside');
c.Label.String = 'Reservoir Quality';

export_fig('-m3',[save_path 'eigen_reconstruction.png']);
%% 3. Perform FPCA on the data variable
addpath('../../common/fda_matlab/');
HistoricalStruct.spline=[4 30]; % 6th order B-Spline with 20 knots
histPCA = ComputeHarmonicScores(HistoricalStruct,0);

%% 4. Perform Regression
addpath('../../common/fda_matlab/');
% TruthRealization = 122;
% TruthRealization = 126;
% TruthRealization = 2;
% TruthRealization = 157;
% TruthRealization = 145;
%TruthRealization = randi(num_reals)

AvailableRealizations = setdiff(1:num_reals,TruthRealization);
NumHistoricalResponses = 5;
nHarmHist = 10;
Df = zeros(length(AvailableRealizations),NumHistoricalResponses*nHarmHist);

for i = 1:NumHistoricalResponses
    harmscrhist=histPCA{i}.harmscr;
    
    % Need to re-arrange harmonic scores into Df such that the first
    % eigenvalues are placed in first
    for j = 1:nHarmHist
        Df(:,(i-1)*nHarmHist + j) = harmscrhist(AvailableRealizations,j);
        dobs_f(:,(i-1)*nHarmHist + j) = harmscrhist(TruthRealization,j);
    end
end

rmpath('../../common/fda_matlab/');
dobs_f = dobs_f(1:NumHistoricalResponses*nHarmHist);
DfStar = [Df; dobs_f];
[~,score,~,~,explained,~] = pca(DfStar);
mixed_pca_var = cumsum(explained)/sum(explained);
Hf = weightsFlipped(AvailableRealizations,:);

% Maximize correlation between
[A,B,r,Dc,Hc] = canoncorr(Df,Hf);

% Project dobs_f into canonical space
dobs_c=(dobs_f-mean(Df))*A;

% Plot what things look like in canonical space
PlotLowDimModels(Dc,Hc,dobs_c,'c',fontsize);
export_fig('-m3',[save_path 'PlotLowDimModels.png']);

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
%realization = randi(CycleSize);
EigenvaluesToKeep = num_weights;
EvStartIndex = num_reals-EigenvaluesToKeep + 1;
EvEndIndex = num_reals;

% take mean of reconstructed images
reconstructed_posterior = zeros(prod(map_size),CycleSize);

for i = 1:CycleSize
    reconstructed_posterior(:,i) = ...
        Eigenimages(:,EvStartIndex:EvEndIndex)*...
        sampledScores(i,:)';
end

reconstructed = Eigenimages(:,EvStartIndex:EvEndIndex)*...
    weights(TruthRealization,:)'+mean_image;

reconstructed = reshape(reconstructed,map_size);

posterior_mean = reshape(mean(reconstructed_posterior,2),map_size);
posterior_var = reshape(var(reconstructed_posterior,0,2),map_size);
posterior_real = reshape(reconstructed_posterior(:,randi(CycleSize)),...
    map_size);
% truth = reshape(saturation_maps(:,TruthRealization)-mean_image,map_size);
% prior_sat = reshape(mean_image,map_size);

min_sat = 0;
max_sat = max(saturation_maps(:,TruthRealization));
fontsize = 28;
% Plot the mean of the posterior vs truth
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
subplot(121);
imagesc(posterior_mean);
%caxis([min_sat,max_sat])
set(gca,'fontsize',fontsize);
axis tight;
axis equal; axis tight;
title('Posterior Mean','fontsize',fontsize);
hold on;

subplot(122);
imagesc(reshape(saturation_maps(:,TruthRealization),map_size));
%colorbar;
title('Truth','fontsize',fontsize);
set(gcf,'color','w');
set(gca,'fontsize',fontsize);
axis equal; axis tight;
hold on;
caxis([min_sat,max_sat])


hp3 = get(subplot(1,2,1),'Position');
hp4 = get(subplot(1,2,2),'Position');
c = colorbar('Position', [hp3(1)  hp3(2)  hp4(4)-0.04  0.05],'location',...
    'southoutside');
c.Label.String = 'Reservoir Quality';
export_fig('-m3',[save_path 'posterior_forecast.png']);

%%
[X,Y] = meshgrid(1:map_size(1),1:map_size(2));
Truth=reshape(saturation_maps(:,TruthRealization),map_size);
Q = zeros(map_size);
QTrue = zeros(map_size);
h=waitbar(0,'Please wait');
for x=1:map_size(1)
    waitbar(x/map_size(1));
    for y = 1:map_size(2)
        
        distance = (sqrt((x-X).^2+(y-Y).^2)+1);
   
        
        invdist = 1./distance;
        invdist(distance>20) = 0;
        Q(x,y) = sum(sum(posterior_mean'.*invdist));
        QTrue(x,y) = sum(sum(Truth'.*invdist));
    end
end
close(h);

%%
[M,I] = max(Q(:));
[I_row, I_col] = ind2sub(size(Q),I);

hold on;
imagesc(flipud(Truth));
plot(I_col,map_size(1)-I_row+8,'rx','markersize',24,'linewidth',7);
axis tight;
set(gcf,'color','w');
set(gca,'fontsize',fontsize-12);
export_fig('-m2',[save_path 'well_loc.png'])
%%
subplot(311)
imagesc(Truth)
subplot(312)
imagesc(QTrue)
subplot(313);
imagesc(posterior_mean);

%%
% Plot the mean of the posterior vs truth
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
subplot(121);
imagesc(reshape(reconstructed_posterior(:,randi(CycleSize)),map_size));
set(gca,'fontsize',fontsize);
axis tight;
axis equal; axis tight;
title('Posterior Realization 1','fontsize',fontsize);
hold on;

subplot(122);
imagesc(reshape(reconstructed_posterior(:,randi(CycleSize)),map_size));
title('Posterior Realization 2','fontsize',fontsize);
set(gcf,'color','w');
set(gca,'fontsize',fontsize);
axis equal; axis tight;
hold on;


hp3 = get(subplot(1,2,1),'Position');
hp4 = get(subplot(1,2,2),'Position');
c = colorbar('Position', [hp3(1)  hp3(2)  hp4(4)-0.04  0.05],'location',...
    'southoutside');
c.Label.String = 'Reservoir Quality';
export_fig('-m3',[save_path 'posterior_realizations.png']);

%% Plot the mean of the posterior vs truth
h=figure('Units', 'normalized', 'Position', [0,0,0.5,0.5]);
imagesc(posterior_var);
%caxis([min_sat,max_sat])
set(gca,'fontsize',fontsize);
axis tight;
axis equal; axis tight;
title('Posterior Variance','fontsize',fontsize);
hold on;
set(gcf,'color','w');
c = colorbar('southoutside');
c.Label.String = 'Variance';
export_fig('-m3',[save_path 'PosteriorVariance.png']);

%
% subplot(122);
% imagesc(reshape(posterior_var,map_size));
% title('Truth','fontsize',fontsize);
% set(gcf,'color','w');
% set(gca,'fontsize',fontsize);
% axis equal; axis tight;
% hold on;
% caxis([min_sat,max_sat])
% legend('Existing Producers','Existing Injectors','fontsize',23);
%
% hp3 = get(subplot(1,2,1),'Position');
% hp4 = get(subplot(1,2,2),'Position');
% c = colorbar('Position', [hp3(1)  hp3(2)  hp4(4)-0.04  0.05],'location',...
%     'southoutside');
% c.Label.String = 'Reservoir Quality';
% export_fig('-m3',[save_path 'posterior_forecast.png']);


%%
% num_real_plot = 4;
% h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
% for i = 1:num_real_plot
%     real_no = randi(num_reals);
%     ax(i)=subplot(2,2,i);
%     imagesc(reshape(norm_saturation_maps(:,real_no),map_size));
%     set(gca,'fontsize',fontsize);
%     title(['Real : ' num2str(real_no)],'fontsize',fontsize);
%     axis image;
% end


%%
plot(cumsum(flipud(diag(D)))/sum(diag(D)),'linewidth',3);
set(gcf,'color','w');
xlabel('Eigenimage Components','fontsize',fontsize);
ylabel('Proportion of variance','fontsize',fontsize);
set(gca,'fontsize',fontsize);
set(gcf,'color','w');

=======
h1  = PlotInputResponse(HistoricalStruct,truth_realization,...
    fontsize);
>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b
