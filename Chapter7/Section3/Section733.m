% Chapter7Scenario3.m
%
%

%% 1. Load production (forecast) and seismic (data)
clear all; close all;

addpath('../common/util');
addpath('../common/cfca');
addpath('../common/fda_matlab');
addpath('../common/export_fig');

case_name = 'Section7_3_3';

raw_data_path = 'C:\Users\Lewis Li\ResearchData\GuangModel\';
figure_output_path = 'C:\Users\Lewis Li\Documents\Github\uqbookcode\Chapter7\figures\';


load([raw_data_path 'timelapse_ensemble.mat']);
load([raw_data_path 'production_summary.mat']);
load([raw_data_path 'date.mat']);

% Create folder for saving figures
save_path = [figure_output_path case_name '\'];
if ~exist(save_path, 'dir')
    mkdir(save_path)
end

% Number of realizations
num_realizations = length(production_summary_samet);

% Number of time steps for 4D seismic
num_time_steps = 9;

% Resolution of seismic
map_size = [200,200];

% Plot font size
FontSize = 22;

% Which time lapse index to use
time_lapse_index = 6;

% Production start index
prod_start_index = floor(length(date_common)*...
    time_lapse_index/num_time_steps);

TruthRealization = 12;

%% 2. Create Data Structure and Dimension Reduction on Production Data
ProductionStruct = struct();
ProductionStruct.name = 'Production (stb/day)';
ProductionStruct.time = date_common(prod_start_index:end);
ProductionStruct.type = 'Forecast';
ProductionStruct.ObjNames = {'Field Oil Rate'};
ProductionStruct.data = zeros(num_realizations,length(date_common)-...
    prod_start_index+1,length(ProductionStruct.ObjNames));

for i = 1:num_realizations
    raw_data = production_summary_samet{i};
    ProductionStruct.data(i,:,:) = raw_data(prod_start_index:end,1);
end

% Plot forecast data
PlotInputResponse(ProductionStruct,0,FontSize,save_path);
%export_fig -m3 Scenario3ForecastPrior.png

% Perform FPCA on Production Data
ProductionStruct.spline = [3 15];
forecstFPCA = ComputeHarmonicScores(ProductionStruct,0);

%%
scatter3(forecstFPCA{1}.harmscr(:,1),forecstFPCA{1}.harmscr(:,2),...
    forecstFPCA{1}.harmscr(:,3),'filled')
axis tight;
xlabel('h^f_1: 65.98% of variance');
ylabel('h^f_2: 22.34% of variance');
zlabel('h^f_3: 7.05% of variance');
set(gcf,'color','w');
set(gca,'fontsize',14);

export_fig('-m3',[save_path 'ForecastScatter.png']);

%% 3. Perform Eigenimage Analysis on Seismic Data
map_start_index = prod(map_size)*(time_lapse_index-1)+1;
map_end_index = prod(map_size)*time_lapse_index;
sat_map = timelapse_ensemble_tosave(map_start_index:map_end_index,:);
%% Plot two random realizations from the prior
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
for i = 1:2
    subplot(1,2,i);
    index = randi(num_realizations);
    image = reshape(sat_map(:,index),map_size)';
    imageZoomed = image(100:200,1:100);
    hcolor =  pcolor(imageZoomed);
    set(hcolor,'EdgeColor','none');
    shading interp;
    colorbar;
    caxis([0 1]);
    axis equal; axis tight;
    set(gcf,'color','w');
    set(gca,'fontsize',22);
    title(['Realization ' num2str(i)],'fontsize',22);
end
export_fig('-m3',[save_path 'PriorDataRealizations.png']);


%%

norm_sat_maps=sat_map-repmat(mean(sat_map,2),1,num_realizations);
S = norm_sat_maps'*norm_sat_maps;

[V,D] = eig(S);
Eigenimages = normc(norm_sat_maps*V);
mean_image = mean(sat_map,2);

% Compute percentage of variance is explained by each eigenvalue
eigenimage_variance = cumsum(flipud(diag(D)))/sum(diag(D));

% Number of eigenvalues to keep.. doesn't really matter since dim(Seismic)
% >> dim(production)
num_weights = 30;
weights = zeros(num_realizations,num_weights);

for i = 1:num_realizations
    weight = Eigenimages'*(sat_map(:,i)-mean_image);
    weights(i,:)  = weight(end-num_weights+1:end);
end

% Need to reverse order of weights
weightsFlipped = fliplr(weights);

%%

image = reshape(sat_map(:,TruthRealization),map_size)';
imageZoomed = image(100:200,1:100);
hcolor =  pcolor(imageZoomed);
set(hcolor,'EdgeColor','none');
shading interp;
set(gcf,'color','w');
set(gca,'FontSize',FontSize);
colorbar;
axis square;
axis tight;
export_fig('-m3',[save_path 'D_obs.png']);

% %%
%
%
% h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
% for i= 1:2
%     subplot(1,2,i);
%     imagesc(reshape(Eigenimages(:,end-i+1),map_size));
%     title(['Eigenimage ' num2str(i)],'FontSize',FontSize);
%     set(gcf,'color','w');
%     set(gca,'FontSize',FontSize);
%     colorbar;
%     axis square;
%     axis tight;
% end
%

%%
close all;
realization = randi(num_realizations);
%realization = 12;
EvStartIndex = num_realizations-num_weights + 1;
EvEndIndex = num_realizations;
h=figure('Units', 'normalized', 'Position', [0,0,1,1]);
reconstructed = Eigenimages(:,EvStartIndex:EvEndIndex)*...
    weights(realization,:)'+mean_image;
image = reshape(reconstructed,map_size)';
imageZoomed = image(100:200,1:100);
subplot(121);
imagesc(imageZoomed);
title(['Reconstructed With ' num2str(num_weights) ' Eigenimages'],'fontsize',FontSize);
axis equal; axis tight;
set(gca,'fontsize',22);
colorbar;
caxis([0 1]);
subplot(122);
image = reshape(sat_map(:,realization),map_size)';
imageZoomed = image(100:200,1:100);
imagesc(imageZoomed);
title(['Original'],'fontsize',FontSize);
set(gcf,'color','w');
colorbar;
axis equal; axis tight;
set(gca,'fontsize',22);
caxis([0 1]);
export_fig('-m3',[save_path 'Eigenreconstruction.png']);
%% Get Functional Components for Production Data
TruthRealization = 12;
AvailableRealizations = setdiff(1:num_realizations,...
    TruthRealization);

MinEigenValues = 2;
EigenTolerance = 0.99;
nHarmPred = GetNumHarmonics(forecstFPCA{1}, MinEigenValues,EigenTolerance);

% Forecast oil production
Hf = forecstFPCA{1}.harmscr(:,1:nHarmPred);

TrueForecastFunctional = Hf(TruthRealization,:);
Hf = Hf(AvailableRealizations,:);
% Get Df
dobs_f = weightsFlipped(TruthRealization,:);
Df = weightsFlipped(AvailableRealizations,:);

[A, B, ~, Dc,Hc] = canoncorr(Df,Hf);

% Project dobs_f into canonical space
dobs_c=(dobs_f-mean(Df))*A;

%PlotLowDimModels(Dc,Hc,dobs_c,'c',FontSize);

% Perform a normal score transform
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

%% Sample posterior using rejection sampling

NumEstimates = 100;
CycleSize = 1000;
NumValid = 1;
NumCycles = 0;

addpath('fda_matlab');
h_output = zeros(NumEstimates,length(ProductionStruct.time));
hc_out = zeros(NumEstimates,length(mu_posterior));
hf_out = zeros(NumEstimates,length(mu_posterior));

FontSize=20;
rng('shuffle');
ReferenceForecastFirstStep = 5000;

while (NumValid < NumEstimates)
    % The posterior distribution is a normal distribution in canonical space
    PosteriorSamples = mvnrnd(mu_posterior',C_posterior,CycleSize)';
    
    % The h_c are not Gaussian, hence we need to backtransform
    PosteriorSamplesTransformed = BackTransform(PosteriorSamples,Hc);
    
    % We need to undo the canonical transform...
    HpostCoef = PosteriorSamplesTransformed'*pinv(B)+repmat(mean(Hf,1)',...
        1,CycleSize)'; % H_f_posterior
    
    % Finally, we reconstruct the time series (mean_FDA + sum(HpostCoef*PhiH))
    numPredCoeffs = size(Hf,2);
    
    % Principal components for H
    PhiH = eval_fd(ProductionStruct.time,forecstFPCA{1}.harmfd);
    
    % Re-construct time series
    h_reconstructed = repmat(eval_fd(ProductionStruct.time,forecstFPCA{1}.meanfd),...
        1,CycleSize) + PhiH(:,1:numPredCoeffs)*HpostCoef(:,1:numPredCoeffs)';
    
    % Compute difference between first forecasted time and observed value
    % at that time
    Difference = (repmat(ReferenceForecastFirstStep,1,CycleSize)-...
        h_reconstructed(1,:));
    
    DirectionalValidityFlag = Difference > 0;
    CycleValid = logical(DirectionalValidityFlag');
    
    NumCycleValid = sum(CycleValid);
    
    % This means we need another cycle of model sampling
    if (NumValid+NumCycleValid < NumEstimates+1)
        h_output(NumValid:NumValid+NumCycleValid-1,:) = ...
            h_reconstructed(:,CycleValid)';
        
        hc_out(NumValid:NumValid+NumCycleValid-1,:) = ...
            PosteriorSamplesTransformed(:,CycleValid)';
        
        hf_out(NumValid:NumValid+NumCycleValid-1,:) = ...
            HpostCoef(CycleValid,:);
        
        NumValid = NumValid+NumCycleValid;
        NumCycles = NumCycles + 1;
    else
        NumCycles = NumCycles + 1;
        RemainingModels = NumEstimates - NumValid + 1;
        display(['Finished after ' num2str(NumCycles) ' cycles.']);
        
        ValidModels = h_reconstructed(:,CycleValid)';
        ValidHc = PosteriorSamplesTransformed(:,CycleValid)';
        ValidHf = HpostCoef(CycleValid,:);
        
        h_output(NumValid:end,:) = ValidModels(1:RemainingModels,:);
        hc_out(NumValid:end,:) = ValidHc(1:RemainingModels,:);
        
        NumOut = size(hf_out(NumValid:end,:),2);
        hf_out(NumValid:end,:) = ValidHf(1:RemainingModels,1:NumOut);
        
        figure(2);
        hold on;
        scatter(Hc(:,1),Hc(:,2),100,[0.5 0.5 0.5]);
        scatter(PosteriorSamples(1,1:NumEstimates),...
            PosteriorSamples(2,1:NumEstimates),100,'b','filled');
        hlegend = legend('Prior Models','Posterior Samples');
        set(hlegend,'fontsize',FontSize-4);
        set(hlegend,'location','best');
        set(gcf,'color','w');
        xlabel('h_1^c','fontsize',FontSize);
        ylabel('h_2^c','fontsize',FontSize);
        grid on;
        set(gca,'fontsize',FontSize-4);
        
        return
    end
    
    display(['After ' num2str(NumCycles) ' cycles. Rejection sampler has found '...
        num2str(NumValid) ' models']);
    
    
end
% Plot posterior
%%
% Compute quantiles
[PriorQuantiles, PosteriorQuantiles] = ComputeQuantiles(...
    ProductionStruct.data, h_output);

PlotPosteriorSamplesAndQuantiles(ProductionStruct,TruthRealization, ...
    h_output,PriorQuantiles,PosteriorQuantiles);
export_fig('-m3',[save_path 'Posterior.png']);
%%
FigHandle = figure('Position', [100, 100, 1049, 895]);
hold on;
h0 = plot(ProductionStruct.time,PriorQuantiles','color',[0.5 0.5 0.5],...
    'LineWidth',3);
h2 = plot(ProductionStruct.time,ProductionStruct.data(TruthRealization,:),...
    'r','LineWidth',3);
h3 = plot(ProductionStruct.time,PosteriorQuantiles,'b--','LineWidth',3);
legend([h0(1), h2(1),h3(1)],'Prior','Reference','Posterior');
xlabel('t(days)');ylabel(['Forecasted: ' ProductionStruct.name]);axis square;
title('Quantiles');
set(gca,'FontSize',24);
set(gcf,'color','w');
axis tight;
export_fig -m2 Scenario3Posterior.png
%%
% Plot sampled responses and quantiles



