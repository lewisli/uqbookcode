function [mpca_scores, mpca_obs] = MixedPCA(FunctionalStruct,truth_real)
%MIXEDPCA Computes Mixed PCA of a FPCA Object
%
% Inputs:
%   FunctionalStruct: Structure containing harmscr for each response
%   variable
%   truth_real[Optional]: Whether to set aside a realization as d_obs
% Outputs:
%   mpca_scores: Scores of response variables with 99% of variance kept
%   mpca_obs: Score of observed data


% This is the number of variables
num_wells = length(FunctionalStruct);

% Concenated normalized scores
norm_scores = [];

% FPCA library has poor choice of naming convention, need to remove for PCA
rmpath('../../common/fda_matlab');


for i = 1:num_wells
    % Perform regular PCA on each well
    [coeff,score,latent] = pca(FunctionalStruct{i}.harmscr);
    
    % Normalize the PCA scores by the first singular value, which is the
    norm_score = FunctionalStruct{i}.harmscr/sqrt(latent(1));

    % Concanate the norm_score
    norm_scores = [norm_scores norm_score];
end

% Perform PCA on concatenated matrix
[~,mpca_scores,~,~,explained] = pca(norm_scores);

% Compute explained variance
explained = cumsum(explained)/sum(explained);

% Check number of components to keep
eigenToKeep = 2;
ix = max(find(explained > 0.99, 1, 'first'),eigenToKeep);

% Whether we set aside a truth realization
if nargin<2
    mpca_scores = mpca_scores(:,1:ix);
    mpca_obs = 0;
else
    avail_real = setdiff(1:size(mpca_scores,1),truth_real);
    mpca_scores = mpca_scores(avail_real,1:ix);
    mpca_obs = mpca_scores(truth_real,1:ix);
end

end

