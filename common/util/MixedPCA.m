function [mpca_scores, mpca_obs] = MixedPCA(FunctionalStruct,truth_real)
%MIXEDPCA Computes Mixed PCA of a FPCA Object
%
% Inputs:
%   FunctionalStruct: Structure containing harmscr for each response
%   variable
%   truth_real[Optional]: Whether to set aside a realization as d_obs
% Outputs:
%   mpca_scores: Scores of response variables
%   mpca_obs: Score of observed data


% This is the number of variables
num_wells = length(FunctionalStruct);

% Concenated normalized scores
norm_scores = [];

% FPCA library has poor choice of naming convention, need to remove for PCA
<<<<<<< HEAD
rmpath('../../common/fda_matlab');
=======
rmpath('../../thirdparty/fda_matlab');
>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b

for i = 1:num_wells
    % Perform regular PCA on each well
    [coeff,score,latent] = pca(FunctionalStruct{i}.harmscr);
    
    % Normalize the PCA scores by the first latent variable
<<<<<<< HEAD
    norm_score = FunctionalStruct{i}.harmscr/latent(1);
=======
    norm_score = score/latent(1);
>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b
    
    % Concanate the norm_score
    norm_scores = [norm_scores norm_score];
end

% Perform PCA on concatenated matrix
[~,mpca_scores,~,~,explained] = pca(norm_scores);

% Compute explained variance
explained = cumsum(explained)/sum(explained);

% Check number of components to keep
eigenToKeep = 1;
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

