% search and add gpml library, then load the non-stationary phase mod from
% './mod/'
listing = dir('./gpml-matlab-v4.2*');

if isempty(listing)
    warning("GPMLv4.2 not found. Please add GPML to the base directory './non-stationary-phase-gp-mod/'.")
else
    addpath(genpath(['./',listing(1).name,'/']));
end

% add mod to path
addpath(genpath('./mod/'), '-begin'); % mod has higher position in search path
addpath(genpath('./utils/'));
addpath(genpath('./demos/'));
addpath(genpath('./paper_results/'));