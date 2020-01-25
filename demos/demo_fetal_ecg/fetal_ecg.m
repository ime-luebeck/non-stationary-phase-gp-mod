% DEMO_FETAL_ECG
% 
% Description
%   Demo for processing fetal ECG signals with the proposed warpSKI method.
%   See the paper for more details. By running this script, processed data
%   is stored to 'paper_results/results_fetal_ecg', where then the plots
%   from the paper can be generated.
% 
% References:
%   Grasshoff, J., Jankowski, A. and Rostalski, P. (2019). Scalable Gaussian
%   Process Regression for Kernels with a non-stationary phase,
%   arxiv.org/abs/1912.11713

%% Load data from Physionet
[hdr, record] = edfread('./demos/demo_fetal_ecg/r01.edf');

abd4 = record(5,:);
abd1 = record(1,:);
fs = hdr.frequency(5);
T = readtable('./demos/demo_fetal_ecg/Physionet/r01_annotations.txt');
qrs = table2array(T(:,1));
Time = (0:(size(record,2)-1))/fs;

qrs_fetal_indizes = arrayfun(@(x)(find(abs(Time-x)==min(abs(Time-x)))),qrs);

%% Preprocessing

% detect Rpeaks (maternal)
qrs_maternal = PeakDetection2(-record(2,:), fs, [], [], [], 0.15, 1);

% remove baseline
[z, p, k] = butter(9, 2/(fs/2), 'high');
[sos, g] = zp2sos(z, p, k);
abd4_filt = filtfilt(sos, g, [abd4(10000:-1:1), abd4]);
abd4_filt = abd4_filt(10001:end);
abd1_filt = filtfilt(sos, g, [abd1(10000:-1:1), abd1]);
abd1_filt = abd1_filt(10001:end);

figure
ax1 = subplot(2, 1, 1);
hold on
plot(Time, abd4_filt)
plot(Time(qrs_maternal == 1), abd4_filt(qrs_maternal == 1), 'o')
plot(Time(qrs_fetal_indizes - 4), abd4_filt(qrs_fetal_indizes - 4), 'o')
title('Channel 4')
ax2 = subplot(2, 1, 2);
hold on
plot(Time, record(2,:))
plot(Time(qrs_maternal == 1), record(2,qrs_maternal == 1), 'o')
linkaxes([ax1, ax2], 'x')
title('Channel 2')

%% Kernel definition
start_t = 1;
end_t = 11;
segm = (Time>=start_t & Time<end_t);
abd4_segm = abd4_filt(segm);
time_segm = Time(segm);

tau_maternal = Time((qrs_maternal==1));
tau_maternal = tau_maternal(tau_maternal>start_t & tau_maternal<end_t);

tau_fetal = Time((qrs_fetal_indizes));
tau_fetal = tau_fetal(tau_fetal>start_t & tau_fetal<end_t);

% likelihood function
likfunc = @likGauss;

% warped kernel for fetal ECG
covPe = {@covProd,{@covSEisoU, @covPeriodic}};
p = @(t) (timeWrap( t, tau_maternal ));
dp = @(t) (dtimeWrap( t, tau_maternal));
covPeWarped_maternal = {@covWarp, covPe, p, dp, 1};

% warped kernel for maternal ECG
p = @(t) (timeWrap( t, tau_fetal ));
dp = @(t) (dtimeWrap( t, tau_fetal));
covPeWarped_fetal = {@covWarp, covPe, p, dp, 1};

% full exact kernel
csu = {'covSum', {covPeWarped_maternal, covPeWarped_fetal}};

% Hyperparameters: length SE, length PE, period, variance
hyp_maternal = log([18, 0.25, 2*pi, 8]); 
hyp_fetal = log([18, 0.1, 2*pi, 5]);

hyp.cov = [ hyp_maternal, hyp_fetal ];
hyp.lik = log([5]);
hyp.mean = [];

% create sets of (warped) inducing points
U_rad_maternal = 2*pi*(-1:0.005:(length(tau_maternal)+1))'; % 0.05
U_maternal = invTimeWrap( U_rad_maternal, tau_maternal);
xg_1 = {{U_maternal}}; % grid

U_rad_fetal = 2*pi*(-1:0.005:(length(tau_fetal)+1))'; % 0.05
U_fetal = invTimeWrap( U_rad_fetal, tau_fetal);
xg_2 = {{U_fetal}}; % grid

figure
hold on
plot(time_segm(1:2:end)', abd4_segm(1:2:end)');
plot(U_maternal, zeros(size(U_maternal)), 'o');
plot(U_fetal, zeros(size(U_fetal)), 'o');
legend('data', 'U_maternal', 'U_fetal')
title('Data and inducing points')

% warpSKI kernels
covGrid_1 = {@apxGrid,{covPeWarped_maternal},xg_1};
covGrid_2 = {@apxGrid,{covPeWarped_fetal},xg_2};

% sum of kernels
csu_grid = {'apxGridSum',{covGrid_1, covGrid_2}};

%% Compare Marginal likelihood surfaces (approximate and exact)

% options
opt.cg_maxit = 1000; opt.cg_tol = 1e-2;
opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 25;
opt.ldB2_maxit = 100;

hyp_test = hyp;

% validate MVMs
K = apx(hyp_test,csu,time_segm(1:2:end)');
K_grid = apx(hyp_test,csu_grid,time_segm(1:2:end)',opt);
test_vector = randn(size(time_segm(1:2:end)'));
figure
hold on
plot(K.mvm(test_vector))
plot(K_grid.mvm(test_vector),'--')
title('MVM with kernel matrix')
legend('dense', 'SKI')

% evaluate marginal likelihood
tic
[post_grid, nlZ_grid, dnlZ_grid] = infGaussLik(hyp_test, {@meanZero}, csu_grid,...
    likfunc,  time_segm(1:2:end)', abd4_segm(1:2:end)', opt);
elaps_time_marg_lik_eval_ski = toc;

tic
[post_dense, nlZ_dense, dnlZ_dense] = infGaussLik(hyp_test, {@meanZero}, csu,...
    likfunc,  time_segm(1:2:end)', abd4_segm(1:2:end)', opt);
elaps_time_marg_lik_eval_batch = toc;

disp(['Lanczos Marg. Lik.: ', num2str(nlZ_grid)])
disp(['Exact Marg. Lik.: ', num2str(nlZ_dense)])

% evaluate Marginal likelihood along one line
x_vec = 1.5:0.05:4;
y_vec = 0;

% exact solution
func = @(x,y) getJthOutput(@infGaussLik, 2:3, struct('cov',[hyp_test.cov(1:3),x,hyp_test.cov(5:8)],'lik', hyp_test.lik,'mean', []),...
     {@meanZero}, csu, likfunc, time_segm(1:2:end)', abd4_segm(1:2:end)', opt);
[ f_mesh_non_warp, grad_mesh_non_warp, x_coord, y_coord ] = meshfun( func, x_vec, y_vec, 4);

% SKI warp solution
func = @(x,y) getJthOutput(@infGaussLik, 2:3, struct('cov',[hyp_test.cov(1:3),x,hyp_test.cov(5:8)],'lik', hyp_test.lik,'mean', []),...
     {@meanZero}, csu_grid, likfunc, time_segm(1:2:end)', abd4_segm(1:2:end)', opt);
[ f_mesh_warp, grad_mesh_warp, x_coord, y_coord ] = meshfun( func, x_vec, y_vec, 4);

figure
plot(x_vec, f_mesh_non_warp)
hold on
plot(x_vec, f_mesh_warp)
title('Comaparison of LogLikelihoods')

% Save results
path = './paper_results/results_fetal_ecg/';
save([path, 'results_marg_lik.mat'],'csu_grid', 'csu', 'likfunc', 'hyp', 'time_segm', 'abd4_segm',...
    'elaps_time_marg_lik_eval_ski', 'elaps_time_marg_lik_eval_batch', 'tau_fetal', 'tau_maternal',...
    'x_vec', 'f_mesh_non_warp', 'f_mesh_warp')

fileID = fopen([path, 'results_marg_lik.txt'], 'w');
fprintf(fileID, 'Fetal ECG - Results for Evaluation of Marginal Likelihood\n-------------------------------------\n');
fprintf(fileID, 'Hyperparameters of Covariance:\n');
fprintf(fileID, '%f; ', hyp.cov);
fprintf(fileID, '\n\nHyperparameters of Likelihood:\n');
fprintf(fileID, '%f; ', hyp.lik);
fprintf(fileID, '\n\nThe fourth hyperparameter is varied, while the others are kept fixed. The marginal likelihood is evaluated with both batch-GP and warp-SKI.\n\n');
fprintf(fileID, 'Time for marginal lik. evaluation (batch): %f', elaps_time_marg_lik_eval_batch);
fprintf(fileID, '\nTime for marginal lik. evaluation (warp-SKI): %f', elaps_time_marg_lik_eval_ski);
fprintf(fileID, '\n\nInducing points maternal: %f', length(U_maternal));
fprintf(fileID, '\nInducing points fetal: %f', length(U_fetal));
fclose(fileID);

%% optimize parameters with batch GP (on 10s segment)
prior.cov = {{@priorDelta} ,{@priorDelta},{@priorDelta},[],{@priorDelta},{@priorDelta},{@priorDelta},[]};
prior.lik = {{@priorDelta}}; % {@priorGauss,5,0.1^2}
inf = {@infPrior,@infGaussLik,prior};

disp('-------------------------------------------------------------------')
disp('Optimize Batch')
disp('-------------------------------------------------------------------')

tic
[hyp_opt, f_val, iter_batch] = minimize_lbfgsb(hyp, @gp, -100, inf, {@meanZero}, csu, likfunc, time_segm(1:2:end)', abd4_segm(1:2:end)');
time_for_learning_batch = toc;

%hyp = hyp_opt;
disp(['Hyperparamters: ', sprintf('%f ', exp(hyp_opt.cov))])
disp(['Hyperparamters of likelihood: ', sprintf('%f ', exp(hyp_opt.lik))])

% get intermediate result
tic
post_exact = infGaussLik( hyp_opt, {@meanZero}, csu, likfunc, time_segm(1:2:end)', abd4_segm(1:2:end)');
time_for_inference_batch = toc;

% get matrices for separation
K1 = feval(covPeWarped_maternal{:}, hyp_opt.cov(1:4), time_segm(1:2:end)', time_segm(1:2:end)');
K2 = feval(covPeWarped_fetal{:}, hyp_opt.cov(5:end), time_segm(1:2:end)', time_segm(1:2:end)');

% do separation
signal_1_exact = K1*post_exact.alpha;
signal_2_exact = K2*post_exact.alpha;

figure
ax1 = subplot(4,1,1);
hold on
plot(time_segm(1:2:end), abd4_segm(1:2:end)')
xlim([start_t, end_t])
title('Mixture')
plot(time_segm(1:2:end), signal_1_exact, '--')
ax2 = subplot(4,1,2);
plot(time_segm(1:2:end), signal_1_exact)
xlim([start_t, end_t])
title('Maternal ECG')
ax3 = subplot(4,1,3);
plot(time_segm(1:2:end), signal_2_exact)
xlim([start_t, end_t])
title('Fetal ECG')

ax4 = subplot(4,1,4);
plot(time_segm(1:2:end), abd4_segm(1:2:end)' - signal_1_exact - signal_2_exact)
xlim([start_t, end_t])
title('Noise')

linkaxes([ax1, ax2, ax3, ax4], 'x');

% evaluate SNR improvement
indices1 = arrayfun(@(x)( find(abs(time_segm(1:2:end)-x)==min(abs(time_segm(1:2:end)-x)),1)), tau_maternal);
indices2 = arrayfun(@(x)( find(abs(time_segm(1:2:end)-x)==min(abs(time_segm(1:2:end)-x)),1)), tau_fetal);

SNR_raw = get_SNR( abd4_segm(1:2:end), indices1, indices2, 0.1, 0.05, round(fs/2));
SNR_filtered_batch = get_SNR( signal_2_exact, indices1, indices2, 0.1, 0.05, round(fs/2));

SNR_improvement_batch = 20*log10(SNR_filtered_batch) - 20*log10(SNR_raw);

%% optimize parameters with warp-SKI (on 10s segment)

% optimize hyperparameters
opt.cg_maxit = 1000; opt.cg_tol = 5e-2; % options
opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 20;
opt.ldB2_maxit = 100;
infg = @(varargin) infGaussLik(varargin{:},opt);
inf = {@infPrior,infg,prior};
disp('-------------------------------------------------------------------')
disp('Optimize SKI')
disp('-------------------------------------------------------------------')

tic
[hyp_opt_lbfgs_gpml, ~, iter_ski] = minimize_lbfgsb(hyp, @gp, -100, inf, [], csu_grid, likfunc, time_segm(1:2:end)', abd4_segm(1:2:end)');
time_for_learning_ski = toc;
disp(['Hyperparamters of kernel: ', sprintf('%f ', exp(hyp_opt_lbfgs_gpml.cov))])
disp(['Hyperparamters of likelihood: ', sprintf('%f ', exp(hyp_opt_lbfgs_gpml.lik))])

opt_learning = opt;

% get intermediate result
opt.cg_maxit = 1000; opt.cg_tol = 5e-3; % options
opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 0;

tic
post_grid = infGaussLik( hyp_opt_lbfgs_gpml, {@meanZero}, csu_grid, likfunc, time_segm(1:2:end)', abd4_segm(1:2:end)', opt);
time_for_inference_ski = toc;

opt_inference = opt;

% get primitives for separation
hyp1=hyp_opt_lbfgs_gpml;
hyp1.cov = hyp_opt_lbfgs_gpml.cov(1:4);
hyp2=hyp_opt_lbfgs_gpml;
hyp2.cov = hyp2.cov(5:end);
K_grid1 = apx(hyp1,csu_grid{2}{1},time_segm(1:2:end)',opt);
K_grid2 = apx(hyp2,csu_grid{2}{2},time_segm(1:2:end)',opt);

% do separation
signal_1 = K_grid1.mvm(post_grid.alpha);
signal_2 = K_grid2.mvm(post_grid.alpha);

figure
ax1 = subplot(4,1,1);
hold on
plot(time_segm(1:2:end), abd4_segm(1:2:end)')
xlim([start_t, end_t])
title('Mixture')
plot(time_segm(1:2:end), signal_1, '--')
ax2 = subplot(4,1,2);
plot(time_segm(1:2:end), signal_1)
xlim([start_t, end_t])
title('Maternal ECG')
ax3 = subplot(4,1,3);
plot(time_segm(1:2:end), signal_2)
xlim([start_t, end_t])
title('Fetal ECG')
ax4 = subplot(4,1,4);
plot(time_segm(1:2:end), abd4_segm(1:2:end)' - signal_1 - signal_2)
xlim([start_t, end_t])
title('Noise')

linkaxes([ax1, ax2, ax3, ax4], 'x');

% Plot comparison
figure
ax1 = subplot(3,1,1);
plot(time_segm(1:2:end), abd4_segm(1:2:end)')
ax2 = subplot(3,1,2);
plot(time_segm(1:2:end), K2*post_exact.alpha)
ylim([min(abd4_segm(1:2:end)), max(abd4_segm(1:2:end))])
ax3 = subplot(3,1,3);
plot(time_segm(1:2:end), K_grid2.mvm(post_grid.alpha))
ylim([min(abd4_segm(1:2:end)), max(abd4_segm(1:2:end))])
linkaxes([ax1,ax2,ax3],'x')

% evaluate SNR improvement
SNR_filtered_ski= get_SNR( signal_2, indices1, indices2, 0.1, 0.05, round(fs/2));

SNR_improvement_ski = 20*log10(SNR_filtered_ski) - 20*log10(SNR_raw);

% Store results
path = './paper_results/results_fetal_ecg/';
save([path, 'results_small_scale.mat'],'csu_grid', 'csu', 'likfunc', 'prior', 'hyp', 'time_segm', 'abd4_segm',...
    'opt_learning', 'opt_inference', 'hyp_opt', 'hyp_opt_lbfgs_gpml', 'time_for_learning_batch',...
    'time_for_learning_ski', 'time_for_inference_batch', 'time_for_inference_ski', 'iter_batch', 'iter_ski', 'tau_fetal', 'tau_maternal',...
    'post_exact', 'post_grid', 'signal_1', 'signal_2', 'post_grid', 'signal_1_exact', 'signal_2_exact',...
    'SNR_improvement_batch', 'SNR_improvement_ski', 'SNR_raw', 'SNR_filtered_batch', 'SNR_filtered_ski');
fileID = fopen([path, 'results_small_scale.txt'], 'w');
fprintf(fileID, 'Number of data-points: %f\n\n', length(abd4_segm(1:2:end)));
fprintf(fileID, 'Learned Hyperparameters of Covariance Batch:\n');
fprintf(fileID, '%f; ', hyp_opt.cov);
fprintf(fileID, '\nLearned Hyperparameters of Covariance warp-SKI:\n');
fprintf(fileID, '%f; ', hyp_opt_lbfgs_gpml.cov);
fprintf(fileID, '\n\nOptions for Learning:\nIterations: %f; \nTolerance: %f; \nHutchinson vectors: %f; \nHutchinson Iterations: %f \n',...
    opt_learning.cg_maxit, opt_learning.cg_tol, opt_learning.ldB2_hutch, opt_learning.ldB2_maxit);
fprintf(fileID, 'Options for Inference:\nIterations: %f; \nTolerance: %f \n', opt_inference.cg_maxit, opt_inference.cg_tol);
fprintf(fileID, '\nTime for Inference (batch): %f s\n', time_for_inference_batch);
fprintf(fileID, 'Time for Inference (warp-SKI): %f s\n', time_for_inference_ski);
fprintf(fileID, 'Time for Learning (batch): %f s\n', time_for_learning_batch);
fprintf(fileID, 'Time for Learning (warp-SKI): %f s\n', time_for_learning_ski);
fprintf(fileID, '\nInducing points maternal: %f', length(U_maternal));
fprintf(fileID, '\nInducing points fetal: %f', length(U_fetal));
fprintf(fileID, '\n\nLBFGS Iterations (batch): %f', iter_batch);
fprintf(fileID, '\nLBFGS Iterations (warp-SKI): %f', iter_ski);
fprintf(fileID, '\n\nSNR Improvement (batch): %f', SNR_improvement_batch);
fprintf(fileID, '\nSNR Improvement (warp-SKI): %f', SNR_improvement_ski);
fclose(fileID);

%% Stress Test
start_t = 1;
end_t = 101;
segm = (Time>=start_t & Time<end_t);
abd4_segm = abd4_filt(segm);
time_segm = Time(segm);

tau_maternal = Time((qrs_maternal==1));
tau_maternal = tau_maternal(tau_maternal>start_t & tau_maternal<end_t);

tau_fetal = Time((qrs_fetal_indizes));
tau_fetal = tau_fetal(tau_fetal>start_t & tau_fetal<end_t);

% create sets of inducing points
U_rad_maternal = 2*pi*(-1:0.01:(length(tau_maternal)+1))'; % 0.05
U_maternal = invTimeWrap( U_rad_maternal, tau_maternal);
xg_1 = {{U_maternal}}; % grid

U_rad_fetal = 2*pi*(-1:0.01:(length(tau_fetal)+1))'; % 0.05
U_fetal = invTimeWrap( U_rad_fetal, tau_fetal);
xg_2 = {{U_fetal}}; % grid

covGrid_1 = {@apxGrid,{covPeWarped_maternal},xg_1};
covGrid_2 = {@apxGrid,{covPeWarped_fetal},xg_2};

csu_grid = {'apxGridSum',{covGrid_1, covGrid_2}};

% optimize hyperparameters
opt.cg_maxit = 4000; opt.cg_tol = 1e-1; % options
opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 20;
opt.ldB2_maxit = 100;
infg = @(varargin) infGaussLik(varargin{:},opt);
inf = {@infPrior,infg,prior};

tic;
[hyp_opt_lbfgs_gpml, ~, iter] = minimize_lbfgsb(hyp, @gp, -100, inf, [], csu_grid, likfunc, time_segm(1:end)', abd4_segm(1:end)');
time_for_learning = toc;

opt_learning = opt;

% get intermediate result
opt.cg_maxit = 2500; opt.cg_tol = 5e-3; % options
opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 0;

tic;
post = infGaussLik( hyp_opt_lbfgs_gpml, {@meanZero}, csu_grid, likfunc, time_segm(1:end)', abd4_segm(1:end)', opt);
time_for_inference = toc;

opt_inference = opt;

hyp1=hyp_opt_lbfgs_gpml;
hyp1.cov = hyp_opt_lbfgs_gpml.cov(1:4);
hyp2=hyp_opt_lbfgs_gpml;
hyp2.cov = hyp2.cov(5:end);
K_grid1 = apx(hyp1,csu_grid{2}{1},time_segm(1:end)',opt);
K_grid2 = apx(hyp2,csu_grid{2}{2},time_segm(1:end)',opt);

% do separation
signal_1 = K_grid1.mvm(post.alpha);
signal_2 = K_grid2.mvm(post.alpha);

figure
ax1 = subplot(4,1,1);
hold on
plot(time_segm(1:end), abd4_segm(1:end)')
xlim([start_t, end_t])
title('Mixture')
plot(time_segm(1:end), signal_1, '--')
plot(time_segm(qrs_maternal(segm) == 1), abd4_segm(qrs_maternal(segm) == 1), 'o')

ax2 = subplot(4,1,2);
plot(time_segm(1:end), signal_1)
xlim([start_t, end_t])
title('Maternal ECG')

ax3 = subplot(4,1,3);
plot(time_segm(1:end), signal_2)
xlim([start_t, end_t])
title('Fetal ECG')

ax4 = subplot(4,1,4);
plot(time_segm(1:end), abd4_segm(1:end)' - signal_1 - signal_2)
xlim([start_t, end_t])
title('Noise')

linkaxes([ax1, ax2, ax3, ax4], 'x');

% evaluate SNR improvement
indices1 = arrayfun(@(x)( find(abs(time_segm-x)==min(abs(time_segm-x)))), tau_maternal);
indices2 = arrayfun(@(x)( find(abs(time_segm-x)==min(abs(time_segm-x)))), tau_fetal);

SNR_raw = get_SNR( abd4_segm, indices1, indices2, 0.1, 0.05, fs);
SNR_filtered = get_SNR( signal_2, indices1, indices2, 0.1, 0.05, fs);

SNR_improvement = 20*log10(SNR_filtered) - 20*log10(SNR_raw);

% Store results
path = './paper_results/results_fetal_ecg/';
save([path, 'results_large_scale.mat'],'csu_grid', 'likfunc', 'prior', 'hyp', 'time_segm', 'abd4_segm',...
    'opt_learning', 'opt_inference', 'hyp_opt_lbfgs_gpml', 'time_for_learning',...
    'time_for_inference', 'iter', 'tau_fetal', 'tau_maternal',...
    'post', 'signal_1', 'signal_2', 'SNR_improvement', 'SNR_raw', 'SNR_filtered');
fileID = fopen([path, 'results_large_scale.txt'], 'w');
fprintf(fileID, 'Number of data-points: %f\n\n', length(abd4_segm));
fprintf(fileID, 'Learned Hyperparameters of Covariance warp-SKI:\n');
fprintf(fileID, '%f; ', hyp_opt_lbfgs_gpml.cov);
fprintf(fileID, '\n\nOptions for Learning:\nIterations: %f; \nTolerance: %f; \nHutchinson vectors: %f; \nHutchinson Iterations: %f \n',...
    opt_learning.cg_maxit, opt_learning.cg_tol, opt_learning.ldB2_hutch, opt_learning.ldB2_maxit);
fprintf(fileID, 'Options for Inference:\nIterations: %f; \nTolerance: %f \n', opt_inference.cg_maxit, opt_inference.cg_tol);
fprintf(fileID, '\nTime for Inference (warp-SKI): %f s\n', time_for_inference);
fprintf(fileID, 'Time for Learning (warp-SKI): %f s\n', time_for_learning);
fprintf(fileID, '\nInducing points maternal: %f', length(U_maternal));
fprintf(fileID, '\nInducing points fetal: %f', length(U_fetal));
fprintf(fileID, '\n\nLBFGS Iterations: %f', iter);
fprintf(fileID, '\n\nSNR Improvement: %f', SNR_improvement);
fclose(fileID);

%% functions
% time warping
function [ theta ] = timeWrap( t, tau )
N = length(tau);
phase = 2*pi*(0:(N-1));
theta = interp1(tau, phase, t, 'linear', 'extrap');
end

% derivative time warping
function [ dtheta ] = dtimeWrap( t, tau )
N = length(tau);
dtau = (2*pi)./diff(tau);
dtau = [dtau(1);dtau(:)];
tau = [min(t)-1e-5; tau(:)];
dtheta = interp1(tau(1:(N)), dtau, t, 'previous', 'extrap');
end

% inverse time warping
function [t] = invTimeWrap( theta, tau)
N = length(tau);
phase = 2*pi*(0:(N-1));
t = interp1(phase, tau, theta, 'linear', 'extrap');
end