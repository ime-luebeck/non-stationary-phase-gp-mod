% DEMO_SOURCE_SEP_NON_STAT_1D_GRIDBASED
% 
% Description
%   Simple example of source separation of two non-stationary kernels in
%   1D.
% 
% References:
%   Grasshoff, J., Jankowski, A. and Rostalski, P. (2019). Scalable Gaussian
%   Process Regression for Kernels with a non-stationary phase,
%   arxiv.org/abs/1912.11713

%% Input Signal
% x vector
X = (0:0.01:10)';

% phase-wrapping
tau1 = [0:0.5:5, 5.5:1:10]';
tau2 = [0:2:5,7:0.8:10]';

X_tilde_1 = timeWrap(X, tau1);
X_tilde_2 = timeWrap(X, tau2);

% two sine signals with different frequencies
Y_orig_1 = sin(X_tilde_1);
Y_orig_2 = sin(X_tilde_2);
sigma2_n = 1;

Y = Y_orig_1 + Y_orig_2 + sqrt(sigma2_n)*randn(size(X));

% plot raw data
figure
hold on
plot(X,Y)
plot(X,Y_orig_1)
plot(X,Y_orig_2)
title('Data')
legend('Measurement', 'function 1', 'function 2')

%% GP regression

% make warped sets of inducing points
U_tilde_1 = (timeWrap(X(1), tau1):0.05:timeWrap(X(end), tau1))';
U_1 = invTimeWrap( U_tilde_1, tau1);
U_tilde_2 = (timeWrap(X(1), tau2):0.05:timeWrap(X(end), tau2))';
U_2 = invTimeWrap( U_tilde_2, tau2);

% inducing points
xg_1 = {{U_1}};
xg_2 = {{U_2}};

% make kernel
cov_per_1 = @covPeriodic;
p1 = @(t) (timeWrap( t, tau1 ));
dp1 = @(t) (dtimeWrap( t, tau1));
covPeWarped_1 = {@covWarp,{cov_per_1},p1,dp1,1};
hyp_1 = [0.5, log(2*pi), 0];

cov_per_2 = @covPeriodic;
p2 = @(t) (timeWrap( t, tau2 ));
dp2 = @(t) (dtimeWrap( t, tau2));
covPeWarped_2 = {@covWarp,{cov_per_2},p2,dp2,1};
hyp_2 = [0.5, log(2*pi), 0];

% make likelihood function
likfunc = @likGauss;

% full kernels
csu_grid = {'apxGridSum',{{@apxGrid, {covPeWarped_1}, xg_1},{@apxGrid, {covPeWarped_2}, xg_2}}};
csu = {'covSum', {covPeWarped_1, covPeWarped_2}};

% hyperparameters
hyp = struct('mean', [], 'cov', [hyp_1, hyp_2], 'lik', 0);

% Compare primitives
opt.cg_maxit = 200; opt.cg_tol = 5e-5; % options
opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 40;
K_grid = apx(hyp,csu_grid,X,opt);
W = ones(length(X),1)/sigma2_n;
[ldB2,solveKiW,dW,dhyp,post_grid.L] = K_grid.fun(W);

K = apx(hyp,csu,X);

% compare MVMs with exact and approximate kernel
test_vector = randn(size(X));

figure
title('Comparison of MVMs')
hold on
plot(K.mvm(test_vector))
plot(K_grid.mvm(test_vector),'--')
legend('exact kernel', 'warp-SKI kernel')


infg = @(varargin) infGaussLik(varargin{:},opt);

%% Signal separation

% get intermediate result
post_grid = infGaussLik( hyp, {@meanZero}, csu_grid, likfunc, X, Y, opt);
post = infGaussLik( hyp, {@meanZero}, csu, likfunc, X, Y);

% get matrices for separation
hyp1=hyp;
hyp1.cov = hyp.cov(1:3);
hyp2=hyp;
hyp2.cov = hyp2.cov(4:6);
K_grid1 = apx(hyp1,csu_grid{2}{1},X,opt);
K_grid2 = apx(hyp2,csu_grid{2}{2},X,opt);
K1 = feval(covPeWarped_1{:}, hyp_1, X, X);
K2 = feval(covPeWarped_2{:}, hyp_2, X, X);

% do separation
signal_1_grid = K_grid1.mvm(post_grid.alpha);
signal_2_grid = K_grid2.mvm(post_grid.alpha);

signal_1 = K1*post.alpha;
signal_2 = K2*post.alpha;

% plot
figure
subplot(3,1,1)
plot(X,Y)
title('Measured')
subplot(3,1,2)
hold on
plot(X,Y_orig_1)
plot(X,signal_1)
plot(X,signal_1_grid,'--')
legend('original','batch','warp-SKI')
title('Source 1')
subplot(3,1,3)
hold on
plot(X,Y_orig_2)
plot(X,signal_2)
plot(X,signal_2_grid,'--')
title('Source 2')
legend('original','batch','warp-SKI')

%% Check log determinant
disp('Log determinant via Lanczos:')
disp(ldB2)

[ldB2,solveKiW,dW,dhyp,post.L] = K.fun(W);

disp('Log determinant via dense solution:')
disp(ldB2)

%% Check log marginal likelihood
[post_grid, nlZ_grid, dnlZ_grid] = infGaussLik(hyp, {@meanZero}, csu_grid, likfunc, X, Y, opt);
[post_dense, nlZ_dense, dnlZ_dense] = infGaussLik(hyp, {@meanZero}, csu, likfunc, X, Y, opt);

disp(['Lanczos Marg. Lik.: ', num2str(nlZ_grid)])
disp(['Exact Marg. Lik.: ', num2str(nlZ_dense)])

disp('Lanczos derivative:')
sprintf('%f, ', dnlZ_grid.cov)
disp('Exact derivative:')
sprintf('%f, ', dnlZ_dense.cov)

%% Learn Hyperparameters
prior.cov  = {[];{'priorDelta'};[];[];{'priorDelta'};[]}; % fix phase parameters
prior.lik = {'priorDelta'};
infg_prior = {@infPrior,infg,prior};

hyp0 = hyp;
hyp0.cov = [2, log(0.5), 2, 2, log(0.5), 2];

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

