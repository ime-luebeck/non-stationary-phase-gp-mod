% DEMO_NON_STATIONARY_1D_GRIDBASED
% 
% Description
%   Simple example of warped structured kernel interpolation for a 1D
%   non-stationary kernel.
% 
% References:
%   Grasshoff, J., Jankowski, A. and Rostalski, P. (2019). Scalable Gaussian
%   Process Regression for Kernels with a non-stationary phase,
%   arxiv.org/abs/1912.11713

%%
% Prepare dataset: equidistant data with phase-warped covariance.
% Between consecutive elements in tau, there is a phase difference of 2*pi.
tau = [1,2,3,5,7,9,11,13,14,15,16,17,18]; 
X = (0:0.05:16)';
n = length(X);
t_rad = timeWrap( X, tau );
dt_rad = dtimeWrap( X, tau);
sigma2_n = 0.2^2;

% Create non-equidistant inducing points: U = phi^-1(U_rad)
U_rad = 2*pi*(-2:0.025:11)'; % 0.05
U = invTimeWrap( U_rad, tau);
xg = {{U}}; % grid
[Mx,~] = apxGrid('interp',xg,X,3);

% Build covariance function using GPML functions.
likfunc = @likGauss;
covPe = {@covPeriodic};
p = @(t) (timeWrap( t, tau ));
dp = @(t) (dtimeWrap( t, tau));
covPeWarped = {@covWarp,covPe,p,dp,1};
covGrid = {@apxGrid,{covPeWarped},xg};

hyp.cov = log([1; 2*pi; 1]);
hyp.lik = log([sqrt(sigma2_n)]);
hyp.mean = [];

% Evaluate the covariance matrices.
[K,dK] = feval(covPeWarped{:}, hyp.cov, X, X);

% Pull random sample from dense GP.
randn('seed',56767857456)
Y_orig = mvnrnd(zeros(n,1),K)';
Y = Y_orig + sqrt(sigma2_n)*randn(size(Y_orig)); % add measurement noise

% Options
opt.cg_maxit = 170; opt.cg_tol = 5e-3; 
opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 25;

% do regression
infg = @(varargin) infGaussLik(varargin{:},opt); % one could switch off variance calculations here
[mu_grid, s2_grid] = gp(hyp, infg, [], covGrid, likfunc, X, Y, X);
[mu_dense, s2_dense] = gp(hyp, infg, [], covPeWarped, likfunc, X, Y, X);
figure
hold on
plot(X,Y)
plot(X,mu_dense)
plot(X,mu_grid,'--')
legend('Exact', 'Grid-based')

% learn hyperparameters
hyp0 = hyp;
hyp0.cov = log([0.1; 2*pi; 0.1]);
prior.cov  = {[];{'priorDelta'};[]}; % fix phase parameters
prior.lik = {'priorDelta'};
infg_prior = {@infPrior,infg,prior};
hyp_opt = minimize(hyp0, @gp, -100, infg_prior, [], covPeWarped, likfunc, X, Y);
hyp_opt_g = minimize(hyp0, @gp, -100, infg_prior, [], covGrid, likfunc, X, Y);

disp(['Exact Hyperparameters: ', mat2str(exp(hyp.cov))])
disp(['Hyperparamters from dense matrix: ', mat2str(exp(hyp_opt.cov))])
disp(['Hyperparameters from SKI: ', mat2str(exp(hyp_opt_g.cov))])

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