% EIT_SIMPLE_SIMULATED_EXAMPLE
% 
% Description
%   This simulation example uses data generated from the true kernel.
%   Therefore no model mismatch can occur here.

%% GP regression toy example (with warping)
x0 = (1:10)'; % Y-axis of image
x1 = (1:10)'; % X-axis of image
t = (1:2:100)'/10;
X = apxGrid('expand',{x0,x1,t});

sigma2_n = 0.2;

% dimension 1 (spatial)
ell_spat1 = 1.5;
cov_spatial = @covSEisoU;
msk1 = [1,0,0];
cov_dim1 = {@covMask,{msk1,cov_spatial}};

% dimension 2 (spatial)
ell_spat2 = 1.5;
cov_spatial = @covSEisoU;
msk2 = [0,1,0];
cov_dim2 = {@covMask,{msk2,cov_spatial}};

% dimension 3 first kernel (time)
tau1 = [0,3.5,7,9,11];
p = @(t_) (timeWrap( t_, tau1));
dp = @(t_) (dtimeWrap( t_, tau1));
ell_SE_time = 15; ell_PE_time = 2; p_time=2*pi; sf_time = 1;
cov_time = {@covProd,{@covSEisoU, @covPeriodic}};
cov_warped_time_1 = {@covWarp, cov_time, p, dp, 1};
msk3 = [0,0,1];
cov_dim3 = {@covMask,{msk3,cov_warped_time_1}};

hyp.cov = log([ell_spat1, ell_spat2, ell_SE_time, ell_PE_time, p_time, sf_time]);

% product
covfunc_1 = {@covProd, {cov_dim1, cov_dim2, cov_dim3}};

% dimension 3 second kernel (time)
tau2 = [0,5,10];
p = @(t_) (timeWrap( t_, tau2));
dp = @(t_) (dtimeWrap( t_, tau2));
ell_SE_time = 15; ell_PE_time = 2; p_time=2*pi; sf_time = 2;
cov_time = {@covProd,{@covSEisoU, @covPeriodic}};
cov_warped_time_2 = {@covWarp, cov_time, p, dp, 1};
msk3 = [0,0,1];
cov_dim3 = {@covMask,{msk3,cov_warped_time_2}};
hyp.cov = [hyp.cov, log([ell_spat1, ell_spat2, ell_SE_time, ell_PE_time, p_time, sf_time])];

% product
covfunc_2 = {@covProd, {cov_dim1, cov_dim2, cov_dim3}};

% sum
csu = {'covSum', {covfunc_1, covfunc_2}};

% likelihood function
likfunc = @likGauss;
hyp.lik = log(sqrt(sigma2_n));

Kxx = feval(csu{:},hyp.cov,X,X); % covfunc

% make test data
Y_orig = mvnrnd(zeros(size(Kxx,1),1),Kxx)';
Y = Y_orig + sqrt(sigma2_n)*randn(size(Y_orig));

Y_orig_imgs = reshape(Y_orig, [length(x0), length(x1), length(t)]);
Y_imgs = reshape(Y, [length(x0), length(x1), length(t)]);

% GP regression dense
hyp.mean=[];
post = infGaussLik( hyp, {@meanZero}, csu, likfunc, X, Y);

% get matrices for separation
K1 = feval(covfunc_1{:}, hyp.cov(1:6), X, X);
K2 = feval(covfunc_2{:}, hyp.cov(7:end), X, X);

% do separation
signal_1 = K1*post.alpha;
signal_2 = K2*post.alpha;
ymu = signal_1 + signal_2;

Y_mu_imgs = reshape(ymu, [length(x0), length(x1), length(t)]);
Y_mu_imgs_sign_1 = reshape(signal_1, [length(x0), length(x1), length(t)]);
Y_mu_imgs_sign_2 = reshape(signal_2, [length(x0), length(x1), length(t)]);

Y_imgs_all = horzcat(Y_imgs, Y_orig_imgs, Y_mu_imgs);

implay(mat2gray(Y_imgs_all, [max(Y_imgs_all(:)), min(Y_imgs_all(:))]), 13);

figure
hold on
plot(Y)
plot(Y_orig)
plot(ymu)

%% GP regression SKI
% create inducing points
U_dim0 = apxGrid('create',x0,true,[12]);
U_dim1 = apxGrid('create',x1,true,[12]);

U_rad = 2*pi*(-1:0.05:(length(tau1)+1))';
U_time = invTimeWrap( U_rad, tau1);
xg = {{U_dim0{1}},{U_dim1{1}},{U_time}};
covGrid_1 = {@apxGrid, {{cov_spatial}, {cov_spatial}, cov_warped_time_1}, xg};

U_rad = 2*pi*(-1:0.05:(length(tau2)+1))';
U_time = invTimeWrap( U_rad, tau2);
xg = {{U_dim0{1}},{U_dim1{1}},{U_time}};
covGrid_2 = {@apxGrid, {{cov_spatial}, {cov_spatial}, cov_warped_time_2}, xg};

csu_grid = {'apxGridSum',{covGrid_1, covGrid_2}};

%[Kg,Mx] = feval(covGrid{:},hyp.cov,X,[],3);

% Options
opt.cg_maxit = 600; opt.cg_tol = 5e-3;
opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 40;
hyp.mean = [];

%% Do inference
post = infGaussLik(hyp, {@meanZero}, csu_grid, {@likGauss}, X, Y, opt);

hyp1=hyp;
hyp1.cov = hyp.cov(1:6);
hyp2=hyp;
hyp2.cov = hyp2.cov(7:end);
K_grid1 = apx(hyp1,csu_grid{2}{1},X,opt);
K_grid2 = apx(hyp2,csu_grid{2}{2},X,opt);

% do separation
signal_1_grid = K_grid1.mvm(post.alpha);
signal_2_grid = K_grid2.mvm(post.alpha);
mu_grid = signal_1_grid + signal_2_grid;

Y_mu_grid_imgs_sign_1 = reshape(signal_1_grid, [length(x0), length(x1), length(t)]);
Y_mu_grid_imgs_sign_2 = reshape(signal_2_grid, [length(x0), length(x1), length(t)]);

Y_mu_grid_imgs = reshape(mu_grid, [length(x0), length(x1), length(t)]);
Y_imgs_all = horzcat(Y_imgs, Y_orig_imgs, Y_mu_imgs, Y_mu_grid_imgs);
implay(mat2gray(Y_imgs_all, [max(Y_imgs_all(:)), min(Y_imgs_all(:))]), 13);

figure
subplot(3,1,1)
hold on
plot(t,squeeze(Y_imgs(5,5,:)))
plot(t,squeeze(Y_orig_imgs(5,5,:)))
plot(t,squeeze(Y_mu_imgs(5,5,:)))
plot(t,squeeze(Y_mu_grid_imgs(5,5,:)))
legend('noisy signal', 'original signal', 'dense GP', 'SKI GP')
subplot(3,1,2)
hold on
plot(t,squeeze(Y_mu_imgs_sign_1(5,5,:)))
plot(t,squeeze(Y_mu_grid_imgs_sign_1(5,5,:)))
legend('dense GP', 'SKI GP')
subplot(3,1,3)
hold on
plot(t,squeeze(Y_mu_imgs_sign_2(5,5,:)))
plot(t,squeeze(Y_mu_grid_imgs_sign_2(5,5,:)))
legend('dense GP', 'SKI GP')

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

% Toeplitz Preconditioner according to Chan.
% Code by Michela Redivo-Zaglia and Giuseppe Rodriguez.
function c = chan_preconditioner(k)
n = length(k);
c = zeros(n,1);
t = [flipud(k);k(2:end)];
c(1) = t(n);
iv = (1:n-1)';
c(iv+1) = ((n-iv).*t(iv+n) + iv.*t(iv)) / n;
end

function y = kron_mvm(K1,K2,x)
n = size(K1,1);
m = size(K2,2);
x_mat = reshape(x,m,n);
y_mat = K2*(x_mat*K1');
y=y_mat(:);
end