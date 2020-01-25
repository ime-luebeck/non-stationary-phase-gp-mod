% EIT_DEMO
% 
% Description
%   Run perfusion/ventilation separation on the full EIT test dataset via
%   warpSKI. In this script the hyperparameters are not optimized but kept
%   fixed. The results are stored into 'paper_results/results_eit/v1/',
%   which is where you can generate the figures from the paper by running
%   'plot_results_eit_v1.m'.
%
%   Run eit_load_data.m before running this script. 
%
% References:
%   Grasshoff, J., Jankowski, A. and Rostalski, P. (2019). Scalable Gaussian
%   Process Regression for Kernels with a non-stationary phase,
%   arxiv.org/abs/1912.11713
%% Load data
load('./demos/demo_EIT/if-neonate-spontaneous/eit_imgs.mat')
fs = 13;
time = ((1:size(imgs,3))-1)/fs;

% plot data
yposns = [45  19 44]; xposns = [50  38 22]; 
figure
subplot(1,2,1)
hold on
axis equal
imagesc(imgs(:,:,1))
for i = 1:length(xposns)
    plot(xposns(i),yposns(i),'s','LineWidth',5);
end

subplot(1,2,2)
hold on
for i = 1:length(xposns)
    plot(squeeze(imgs(yposns(i),xposns(i),:)))
end

%% preprocess data

% remove last faulty images
imgs = imgs(:,:,1:215);
time = time(1:215);

% remove fixed baseline from all pixels
M = movmean(imgs,50,3);
imgs = imgs - M;

% normalize values
normalizer = max(max(rms(imgs,3)));
imgs = imgs/normalizer;

% estimate respiratory rate
x = 22; y = 44;
resp_signal = squeeze(imgs(y,x,:));
time_ups = ((1:3*size(imgs,3))-1)/(3*fs);
resp_signal = interp1(time, resp_signal, time_ups, 'pchip');
resp_gates = 1*(resp_signal>0);
in_s = find(diff([0;resp_gates(:);0])>0.5);
in_e = find(diff([0;resp_gates(:);0])<-0.5);

tau_resp = [];
for i=1:length(in_s)
    resp_segm = resp_signal(in_s(i):in_e(i));
    resp_segm(resp_segm<median(resp_segm)) = 0;
    center_of_mass = sum(time_ups(in_s(i):in_e(i)).*resp_segm)/sum(resp_segm);
    tau_resp = [tau_resp, center_of_mass];
end

% estimate heart rate
x = 38; y = 19;
card_signal = squeeze(imgs(y,x,:));
time_ups = ((1:3*size(imgs,3))-1)/(3*fs);
card_signal = interp1(time, card_signal, time_ups, 'pchip');
card_gates = 1*(card_signal>0);
in_s = find(diff([0;card_gates(:);0])>0.5);
in_e = find(diff([0;card_gates(:);0])<-0.5);

tau_card = [];
for i=1:length(in_s)
    card_segm = card_signal(in_s(i):in_e(i));
    card_segm(card_segm<median(card_segm)) = 0;
    center_of_mass = sum(time_ups(in_s(i):in_e(i)).*card_segm)/sum(card_segm);
    tau_card = [tau_card, center_of_mass];
end

figure
subplot(2,1,1)
hold on
plot(time_ups,resp_signal)
plot(tau_resp, zeros(size(tau_resp)),'o')
subplot(2,1,2)
hold on
plot(time_ups,card_signal)
plot(tau_card, zeros(size(tau_card)),'o')

% plot data
yposns = [45  19 44]; xposns = [50  38 22]; 
figure
subplot(1,2,1)
hold on
axis equal
imagesc(imgs(:,:,1))
for i = 1:length(xposns)
    plot(xposns(i),yposns(i),'s','LineWidth',5);
end

subplot(1,2,2)
hold on
for i = 1:length(xposns)
    plot(time,squeeze(imgs(yposns(i),xposns(i),:)))
end

implay(mat2gray(imgs, [max(imgs(:)), min(imgs(:))]), 13);
implay(mat2gray(M, [max(M(:)), min(M(:))]), 13);

%% create kernels etc.
x0 = (1:64)'; % Y-axis of image
x1 = (1:64)'; % X-axis of image
X = apxGrid('expand',{x0,x1,time'});
Y = imgs(:);

imgs_rec = reshape(Y, [length(x0), length(x1), length(time)]);
imgs_comp = horzcat(imgs, imgs_rec);
implay(mat2gray(imgs_comp, [max(imgs_rec(:)), min(imgs_rec(:))]), 13);

X(isnan(Y),:) = [];
Y(isnan(Y)) = [];

% GP model
sigma2_n = 0.2;

% kernel definition
% dimension 1 + 2 (spatial)
ell_spat1 = 10; ell_spat2 = 10; % pixel
cov_spatial = @covSEisoU;
cov_dim1 = {@covMask, {[1,1,0], cov_spatial}};

% dimension 3 resp kernel (time)
p = @(t_) (timeWrap( t_, tau_resp)); dp = @(t_) (dtimeWrap( t_, tau_resp));
ell_SE_time = 10; ell_PE_time = 1; p_time=2*pi; sf_time = 2;
cov_warped_time_resp = {@covWarp, {@covProd,{@covSEisoU, @covPeriodic}}, p, dp, 1};
cov_dim3 = {@covMask,{[0,0,1],cov_warped_time_resp}};
hyp.cov = log([ell_spat1, ell_SE_time, ell_PE_time, p_time, sf_time]);

covfunc_resp = {@covProd, {cov_dim1, cov_dim3}};

% dimension 3 card kernel (time)
p = @(t_) (timeWrap( t_, tau_card)); dp = @(t_) (dtimeWrap( t_, tau_card));
ell_SE_time = 10; ell_PE_time = 0.5; p_time=2*pi; sf_time = 1;
cov_warped_time_card = {@covWarp, {@covProd,{@covSEisoU, @covPeriodic}}, p, dp, 1};
cov_dim3 = {@covMask, {[0,0,1], cov_warped_time_card}};

hyp.cov = [hyp.cov, log([ell_spat1, ell_SE_time, ell_PE_time, p_time, sf_time])];

covfunc_card = {@covProd, {cov_dim1, cov_dim3}};

% likelihood function
likfunc = @likGauss;
hyp.lik = log(sqrt(sigma2_n));

% create equidistant/non-equidistant inducing points
U_dim0 = apxGrid('create',x0,true,[50]);
U_dim1 = apxGrid('create',x1,true,[50]);
U_rad = 2*pi*(-1:0.01:(length(tau_resp)+1))';
U_time = invTimeWrap( U_rad, tau_resp);
xg = {{U_dim0{1},U_dim1{1}},{U_time}};
covGrid_resp = {@apxGrid, {{cov_spatial}, cov_warped_time_resp}, xg};

U_rad = 2*pi*(-1:0.05:(length(tau_card)+1))';
U_time = invTimeWrap( U_rad, tau_card);
xg = {{U_dim0{1},U_dim1{1}},{U_time}};
covGrid_card = {@apxGrid, {{cov_spatial}, cov_warped_time_card}, xg};

% full kernel
csu_grid = {'apxGridSum',{covGrid_resp, covGrid_card}};

hyp.mean = [];

%% Report MVM time
K = apx(hyp, csu_grid, X);
test_vec = randn(size(X,1),1);
tic
res = K.mvm(test_vec);
elapsed_time_mvm = toc;
disp(['Time for MVM: ', num2str(elapsed_time_mvm)]);

%% Hyperparameters (not optimized)
hyp_opt_lbfgs_gpml = hyp;
%% do regression
opt_inf.cg_maxit = 2500; opt_inf.cg_tol = 0.01; %opt.cg_tol = 5e-3;
opt_inf.ldB2_method = 'lancz'; opt_inf.ldB2_hutch = 0; opt_inf.ldB2_maxit = 0;
tic
post = infGaussLik(hyp_opt_lbfgs_gpml, {@meanZero}, csu_grid, {@likGauss}, X, Y, opt_inf);
elapsed_time = toc;
disp(['Time for Inference: ', num2str(elapsed_time)]);

% get components
hyp1=hyp_opt_lbfgs_gpml;
hyp1.cov = hyp_opt_lbfgs_gpml.cov(1:5);
hyp2=hyp_opt_lbfgs_gpml;
hyp2.cov = hyp2.cov(6:end);
K_grid1 = apx(hyp1,csu_grid{2}{1},X,opt_inf);
K_grid2 = apx(hyp2,csu_grid{2}{2},X,opt_inf);

% do separation
signal_1_grid = K_grid1.mvm(post.alpha);
signal_2_grid = K_grid2.mvm(post.alpha);
mu_grid = signal_1_grid + signal_2_grid;

%% rebuild and show pictures
Y_all = nan*ones(size(imgs(:)));
Y_all(~isnan(imgs(:))) = mu_grid;
imgs_regression = reshape(Y_all, [length(x0), length(x1), length(time)]);

implay(mat2gray(imgs_regression, [max(imgs_regression(:)), min(imgs_regression(:))]), 13);

Y_1 = nan*ones(size(imgs(:)));
Y_1(~isnan(imgs(:))) = signal_1_grid;
imgs_regression_1 = reshape(Y_1, [length(x0), length(x1), length(time)]);
implay(mat2gray(imgs_regression_1, [max(imgs_regression_1(:)), min(imgs_regression_1(:))]), 13);

Y_2 = nan*ones(size(imgs(:)));
Y_2(~isnan(imgs(:))) = signal_2_grid;
imgs_regression_2 = reshape(Y_2, [length(x0), length(x1), length(time)]);
implay(mat2gray(imgs_regression_2, [max(imgs_regression_2(:)), min(imgs_regression_2(:))]), 13);

% concatenate videos
array_nan = nan*ones(length(x0), 1, length(time));
M = M/normalizer;
imgs_comparison = horzcat(M+imgs, array_nan, M+imgs_regression_1, array_nan, imgs_regression_2);
implay(mat2gray(imgs_comparison, [max(M(:)+imgs(:)), min(M(:)+imgs(:))]), 13);

% plot separate pixels
xposns = [50  30 36]; yposns = [30 15 19];
figure
subplot(length(xposns)+1, 1, 1)
hold on
axis equal
imagesc(M(:,:,71) + imgs(:,:,71))
for i = 1:length(xposns)
    plot(xposns(i),yposns(i),'s','LineWidth',2,'Color','k');
    text(xposns(i)+3,yposns(i),num2str(i))
end
set(gca,'YDir','reverse')

for i = 1:length(xposns)
    subplot(length(xposns)+1,1,i+1)
    hold on
    plot(time,squeeze(M(yposns(i),xposns(i),:) + imgs(yposns(i),xposns(i),:)))
    plot(time, squeeze(M(yposns(i),xposns(i),:) + imgs_regression_1(yposns(i),xposns(i),:)))
    plot(time, squeeze(imgs_regression_2(yposns(i),xposns(i),:)))
    legend('mixed', 'respiratory', 'cardiac');
end

% store results
path = './paper_results/results_eit/v1/';
store_video([path, 'eit_out.avi'], imgs_comparison);
save([path, 'result.mat'],'csu_grid', 'hyp', 'X', 'Y',...
    'post', 'elapsed_time', 'tau_resp', 'tau_card', 'imgs', 'M', 'time', 'fs',...
    'signal_1_grid', 'signal_2_grid')
fileID = fopen([path, 'settings.txt'], 'w');
fprintf(fileID, 'Hyperparameters of Covariance:\n');
fprintf(fileID, '%f; ', hyp.cov);
fprintf(fileID, '\n\nHyperparameters of Likelihood:\n');
fprintf(fileID, '%f; ', hyp.lik);
fprintf(fileID, '\n\nOptions:\nIterations: %f; Tolerance: %f \n', opt_inf.cg_maxit, opt_inf.cg_tol);
fprintf(fileID, '\nTime for Inference: %f; Time for MVM: %f \n', elapsed_time, elapsed_time_mvm);
fprintf(fileID, '\nInducing Point set (resp): %f x %f x %f \n', length(covGrid_resp{3}{1}{1}), length(covGrid_resp{3}{1}{2}), length(covGrid_resp{3}{2}{1}));
fprintf(fileID, '\nInducing Point set (card): %f x %f x %f', length(covGrid_card{3}{1}{1}), length(covGrid_card{3}{1}{2}), length(covGrid_card{3}{2}{1}));
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
