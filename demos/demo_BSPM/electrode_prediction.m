% ELECTRODE_PREDICTION
% 
% Description
%   Run electrode prediction on the body surface potential dataset from
%   Nijmegen. The results are stored into 'paper_results/results_bsp/v4/',
%   which is where you can generate the figures from the paper by running
%   'plot_results.m'.
%   Three of the 65 electrodes are excluded (due to some erratic
%   measurements) and then six of the electrodes are predicted via warpSKI.
%
% References:
%   Grasshoff, J., Jankowski, A. and Rostalski, P. (2019). Scalable Gaussian
%   Process Regression for Kernels with a non-stationary phase,
%   arxiv.org/abs/1912.11713

%% Load data

load('demos/demo_BSPM/Nijmegen-2004-12-09/Interventions/Noisy_room/PPD2_sessie1_exp_04.mat')
pots([1,2,61],:) = [];

% downsample signals from (presumably) 1000Hz to 200Hz
pots = pots(:,1:5:end);

% artificial time vector
dt = 1/200;
t = dt*(1:size(pots, 2))';

% take second half of dataset
pots = pots(:,800:end);
t = t(800:end);

figure
plot(pots(1:10,:)')

% 3D mesh
load('demos/demo_BSPM/Nijmegen-2004-12-09/Meshes/model_65lead.mat')

nodes = geom.node;
faces = geom.face;

% electrode locations
load('demos/demo_BSPM/Nijmegen-2004-12-09/Meshes/Electrode_LOC_65L.mat')
pts = geom.pts;

% exclude electrodes
pts([1,2,61],:) = [];

% Project points to polar coordinates
Z = pts;
X = project_to_2D(Z);

% plot points in 2D
figure
hold on
for i=1:size(X,1)
    plot(X(i,1),X(i,2), 'o')
    text(X(i,1),X(i,2),num2str(i))
end
set(gca, 'Xdir', 'reverse')

% remove 6 random electrodes for prediction purposes
rng(4321)
el1 = 21 + randsample(7,1);
el2 = 14 + randsample(7,1);
el3 = 7 + randsample(7,1);
el4 = 28 + randsample(7,1);
el5 = 35 + randsample(7,1);
el6 = 42 + randsample(7,1);

to_remove = [el1, el2, el3, el4, el5, el6];

% remove 
X_train = X;
X_train(to_remove, :) = [];

% Data
pots_train = pots;
pots_train(to_remove, :) = [];
Y = pots_train(:);

% plot 3D model
figure
hold on
s = trisurf(faces,nodes(:,1),nodes(:,2),nodes(:,3),'FaceColor', 'interp', 'EdgeColor', 'none');
alpha 0.5
plot3(pts(:,1), pts(:,2), pts(:,3), '.g', 'MarkerSize',20)
plot3(pts(to_remove,1), pts(to_remove,2), pts(to_remove,3), '.r', 'MarkerSize', 20)
for i=1:length(to_remove)
    text(pts(to_remove(i),1), pts(to_remove(i),2), pts(to_remove(i),3), num2str(to_remove(i)))
end
light('Position',[100.0,-100.0,100.0],'Style','infinite');
lighting phong;
axis equal

%% Do spatio-temporal regression

tau_indizes = [623, 1474, 2340, 3246, 4202, 5161, 6089, 7005, 7895, 8793, 9674];
tau = tau_indizes/1000;

% Likelihood
sn = 0.15;
hyp.lik = log(sn);

% kernel
p = @(t) (timeWrap( t, tau ));
dp = @(t) (dtimeWrap( t, tau));
covPe = {@covProd,{@covSEisoU, @covPeriodic}};
covPeWarped = {@covWarp,covPe,p,dp,1};

U_rad = 2*pi*(-0.75:0.015:10.5)';
U_t = invTimeWrap( U_rad, tau);
U_dim1 = (-pi:0.1:pi)';
U_dim2 = (-4:0.1:0)';
%U_dim2 = (-5:0.1:3)';

xg = {{U_dim1},{U_dim2},{U_t}};

kernel_SKI = {@apxGrid, {@covPeriodic, @covSEisoU, covPeWarped}, xg};

% Likelihood and Mean
likfunc = @likGauss;
hyp.mean = [];

% Hyperparameters
ell = 1; sf = 0.5; p = 2*pi; ell_SE_temp = 7.5; ell_Pe_temp = 0.1; p_temp = 2*pi; sf_temp = 1;
hyp.cov = log([ell; p; sf; ell; ell_SE_temp; ell_Pe_temp; p_temp; sf_temp]);

X_space = repmat(X_train, length(t), 1);
X_t = kron(t, ones(size(X_train,1),1));

X_all = [X_space, X_t];

% do learning
opt.cg_maxit = 2000; opt.cg_tol = 0.1;
opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 20;
opt.ldB2_maxit = 100;

% priors: first test (no prior on sf), second test (prior on sf)
disp('Start Optimization')
prior.cov = {{@priorGauss,log(1),0.1^2}, {@priorDelta}, {@priorGauss,log(0.75),0.1^2}, {@priorGauss,log(1),0.1^2},...
    {@priorDelta}, {@priorDelta}, {@priorDelta}, {@priorDelta}};

prior.lik = {{@priorDelta}};

infg = @(varargin) infGaussLik(varargin{:},opt);
inf = {@infPrior,infg,prior};
tic
[hyp_opt_lbfgs_gpml, fX, iter] = minimize_lbfgsb(hyp, @gp, -40, inf, [], kernel_SKI, likfunc, X_all, Y);
elapsed_time_learning = toc;
disp(['Time for Hyp-Learning: ', num2str(elapsed_time_learning)]);

hyp = hyp_opt_lbfgs_gpml;

version = 'v4';

save(['paper_results/results_BSPM/', version ,'/hyperparameters.mat'], 'hyp_opt_lbfgs_gpml');

% Do regression
opt_inf.cg_maxit = 2500; opt_inf.cg_tol = 0.01; 
opt_inf.ldB2_method = 'lancz'; opt_inf.ldB2_hutch = 0; opt_inf.ldB2_maxit = 0;

tic
post = infGaussLik(hyp, {@meanZero}, kernel_SKI, {@likGauss}, X_all, Y, opt_inf);
elapsed_time_inference = toc;

save(['paper_results/results_BSPM/', version ,'/regression.mat'],  'post', 'kernel_SKI',...
    'X_all', 'Y', 'elapsed_time_learning', 'iter', 'elapsed_time_inference', 't', 'X', 'to_remove')

% predict all electrodes (including the removed ones)
% K = apx(hyp, kernel_SKI, X_all, opt_inf);
% ymu = K.mvm(post.alpha);
Xtest = X;
X_test_space = repmat(Xtest, length(t), 1);
X_t = kron(t, ones(size(Xtest,1),1));
X_test_all = [X_test_space, X_t];
[Mx_pred1,~] = apxGrid('interp',kernel_SKI{3},X_test_all,3);
[Kg1,Mx1] = feval(kernel_SKI{:},hyp.cov,X_all,[],3);

ymu = Mx_pred1*Kg1.mvm(Mx1'*post.alpha);

pots_test_mu = reshape(ymu, size(Xtest,1), length(t));

%% plot results (over time)
%electrode_indices = [1,10,20,30,40,50,60];
electrode_indices = to_remove;
num_plots = length(electrode_indices);

figure
for i=1:num_plots
    subplot(num_plots,1,i)
    hold on
    title(num2str(electrode_indices(i)))
    plot(t,pots(electrode_indices(i),:))
    plot(t,pots_test_mu(electrode_indices(i),:))
end

%% generate images (for every 4th time step --> corresponding to 20 Hz)
t_50Hz = t(1:4:end);
xtest1 = -pi:0.1:pi;
xtest2 = -5:0.1:3; %xtest2 = -4:0.1:0;
[Xtest1, Xtest2, Xtest3] = meshgrid(xtest1, xtest2, t_50Hz);
Xtest = [Xtest1(:), Xtest2(:), Xtest3(:)];

% get full spatial regression result
U_dim2_ = (-5:0.1:3)';
xg_2 = {{U_dim1},{U_dim2_},{U_t}};
kernel_SKI_2 = {@apxGrid, {@covPeriodic, @covSEisoU, covPeWarped}, xg_2};
[Mx_pred1,~] = apxGrid('interp',kernel_SKI_2{3},Xtest,3);
[Kg1,Mx1] = feval(kernel_SKI_2{:},hyp.cov,X_all,[],3);

ymu = Mx_pred1*Kg1.mvm(Mx1'*post.alpha);

images = reshape(ymu, length(xtest2), length(xtest1), length(t_50Hz));

pots_50Hz = pots(:,1:4:end);

% plot spatial results (for selected times)
time_indices = [50, 52, 54, 56, 58, 60]*3;
num_plots = ceil(sqrt(length(time_indices)));

figure
k=1;
for i=1:num_plots
    for j=1:num_plots
        if (k <= length(time_indices))
            subplot(num_plots, num_plots, k)
            plot3(X(:,1),X(:,2), pots_50Hz(:,time_indices(k))', 'or')
            hold on
            surf(Xtest1(:,:,time_indices(k)), Xtest2(:,:,time_indices(k)), images(:,:,time_indices(k)))
        end
        k = k + 1;
    end
end

%% plot in 3D (with higher time resolution)
nodes_2D = project_to_2D(nodes);

t_100Hz = t(1:2:end);

X_nodes_space = repmat(nodes_2D, length(t), 1);
X_t = kron(t, ones(size(nodes_2D,1),1));

X_nodes_all = [X_nodes_space, X_t];

% make inducing points with higher support domain
% U_dim2_ = (-5:0.1:3)';
% xg_2 = {{U_dim1},{U_dim2_},{U_t}};
% kernel_SKI_2 = {@apxGrid, {@covPeriodic, @covSEisoU, covPeWarped}, xg_2};

[Mx_pred1,~] = apxGrid('interp', xg_2, X_nodes_all, 3);
[Kg1,Mx1] = feval(kernel_SKI_2{:},hyp.cov,X_all,[],3);

ymu = Mx_pred1*Kg1.mvm(Mx1'*post.alpha);

pots_nodes = reshape(ymu, size(nodes_2D,1), length(t));

figure
k=1;
for i=1:num_plots
    for j=1:num_plots
        if (k <= length(time_indices)) % time_indices are off here due to other sampling rate
            subplot(num_plots, num_plots, k)
            hold on
            %c = randn(length(nodes(:,3)),1);
            c = pots_nodes(:,time_indices(k));
            s = trisurf(faces,nodes(:,1),nodes(:,2),nodes(:,3),c, 'FaceColor', 'interp', 'EdgeColor', 'none');
            plot3(pts(:,1), pts(:,2), pts(:,3), '.r', 'MarkerSize',20)
            light('Position',[100.0,-100.0,100.0],'Style','infinite');
            lighting phong;
            axis equal
        end
        k = k + 1;
    end
end

% store video
file = ['paper_results/results_BSPM/', version ,'/torso.avi'];
outputVideo = VideoWriter(fullfile(file));
outputVideo.Quality = 99;
outputVideo.FrameRate = 50;
open(outputVideo)

figure;
for i = 1:length(t)
    
    
    %c = randn(length(nodes(:,3)),1);
    c = pots_nodes(:,i);
    s = trisurf(faces,nodes(:,1),nodes(:,2),nodes(:,3),c, 'FaceColor', 'interp', 'EdgeColor', 'none');
    hold on
    caxis([-0.8,1.7])
    plot3(pts(:,1), pts(:,2), pts(:,3), '.r', 'MarkerSize',20)
    light('Position',[100.0,-100.0,100.0],'Style','infinite');
    lighting phong;
    view(32.15,8.25)
    axis equal
    frame = getframe(gcf);
    writeVideo(outputVideo,frame);
    %close(f)
    hold off
end

close(outputVideo)

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