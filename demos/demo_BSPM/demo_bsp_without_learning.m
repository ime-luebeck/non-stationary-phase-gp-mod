% DEMO_BSP_WITHOUT_LEARNING
% 
% Description
%   Run electrode prediction on the body surface potential dataset from
%   Nijmegen. In this script, the hyperparameters are not learned but kept 
%   fixed. 
%   Three of the 65 electrodes are excluded (due to some erratic
%   measurements) and then six of the electrodes are predicted via warpSKI.
%
% References:
%   Grasshoff, J., Jankowski, A. and Rostalski, P. (2019). Scalable Gaussian
%   Process Regression for Kernels with a non-stationary phase,
%   arxiv.org/abs/1912.11713

%% Load data
load('demos\demo_BSPM\Nijmegen-2004-12-09\Interventions\Noisy_room\PPD2_sessie1_exp_04.mat')
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
load('demos\demo_BSPM\Nijmegen-2004-12-09\Meshes\model_65lead.mat')

nodes = geom.node;
faces = geom.face;

% electrode locations
load('demos\demo_BSPM\Nijmegen-2004-12-09\Meshes\Electrode_LOC_65L.mat')
pts = geom.pts;

% exclude electrodes
pts([1,2,61],:) = [];

figure
hold on
c = randn(length(nodes(:,3)),1);
s = trisurf(faces,nodes(:,1),nodes(:,2),nodes(:,3),c, 'FaceColor', 'interp', 'EdgeColor', 'none');
plot3(pts(:,1), pts(:,2), pts(:,3), '.r', 'MarkerSize',20)
light('Position',[100.0,-100.0,100.0],'Style','infinite');
lighting phong;
axis equal

% Project points to polar coordinates
Z = pts;
X = project_to_2D(Z);

figure
plot(X(:,1),X(:,2), 'o')

%% Do simple spatial (batch) interpolation

% kernel definition
msk1 = [1,0];
cov_dim1 = {@covMask, {msk1, @covPeriodic}};
msk2 = [0,1];
cov_dim2 = {@covMask, {msk2, @covSEisoU}};
covfunc = {@covProd, {cov_dim1, cov_dim2}};
ell = 0.6; sf = 0.5; p = 2*pi;
hyp.cov = log([ell; p; sf; ell]);
likfunc = @likGauss;
sn = 0.15;
hyp.lik = log(sn);

xtest1 = -pi:0.1:pi;
xtest2 = -5:0.1:3;
[Xtest1, Xtest2] = meshgrid(xtest1, xtest2);
Xtest = [Xtest1(:), Xtest2(:)];

Y = pots(:,422);

[ymu ys2 fmu fs2] = gp(hyp, @infExact, [], covfunc, likfunc, X, Y, Xtest);

im = reshape(ymu, length(xtest2), length(xtest1));

figure
hold on
imagesc(xtest1, xtest2, im)
plot(X(:,1),X(:,2), 'o')
colorbar

figure
plot3(X(:,1),X(:,2), Y, 'or')
hold on
surf(Xtest1, Xtest2, im)

%% Do spatial (SKI) regression

U_dim1 = (-pi:0.1:pi)';
U_dim2 = (-4:0.1:0)';

xg = {{U_dim1},{U_dim2}};

kernel_SKI = {@apxGrid, {@covPeriodic, @covSEisoU}, xg};

opt_inf.cg_maxit = 2500; opt_inf.cg_tol = 0.05; 
opt_inf.ldB2_method = 'lancz'; opt_inf.ldB2_hutch = 0; opt_inf.ldB2_maxit = 0;

hyp.mean = [];
post = infGaussLik(hyp, {@meanZero}, kernel_SKI, {@likGauss}, X, Y, opt_inf);

% get regression result
U_dim2_ = (-5:0.1:3)';
xg_2 = {{U_dim1},{U_dim2_}};
kernel_SKI_2 = {@apxGrid, {@covPeriodic, @covSEisoU}, xg_2};
[Mx_pred1,~] = apxGrid('interp',kernel_SKI_2{3},Xtest,3);
[Kg1,Mx1] = feval(kernel_SKI_2{:},hyp.cov,X,[],3);

ymu = Mx_pred1*Kg1.mvm(Mx1'*post.alpha);

im = reshape(ymu, length(xtest2), length(xtest1));

figure
plot3(X(:,1),X(:,2), Y, 'or')
hold on
surf(Xtest1, Xtest2, im)

%% Do temporal regression
tau_indizes = [623, 1474, 2340, 3246, 4202, 5161, 6089, 7005, 7895, 8793, 9674];

tau = tau_indizes/1000;

% create non-equidistant inducing points: U = phi^-1(U_rad)
%U_rad = 2*pi*(-0.75:0.015:10.5)';
U_rad = 2*pi*(3.5:0.015:10.5)';

U = invTimeWrap( U_rad, tau);
xg = {{U}}; % grid

% plot inducing points
figure
plot(t, pots(1,:))
hold on
plot(U, zeros(size(U)),'o')

% make kernel
likfunc = @likGauss;
%covPe = {@covPeriodic};
covPe = {@covProd,{@covSEisoU, @covPeriodic}};
p = @(t) (timeWrap( t, tau ));
dp = @(t) (dtimeWrap( t, tau));
covPeWarped = {@covWarp,covPe,p,dp,1};
temporal_kernel = {@apxGrid,{covPeWarped},xg};

hyp.cov = log([8; 0.1; 2*pi; 0.5]);
hyp.lik = log(sn);
hyp.mean = [];

opt_inf.cg_maxit = 2500; opt_inf.cg_tol = 0.001; 
opt_inf.ldB2_method = 'lancz'; opt_inf.ldB2_hutch = 0; opt_inf.ldB2_maxit = 0;

X = t;
Y = pots(1,:)';

hyp.mean = [];
post = infGaussLik(hyp, {@meanZero}, temporal_kernel, {@likGauss}, X, Y, opt_inf);

K = apx(hyp, temporal_kernel, X, opt_inf);
ymu = K.mvm(post.alpha);

figure
plot(t,ymu)
hold on
plot(t, Y)

%% Do spatio-temporal regression

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

% Data
Y = pots(:);
X = project_to_2D(Z);

X_space = repmat(X, length(t), 1);
X_t = kron(t, ones(size(X,1),1));

X_all = [X_space, X_t];

% Do regression
opt_inf.cg_maxit = 2500; opt_inf.cg_tol = 0.01; 
opt_inf.ldB2_method = 'lancz'; opt_inf.ldB2_hutch = 0; opt_inf.ldB2_maxit = 0;

post = infGaussLik(hyp, {@meanZero}, kernel_SKI, {@likGauss}, X_all, Y, opt_inf);

K = apx(hyp, kernel_SKI, X_all, opt_inf);
ymu = K.mvm(post.alpha);

pots_mu = reshape(ymu, size(Z,1), length(t));

% plot results (over time)
electrode_indices = [1,10,20,30,40,50,60];
num_plots = length(electrode_indices);

figure
for i=1:num_plots
    subplot(num_plots,1,i)
    hold on
    plot(t,pots(electrode_indices(i),:))
    plot(t,pots_mu(electrode_indices(i),:))
end

% generate images (for every 4th time step --> corresponding to 20 Hz)
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
file = 'torso.avi';
outputVideo = VideoWriter(fullfile(file));
outputVideo.Quality = 99;
outputVideo.FrameRate = 50;
open(outputVideo)

f = figure;
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