% DEMO_NON_STATIONARY_2D_GRIDBASED
% 
% Description
%   Simple example of warped structured kernel interpolation for a 2D
%   non-stationary kernel.
% 
% References:
%   Grasshoff, J. Jankowski, A. and Rostalski, P. (2019). Scalable Gaussian
%   Process Regression for Kernels with a non-stationary phase,
%   arxiv.org/abs/1912.11713

%% define input data
x0 = linspace(-1.2,0.75,40)';
x1 = linspace(-2.5,2.5,25)';
[X0 X1] = ndgrid(x0,x1);
X = [X0(:), X1(:)]; % does the same as apxGrid('expand')
X = apxGrid('expand',{x0,x1});

tau = [-2.5,0,1]; 

p = @(t) (timeWrap( t, tau ));
dp = @(t) (dtimeWrap( t, tau));


%% Kernel definition with warping (batch kernel and warp-SKI kernel)

% dimension 1
cov_mat1 = @covSEiso;
cov_dim1 = {@covMask,{[1,0],cov_mat1}};

% dimension 2
cov_mat2 = @covSEisoU;
cov_dim2 = {@covMask,{[0,1],cov_mat2}};

% product
covfunc = {@covProd, {cov_dim1,cov_dim2}};

% dimension 1 warped
cov_dim1_warped = {@covWarp,cov_dim1,p,dp,1};

% product warped
covfunc_warped = {@covProd, {cov_dim1_warped,cov_dim2}}; % batch kernel

% hyperparameters
ell = 0.4; sf = 1.5;
hyp.cov = log([ell; sf; ell]);
likfunc = @likGauss;
sn = 1;
hyp.lik = log(sn);

% Inducing points U = phi^-1(U_rad)
U_dim1_rad = 2*pi*(0.2:0.03:2)';
U_dim1 = invTimeWrap( U_dim1_rad, tau);
U_dim2 = apxGrid('create',x1,true,[60]);
U_dim2 = U_dim2{1};
xg = {{U_dim1},{U_dim2}};
xg_rad = {{U_dim1_rad},{U_dim2}};
U = apxGrid('expand',xg);
U_rad = apxGrid('expand',xg_rad);

opt.cg_maxit = 400; opt.cg_tol = 5e-3;

%% Kernel definition without warping (standard SKI)

X = [X0(:), X1(:)]; 

X_tilde = [p(X0(:)), X1(:)];

covfunc_non_warped2 = {cov_mat1,cov_mat2};
covGrid_non_warped = {@apxGrid,covfunc_non_warped2,xg_rad}; % standard SKI kernel (equidistant inducing points)

%% Generate data 

% get higher number of inducing points (to achieve better accuracy for data generation)
U_dim1_rad_dense = 2*pi*(0.2:0.02:2)';
U_dim2_dense = apxGrid('create',x1,true,[90]);
U_dim2_dense = U_dim2_dense{1};
xg_rad_dense = {{U_dim1_rad_dense},{U_dim2_dense}};

% get kernel matrices (dimension wise to exploit Kronecker structure)
K_U = feval(covfunc{:}, hyp.cov, U_rad, U_rad); % full matrix
K_U1 = feval(cov_mat1, hyp.cov(1:2), U_dim1_rad_dense);
[V1,D1] = eig((K_U1+K_U1')/2);
K_U2 = feval(cov_mat2, hyp.cov(3), U_dim2_dense);
[V2,D2] = eig((K_U2+K_U2')/2);

% Sample
test_vec = randn(size(K_U1,1)*size(K_U2,1),1);
sample = real(kron_mvm(V2, V1, kron(real(sqrt(diag(D2))),real(sqrt(diag(D1)))).*test_vec ));

% Warp sample
Mx = apxGrid('interp',xg_rad_dense, X_tilde, 3);
sample = Mx*sample;
Y_orig = sample;

Y = Y_orig+0.5*randn(size(X1(:)));

figure
subplot(1,2,1)
surf(reshape(Y_orig, size(X0)))
title('Noisy data')
subplot(1,2,2)
imagesc(x0, x1, reshape(Y,size(X0)).')
title('Noisy data')

figure
subplot(1,2,1)
surf(p(X0),X1,reshape(Y,size(X0)))
title('Noisy data (dewarped)')
subplot(1,2,2)
surf(X0,X1,reshape(Y,size(X0)))
title('Noisy data')

%% do regression with standard SKI
% do regression (but turn of variance)
hyp.mean = [];
[post,nlZ,dnlZ] = infGrid(hyp, {@meanZero}, covGrid_non_warped, {@likGauss}, X_tilde, Y, opt);

post.L = @(a) zeros(size(a));
infg = @(varargin) infGaussLik(varargin{:},opt);
mu_grid = gp(hyp,infg,[],covGrid_non_warped,[],X_tilde,post,X_tilde);

figure
imagesc(x0, x1, reshape(mu_grid,size(X0)).')
set(gca,'YDir','normal')
title('SKI based regression results')

%% do regression with warp-SKI
% grid based warped covariance function
cov_dim1_warped = {@covWarp,{cov_mat1},p,dp,1};
covfunc_warped2 = {cov_dim1_warped,cov_mat2};
covGrid = {@apxGrid,covfunc_warped2,xg}; % warp-SKI kernel (non-equidistant inducing points)

% do regression (but turn of variance)
hyp.mean = [];
[post,nlZ,dnlZ] = infGrid(hyp, {@meanZero}, covGrid, {@likGauss}, X, Y, opt);

post.L = @(a) zeros(size(a));
infg = @(varargin) infGaussLik(varargin{:},opt);
mu_grid = gp(hyp,infg,[],covGrid,[],X,post,X);

figure
imagesc(x0, x1, reshape(mu_grid,size(X0)).')
set(gca,'YDir','normal')
title('warpSKI based regression results')

%% compare matrix vector multiplication

[Kg_non_warp,Mx_non_warp] = feval(covGrid_non_warped{:}, hyp.cov, X_tilde, [], 3);
[Kg, Mx] = feval(covGrid{:}, hyp.cov, X, [], 3);

test_vector = [1; zeros(length(U_rad)-1,1)];
figure
hold on
plot(Kg.mvm(test_vector))
plot(Kg_non_warp.mvm(test_vector), '--')
legend('SKI', 'warp-SKI')
title('Comparison of MVM with test vector')

%% compare marginal likelihood
hyp_opt = minimize(hyp, @gp, -100, [], [], covfunc_warped, likfunc, X, Y); % search optimum by dense method

opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 20;
opt.cg_maxit = 300; opt.cg_tol = 0.1;

x_vec = -1.5:0.05:-0.5;
y_vec = 0:0.1:2;

% standard SKI
func = @(x,y) getJthOutput(@infGaussLik, 2:3, struct('cov',[x,y,hyp_opt.cov(3)],'lik', hyp_opt.lik,'mean', []),...
     {@meanZero}, covGrid_non_warped, likfunc, X_tilde, Y, opt);
[ f_mesh_non_warp, grad_mesh_non_warp, x_coord, y_coord ] = meshfun( func, x_vec, y_vec, 1:2);

% warp-SKI
func = @(x,y) getJthOutput(@infGaussLik, 2:3, struct('cov',[x,y,hyp_opt.cov(3)],'lik', hyp_opt.lik,'mean', []),...
     {@meanZero}, covGrid, likfunc, X, Y, opt);
[ f_mesh_warp, grad_mesh_warp, x_coord, y_coord ] = meshfun( func, x_vec, y_vec, 1:2);

% batch
func = @(x,y) getJthOutput(@infGaussLik, 2:3, struct('cov',[x,y,hyp_opt.cov(3)],'lik', hyp_opt.lik,'mean', []),...
     {@meanZero}, covfunc_warped, likfunc, X, Y);
[ f_mesh_exact, grad_mesh_exact, x_coord, y_coord ] = meshfun( func, x_vec, y_vec, 1:2);

figure
subplot(1,3,1)
surf(x_vec, y_vec, f_mesh_exact)
title('LogLik Batch solution')
subplot(1,3,2)
surf(x_vec, y_vec, f_mesh_non_warp)
title('LogLik Standard SKI solution')
subplot(1,3,3)
surf(x_vec, y_vec, f_mesh_warp)
title('LogLik Warp-SKI solution')

figure
fig_pos = get(gcf, 'Position');
set(gcf, 'position', [fig_pos(1:2)-50, 1200, 500])
subplot(1,3,1)
hold on
contourf(x_vec, y_vec, f_mesh_exact, 35)
plot(hyp_opt.cov(1), hyp_opt.cov(2),'x')
plot(hyp.cov(1), hyp.cov(2), 'o')
colorbar()
title('LogLik')
hSub2 = subplot(1,3,2);
hold on
contourf(x_vec,y_vec, f_mesh_non_warp, 35)
xlim([-1.5,-0.5])
ylim([0,2])
colorbar()
plot(hyp_opt.cov(1), hyp_opt.cov(2),'x')
plot(hyp.cov(1), hyp.cov(2), 'o')
title('Approximate LogLik')
hSub3 = subplot(1,3,3); plot(1, nan, 1, nan, 'r'); set(hSub3, 'Visible', 'off');
sub_pos = get(hSub3,'position');          % gca points at the second one
sub_pos(1,3) = sub_pos(1,3) / 2;              % reduce the height by half
set(hSub3,'position',sub_pos);            % set the values you just changed

%% Compare Hyperparameter Optimization

% set reasonable starting point
hyp0 = hyp;
hyp0.cov = zeros(size(hyp.cov));

disp('--------------------------------------------------------------------------------')
disp('GPML CG Optimzation (standard SKI)')
disp('--------------------------------------------------------------------------------')
opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 20;
opt.cg_maxit = 1000; opt.cg_tol = 0.1;
infg = @(varargin) infGaussLik(varargin{:},opt);
%hyp0 = hyp;
hyp_opt_cg = minimize(hyp0, @gp, -100, infg, [], covGrid_non_warped, likfunc, X_tilde, Y);
plot(hSub2, hyp_opt_cg.cov(1), hyp_opt_cg.cov(2),'x')

%% Try other optimizers
disp('--------------------------------------------------------------------------------')
disp('GPML LBFGS Optimzation (standard SKI)')
disp('--------------------------------------------------------------------------------')
hyp_opt_lbfgs_gpml = minimize_lbfgsb(hyp0, @gp, -100, infg, [], covGrid_non_warped, likfunc, X_tilde, Y);
plot(hSub2, hyp_opt_lbfgs_gpml.cov(1), hyp_opt_lbfgs_gpml.cov(2),'x')

disp('--------------------------------------------------------------------------------')
disp('GPML LBFGS Optimzation (warp-SKI)')
disp('--------------------------------------------------------------------------------')
hyp_opt_lbfgs_gpml_warpSKI = minimize_lbfgsb(hyp0, @gp, -100, infg, [], covGrid, likfunc, X, Y);
plot(hSub2, hyp_opt_lbfgs_gpml_warpSKI.cov(1), hyp_opt_lbfgs_gpml_warpSKI.cov(2),'o')

% Make legend
legend1 = legend(hSub2,'show','Z','CG batch','true', 'CG SKI', 'LBFGS SKI', 'LBFGS warp-SKI');
leg_pos = get(legend1, 'Position');
set(legend1, 'Position', [sub_pos(1:2), leg_pos(3:4)])


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

% Fast method for drawing samples from GP.
% Code by Catherine E. Powell.
function [Z1,Z2] = circ_cov_sample(c1)
M=length(c1); xi=randn(M,2)*[1; sqrt(-1)];
d=ifft(c1,'symmetric')*M;
Z=fft((d.^0.5).*xi)/sqrt(M);
Z1=real(Z); Z2=imag(Z);
end

% Fast method for drawing samples from GP.
% Provided by Catherine E. Powell.
function [Z1,Z2]=circulant_embed_sample(c1)
tilde_c=[c1; c1(end-1:-1:2)];
[Z1,Z2]=circ_cov_sample(tilde_c);
M=length(c1); 
Z1=Z1(1:M); 
Z2=Z2(1:M);
end

function y = kron_mvm(K1,K2,x)
n = size(K1,1);
m = size(K2,2);
x_mat = reshape(x,m,n);
y_mat = K2*(x_mat*K1');
y=y_mat(:);
end

function y = toeplitz_mvm(k, x)
n = length(k);
c = [k,fliplr(k(2:(n-1)))];
x = [x; zeros(n-2,1)];

y = ifft(fft(c).*fft(x'));
y = y(1:n)';
end