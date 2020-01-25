% DEMO_NON_STATIONARY_2D_GRIDBASED_STRESSTEST_UNSTRUCT
% 
% Description
%   Stresstest of the warpSKI method for different number of input points
%   and inducing points. These results are stored to
%   'paper_results/results_stresstest/' and can be plotted from there the
%   retrieve the figures from the paper.
% 
% References:
%   Grasshoff, J., Jankowski, A. and Rostalski, P. (2019). Scalable Gaussian
%   Process Regression for Kernels with a non-stationary phase,
%   arxiv.org/abs/1912.11713

points = [10^2.75, 10^3, 10^3.5, 10^4, 10^4.5, 10^5];
inducing_points = [10000, 20000, 50000, 100000];

for i = 1:length(points)
for j = 1:length(inducing_points)
    disp(['Number of points: ', num2str(points(i))])
    disp(['Number of inducing points: ', num2str(inducing_points(j))])
    rng(45678)

    %% define input data
    num_of_points = points(i);
    
    % for structured inputs these two numbers determine the ratio of points on the two input axes
    num_of_x0_points = 40;
    num_of_x1_points = 25;
    factor = sqrt(num_of_points/(num_of_x0_points*num_of_x1_points));
    
    % make structured input points
    % x0 = linspace(-1.2,0.75,num_of_x0_points*factor)';
    % x1 = linspace(-2.5,2.5,num_of_x1_points*factor)';  
    
    % make unstructured input points
    x1 = linspace(-2.5,2.5,num_of_x1_points*factor)';  
    X0 = (1.2+0.75)*rand(round(num_of_x0_points*factor), round(num_of_x1_points*factor)) - 1.2;
    X1 = (5)*rand(round(num_of_x0_points*factor), round(num_of_x1_points*factor)) - 2.5;
    X = [X0(:), X1(:)];

    % make warping function (the inverse was derived analytically in closed form)
    p = @(t) (2*t.^3 + t);
    dp = @(t) (2*3*t.^2 + 1);
    invp = @(x) ((1./((6^(1/3))* (sqrt(3)*sqrt(27*(x.^2) + 2) - 9*x).^(1/3))) -...
        (((sqrt(3)*sqrt(27*x.^2 + 2) - 9*x).^(1/3))/6^(2/3)));

    %% Kernels and inducing points

    % dimension 1
    cov_mat1 = @covSEiso;
    cov_dim1 = {@covMask,{[1,0],cov_mat1}};

    % dimension 2
    cov_mat2 = @covSEisoU;
    cov_dim2 = {@covMask,{[0,1],cov_mat2}};

    % set hyperparameters and likelihood
    ell = 0.4; sf = 1.5;
    hyp.cov = log([ell; sf; ell]);
    likfunc = @likGauss;
    sn = 0.5;
    hyp.lik = log(sn);

    % Inducing points U = phi^-1(U_rad)
    num_of_inducing_points = inducing_points(j);
    factor = sqrt(num_of_inducing_points/6000);
    U_dim1_rad = linspace(-5.5,2,factor*100)';
    U_dim1 = invp(U_dim1_rad); 
    U_dim2 = apxGrid('create',x1,true,[round(factor*60)]);
    U_dim2 = U_dim2{1};

    xg = {{U_dim1},{U_dim2}}; % non-equidistant U
    xg_rad = {{U_dim1_rad},{U_dim2}}; % equidistant U
    U = apxGrid('expand',xg);
    U_rad = apxGrid('expand',xg_rad);

    % set options for inference
    opt.cg_maxit = 1500; opt.cg_tol = 1e-1;

    %% Kernel definition without warping (standard SKI)

    X_tilde = [p(X0(:)), X1(:)];

    covfunc_non_warped2 = {cov_mat1,cov_mat2};
    covGrid_non_warped = {@apxGrid,covfunc_non_warped2,xg_rad}; % standard SKI kernel (equidistant inducing points)

    %% Generate data 

    % get higher number of inducing points (to achieve high accuracy for data generation)
    U_dim1_rad_dense = (-5.5:0.01:1.5)';
    U_dim2_dense = apxGrid('create',x1,true,[90]);
    U_dim2_dense = U_dim2_dense{1};
    xg_rad_dense = {{U_dim1_rad_dense},{U_dim2_dense}};

    % get kernel matrices and eigendecompositions (which are needed for fast data generation)
    K_U1 = feval(cov_mat1, hyp.cov(1:2), U_dim1_rad_dense);
    [V1,D1] = eig((K_U1+K_U1')/2);
    K_U2 = feval(cov_mat2, hyp.cov(3), U_dim2_dense);
    [V2,D2] = eig((K_U2+K_U2')/2);

    % Sample (generated exploiting Kronecker structure)
    test_vec = randn(size(K_U1,1)*size(K_U2,1),1);
    sample = real(kron_mvm(V2, V1, kron(real(sqrt(diag(D2))),real(sqrt(diag(D1)))).*test_vec ));

    % Warp sample
    Mx = apxGrid('interp',xg_rad_dense, X_tilde, 3);
    sample = Mx*sample;
    Y_orig = sample;

    Y = Y_orig + sn*randn(size(X1(:)));

    % plot generated sample
    figure
    scatter3(X0(:),X1(:),Y_orig,[],Y_orig)

    %% kernel definition with warping (warp-SKI)

    cov_dim1_warped = {@covWarp,{cov_mat1},p,dp,1};
    
    covfunc_warped2 = {cov_dim1_warped,cov_mat2};
    covGrid = {@apxGrid,covfunc_warped2,xg}; % warp-SKI kernel (non-equidistant inducing points)

    %% do regression with standard SKI
    
    % do regression (but turn of variance)
    hyp.mean = [];
    [post,nlZ,dnlZ] = infGrid(hyp, {@meanZero}, covGrid_non_warped, {@likGauss}, X_tilde, Y, opt);

    post.L = @(a) zeros(size(a));
    infg = @(varargin) infGaussLik(varargin{:},opt);
    mu_grid = gp(hyp,infg,[],covGrid_non_warped,[],X_tilde,post,X_tilde);

    % plot result
    figure
    subplot(1,2,1)
    scatter3(X0(:),X1(:),mu_grid,[],mu_grid)
    title('Standard SKI regression results')

    %% do regression with warp-SKI
    
    % do regression (but turn of variance)
    hyp.mean = [];
    [post,nlZ,dnlZ] = infGrid(hyp, {@meanZero}, covGrid, {@likGauss}, X, Y, opt);

    post.L = @(a) zeros(size(a));
    infg = @(varargin) infGaussLik(varargin{:},opt);
    mu_grid = gp(hyp,infg,[],covGrid,[],X,post,X);

    % plot result
    subplot(1,2,2)
    scatter3(X0(:),X1(:),mu_grid,[],mu_grid)
    title('Warp-SKI based regression results')


    %% Evaluate marginal likelihood with warp-SKI
    opt.ldB2_method = 'lancz'; opt.ldB2_hutch = 20;
    opt.cg_maxit = 1000; opt.cg_tol = 0.25;
    infg = @(varargin) infGaussLik(varargin{:},opt);
    
    % measure time for evaluation
    tic();
    [~, nlZ, dnlZ] = infGaussLik(hyp, {@meanZero}, covGrid, likfunc, X, Y, opt);
    result.marg_lik_evaluation_time = toc();
     
    %% Run Hyperparameter Optimization + GP regression afterwards

    % set reasonable starting point
    hyp0 = hyp;
    hyp0.cov = zeros(size(hyp.cov));

    % set options specifically for inference
    opt_inf.cg_maxit = 1500; opt_inf.cg_tol = 1e-1;
    opt_inf.ldB2_method = 'lancz'; opt_inf.ldB2_hutch = 0; % turn of marg. likelihood calculation for inference

    % Run LBFGS using warpSKI
    disp('--------------------------------------------------------------------------------')
    disp('GPML LBFGS Optimzation (warp-SKI)')
    disp('--------------------------------------------------------------------------------')
    tic();
    [hyp_opt_lbfgs_gpml_warpSKI, fval, iter] = minimize_lbfgsb(hyp0, @gp, -100, infg, [], covGrid, likfunc, X, Y);
    
    result.lbfgs.learning_time = toc();
    result.lbfgs.learning_opt = opt;
    result.lbfgs.hyp0 = hyp0;

    % do regression with warp-SKI using optimized hyperparameters
    % do regression (but turn of variance)
    tic();
    hyp_opt_lbfgs_gpml_warpSKI.mean = [];
    [post,~,~] = infGrid(hyp_opt_lbfgs_gpml_warpSKI, {@meanZero}, covGrid, {@likGauss}, X, Y, opt_inf);

    post.L = @(a) zeros(size(a));
    infg = @(varargin) infGaussLik(varargin{:},opt_inf);
    mu_grid = gp(hyp_opt_lbfgs_gpml_warpSKI,infg,[],covGrid,[],X,post,X);
    
    % store performance results
    result.lbfgs.inference_time = toc();
    result.lbfgs.inference_opt = opt_inf;
    result.lbfgs.inference_rmse = sqrt(mean((Y-mu_grid).^2));
    result.lbfgs.mu_grid = mu_grid;
    result.lbfgs.learned_hyp = hyp_opt_lbfgs_gpml_warpSKI;
    result.lbfgs.iter = iter;
    result.lbfgs.fval = fval;

    % plot regression result
    figure
    scatter3(X0(:),X1(:),mu_grid,[],mu_grid)
    title('LBFGS optimization - regression results')

    % store other stuff
    result.X0 = X0;
    result.X1 = X1;
    result.Y = Y;
    result.Y_orig = Y_orig;
    result.X = X;
    result.p = p;
    result.dp = dp;
    result.invp = invp;
    result.X_tilde = X_tilde;
    result.covGrid = covGrid;
    result.covGrid_non_warped = covGrid_non_warped;

    % save all the data to results directory
    save(['paper_results/results_stresstest/experiment_2D_warped_', num2str(num_of_points), '_points_', num2str(num_of_inducing_points), '_inducing_points.mat'], 'result')

end
end

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