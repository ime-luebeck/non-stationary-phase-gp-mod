clear all
close all

load('demos/demo_BSPM/Nijmegen-2004-12-09/Interventions/Noisy_room/PPD2_sessie1_exp_04.mat')
pots([1,2,61],:) = [];

% downsample signals from presumably 1000Hz to 200Hz
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

% load calculated results
load('paper_results/results_BSPM/v4/hyperparameters.mat')
load('paper_results/results_BSPM/v4/regression.mat')

hyp = hyp_opt_lbfgs_gpml;

% remove 
X_train = X;
X_train(to_remove, :) = [];

% Data
pots_train = pots;
pots_train(to_remove, :) = [];
Y = pots_train(:);

%% 3D plot
% plot 3D model
figure
hold on
s = trisurf(faces,nodes(:,1),nodes(:,2),nodes(:,3),'FaceColor', 'interp', 'EdgeColor', 'none');
%alpha 0.5
plot3(pts(:,1), pts(:,2), pts(:,3), '.g', 'MarkerSize',20)
plot3(pts(to_remove,1), pts(to_remove,2), pts(to_remove,3), '.r', 'MarkerSize', 20)
for i=1:length(to_remove)
    text(pts(to_remove(i),1), pts(to_remove(i),2), pts(to_remove(i),3), num2str(to_remove(i)))
end
light('Position',[100.0,-100.0,100.0],'Style','infinite');
lighting phong;
axis equal

%% electrode prediction
% predict all electrodes (including the removed ones)
Xtest = X;
X_test_space = repmat(Xtest, length(t), 1);
X_t = kron(t, ones(size(Xtest,1),1));
X_test_all = [X_test_space, X_t];
[Mx_pred1,~] = apxGrid('interp',kernel_SKI{3},X_test_all,3);
[Kg1,Mx1] = feval(kernel_SKI{:},hyp.cov,X_all,[],3);

ymu = Mx_pred1*Kg1.mvm(Mx1'*post.alpha);

pots_test_mu = reshape(ymu, size(Xtest,1), length(t));

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

% Export electrode prediction data
segm = t>=6.6 & t<7.7;
data_to_export = [pots(electrode_indices, segm); pots_test_mu(electrode_indices, segm)];

data_to_export = [t(segm)-6.65, data_to_export'];
writematrix(data_to_export, 'paper_results/results_BSPM/v4/ecg_signals.csv');

%% plot in 3D (do prediction with reduced time resolution)
nodes_2D = project_to_2D(nodes);

t_100Hz = t(1:2:end);

X_nodes_space = repmat(nodes_2D, length(t), 1);
X_t = kron(t, ones(size(nodes_2D,1),1));

X_nodes_all = [X_nodes_space, X_t];

% make inducing points with higher support domain
U_dim2_ = (-5:0.1:3)';
xg_2 = {kernel_SKI{3}{1},{U_dim2_},kernel_SKI{3}{3}};
kernel_SKI_2 = {@apxGrid, kernel_SKI{2}, xg_2};

[Mx_pred1,~] = apxGrid('interp', xg_2, X_nodes_all, 3);
[Kg1,Mx1] = feval(kernel_SKI_2{:},hyp.cov,X_all,[],3);

ymu = Mx_pred1*Kg1.mvm(Mx1'*post.alpha);

pots_nodes = reshape(ymu, size(nodes_2D,1), length(t));

time_indices = [1, 101, 201, 301, 401, 501, 601, 151, 251];
num_plots = ceil(sqrt(length(electrode_indices)));

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
            caxis([-0.8,1.7])
            plot3(pts(:,1), pts(:,2), pts(:,3), '.g', 'MarkerSize',20)
            plot3(pts(to_remove,1), pts(to_remove,2), pts(to_remove,3), '.r', 'MarkerSize', 20)
            view(32.15,8.25)
            light('Position',[100.0,-100.0,100.0],'Style','infinite');
            lighting phong;
            axis equal
        end
        k = k + 1;
    end
end

%% plot single 3D figure (during R-peak)
time_index = 603;

figure
hold on
c = pots_nodes(:,time_index);
s = trisurf(faces,nodes(:,1),nodes(:,2),nodes(:,3),c, 'FaceColor', 'interp', 'EdgeColor', 'none');
%s = trisurf(faces,nodes(:,1),nodes(:,2),nodes(:,3),c);
caxis([-0.8,1.7])
plot3(pts(:,1), pts(:,2), pts(:,3), '.k', 'MarkerSize',7)
plot3(pts(to_remove,1), pts(to_remove,2), pts(to_remove,3), '.r', 'MarkerSize', 7)
view(36.6327,17.8668)
light('Position',[100.0,-100.0,100.0],'Style','infinite');
lighting phong;
axis equal
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
set(gca,'ZTickLabel',[]);
set(gcf,'Units','inches');
screenposition = get(gcf,'Position');
set(gcf,...
    'PaperPosition',[0 0 0.6*screenposition(3) screenposition(4)],...
    'PaperSize',[0.6*screenposition(3), screenposition(4)]);
print(gcf,'paper_results/results_BSPM/v4/torso.png','-dpng','-r700')

%% Calculate nRMSE
predicted = pots_test_mu(electrode_indices,:);
actual = pots(electrode_indices,:);
nrmse = sqrt(mean((actual-predicted).^2))/std(actual);

