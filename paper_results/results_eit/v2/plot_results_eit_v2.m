clear all
close all

%% load EIT results (of non-learning version)
x0 = (1:64)'; % Y-axis of image
x1 = (1:64)'; % X-axis of image
load('./results/results_eit/v2/result.mat')

mu_grid = signal_1_grid + signal_2_grid;

% rebuild and show pictures
Y_all = nan*ones(size(imgs(:)));
Y_all(~isnan(imgs(:))) = mu_grid;
imgs_regression = reshape(Y_all, [length(x0), length(x1), length(time)]);

Y_1 = nan*ones(size(imgs(:)));
Y_1(~isnan(imgs(:))) = signal_1_grid;
imgs_regression_1 = reshape(Y_1, [length(x0), length(x1), length(time)]);

Y_2 = nan*ones(size(imgs(:)));
Y_2(~isnan(imgs(:))) = signal_2_grid;
imgs_regression_2 = reshape(Y_2, [length(x0), length(x1), length(time)]);

array_nan = nan*ones(length(x0), 1, length(time));

imgs_comparison = horzcat(M+imgs, array_nan, M+imgs_regression_1, array_nan, imgs_regression_2);

implay(mat2gray(imgs_comparison, [max(M(:)+imgs(:)), min(M(:)+imgs(:))]), 13);

%% plot separate pixels
xposns = [50  30 36]; yposns = [30 15 19];
fig = figure;
%subplot(length(xposns)+1, 1, 1)
hold on
%axis equal
imagesc(M(:,:,71) + imgs(:,:,71))
for i = 1:length(xposns)
    plot(xposns(i),yposns(i),'s','LineWidth',2,'Color','k');
    text(xposns(i)+3,yposns(i),num2str(i))
end
set(gca,'YDir','reverse')
xlim([0,64])
ylim([0,64])

print_fig_to_png(fig, strcat('./results/results_eit/v2/lung1.png'), 5, 5);

to_save = M(:,:,71) + imgs(:,:,71);
to_save = to_save - min(to_save(:));
max_val = max(to_save(:));
imwrite((kron(to_save,ones(10)))*64.0/max_val, parula, './results/results_eit/v2/lung2.png');

print_fig_to_png(fig, strcat('./results/results_eit/v2/separation.png'), 6, 4);

writematrix(to_save, './results/results_eit/v2/eit_traces.csv');

%% plot predicted image
load('./results/results_eit/v2/result_prediction.mat')
orig = load('./demos/demo_EIT/if-neonate-spontaneous/eit_imgs.mat');

figure
subplot(1,2,1)
imagesc(img_test+M(:,:,215))
title('predicted')
colorbar
subplot(1,2,2)
imagesc(orig.imgs(:,:,216)./normalizer)
title('measured')
colorbar
