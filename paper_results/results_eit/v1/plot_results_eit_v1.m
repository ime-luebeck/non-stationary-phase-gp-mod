clear all
close all

%% load EIT results (of non-learning version)
x0 = (1:64)'; % Y-axis of image
x1 = (1:64)'; % X-axis of image
load('./results/results_eit/v1/result.mat')

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

%% plot separate pixels
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