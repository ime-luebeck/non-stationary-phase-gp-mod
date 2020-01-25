clear all
close all

load('results\results_fetal_ecg\results_small_scale.mat')
fig = figure;
ax1 = subplot(3,1,1);
plot(time_segm(1:2:end), abd4_segm(1:2:end))
xlabel('Time (seconds)')
title('Measurement')
ax2 = subplot(3,1,2);
hold on
plot(time_segm(1:2:end), signal_1_exact)
plot(time_segm(1:2:end), signal_2_exact)
xlabel('Time (seconds)')
title('Batch GP separation')
ax3 = subplot(3,1,3);
hold on
plot(time_segm(1:2:end), signal_1)
plot(time_segm(1:2:end), signal_2)
xlabel('Time (seconds)')
title('Warp-SKI GP separation')
linkaxes([ax1,ax2,ax3], 'x')
xlim([time_segm(1), time_segm(end)])

print_fig_to_png(fig, strcat('./results/results_fetal_ecg/small_scale_test.png'), 7, 5);

load('results\results_fetal_ecg\results_marg_lik.mat')
fig = figure;
semilogx(exp(x_vec), f_mesh_non_warp)
hold on
semilogx(exp(x_vec), f_mesh_warp)
xlim([exp(x_vec(1)), exp(x_vec(end))])
ylabel('Negative log marg likelihood')
xlabel('Variance kernel parameter')
legend('Batch GP', 'Warp-SKI')

print_fig_to_png(fig, strcat('./results/results_fetal_ecg/marg_lik_evaluation.png'), 6, 5);

load('results\results_fetal_ecg\results_large_scale.mat')
fig = figure;
ax1 = subplot(3,1,1);
plot(time_segm, abd4_segm)
xlabel('Time (seconds)')
title('Measurement')
ax2 = subplot(3,1,2);
plot(time_segm, signal_1)
xlabel('Time (seconds)')
title('Maternal ECG')
ax3 = subplot(3,1,3);
plot(time_segm, signal_2)
xlabel('Time (seconds)')
title('Fetal ECG')
linkaxes([ax1,ax2,ax3], 'x')
xlim([time_segm(1), time_segm(end)])

print_fig_to_png(fig, strcat('./results/results_fetal_ecg/large_scale_test.png'), 7, 5);

xlim([24, 33])
print_fig_to_png(fig, strcat('./results/results_fetal_ecg/large_scale_test_closeup.png'), 7, 5);

% save data to csv
segm = time_segm>=25 & time_segm<=31;
segm(1:2:end)=0;
writematrix([time_segm(segm)', abd4_segm(segm)', signal_1(segm), signal_2(segm)],...
    './results/results_fetal_ecg/ecg_excerpt.csv');

writematrix([exp(x_vec'), f_mesh_non_warp', f_mesh_warp'],...
    './results/results_fetal_ecg/ecg_log_lik.csv');