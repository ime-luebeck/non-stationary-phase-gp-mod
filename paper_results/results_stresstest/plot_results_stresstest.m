close all
clear all

% Find all results files
path = './paper_results/results_stresstest/';

S = dir([path, 'experiment_2D_warped*']);

inference_times = zeros(length(S),1);
points = zeros(length(S),1);
inducing_points = zeros(length(S),1);
learning_times = zeros(length(S),1);
iterations = zeros(length(S),1);
marg_lik_evaluation_time = zeros(length(S),1);
inference_errors_lbfgs = zeros(length(S),1);

for i = 1:length(S)
    file = [S(i).folder, '/', S(i).name];
    load(file)
    inference_times(i) = result.lbfgs.inference_time;
    learning_times(i) = result.lbfgs.learning_time;
    points(i) = size(result.X,1);
    iterations(i) = result.lbfgs.iter;
    inference_errors_lbfgs(i) = sqrt(mean((result.Y_orig-result.lbfgs.mu_grid).^2));
    inducing_points(i) = length(result.covGrid{3}{1}{1}) * length(result.covGrid{3}{2}{1});
    marg_lik_evaluation_time(i) = result.marg_lik_evaluation_time;
end

% Plot inference times
fig = figure;
hold on
legend_entries = {};
export_to_csv = [];
for i=unique(inducing_points)'
    current = find((inducing_points==i));
    [~,I] = sort(points(current));
    current_sorted = current(I);
    
    plot(points(current_sorted), inference_times(current_sorted), 'o-')
    legend_entries{end+1} = ['m=', num2str(i)];
    
    to_save = inference_times(current_sorted);
    export_to_csv = [export_to_csv, to_save(2:end)];
end

to_save = points(current_sorted);
export_to_csv = [to_save(2:end), export_to_csv];
writematrix(export_to_csv, './paper_results/results_stresstest/inference_times.csv');

set(gca, 'XScale', 'log', 'YScale', 'log');
title('Inference Time')
ylabel('CPU Time (seconds)')
xlabel('Number of data points')
legend(legend_entries)

print_fig_to_png(fig, strcat('./paper_results/results_stresstest/inference_times.png'), 5, 5);

% Plot marg-lik evaluation times
fig = figure;
hold on
legend_entries = {};
export_to_csv = [];
for i=unique(inducing_points)'
    current = find((inducing_points==i));
    [~,I] = sort(points(current));
    current_sorted = current(I);
    
    plot(points(current_sorted), marg_lik_evaluation_time(current_sorted), 'o-')
    legend_entries{end+1} = ['m=', num2str(i)];
    to_save = marg_lik_evaluation_time(current_sorted);
    export_to_csv = [export_to_csv, to_save(2:end)];
end

to_save = points(current_sorted);
export_to_csv = [to_save(2:end), export_to_csv];
writematrix(export_to_csv, './paper_results/results_stresstest/marg_lik_eval_times.csv');

set(gca, 'XScale', 'log', 'YScale', 'log');
title('Marginal likelihood evaluation time')
legend(legend_entries)
ylabel('CPU Time (seconds)')
xlabel('Number of data points')
print_fig_to_png(fig, strcat('./paper_results/results_stresstest/marg_lik_eval_times.png'), 5, 5);


% Plot Learning time
fig = figure;
hold on
legend_entries = {};
export_to_csv = [];
for i=unique(inducing_points)'
    current = find((inducing_points==i));
    [~,I] = sort(points(current));
    current_sorted = current(I);
    
    plot(points(current_sorted), learning_times(current_sorted), 'o-')
    legend_entries{end+1} = ['m=', num2str(i)];
    to_save = learning_times(current_sorted);
    export_to_csv = [export_to_csv, to_save(2:end)];
end

to_save = points(current_sorted);
export_to_csv = [to_save(2:end), export_to_csv];
writematrix(export_to_csv, './paper_results/results_stresstest/learning_times.csv');

set(gca, 'XScale', 'log', 'YScale', 'log');
title('Hyperparameter learning time')
legend(legend_entries)
ylabel('CPU Time (seconds)')
xlabel('Number of data points')
print_fig_to_png(fig, strcat('./paper_results/results_stresstest/learning_times.png'), 5, 5);

% Plot errors LBFGS
fig = figure;
hold on
legend_entries = {};
export_to_csv = [];
for i=unique(inducing_points)'
    current = find((inducing_points==i));
    [~,I] = sort(points(current));
    current_sorted = current(I);
    
    plot(points(current_sorted), inference_errors_lbfgs(current_sorted), 'o-')
    legend_entries{end+1} = ['m=', num2str(i)];
    to_save = inference_errors_lbfgs(current_sorted);
    export_to_csv = [export_to_csv, to_save(2:end)];
end

to_save = points(current_sorted);
export_to_csv = [to_save(2:end), export_to_csv];
writematrix(export_to_csv, './paper_results/results_stresstest/regression_errors.csv');

set(gca, 'XScale', 'log', 'YScale', 'linear');
title('RMS regression error')
legend(legend_entries)
ylabel('RMS')
xlabel('Number of data points')
print_fig_to_png(fig, strcat('./paper_results/results_stresstest/regression_errors.png'), 5, 5);
