function supplied_graph_bbc_diagnostic()
% Revised solver with the undocumented supplied graph. NOT FOR REPORTING.
root_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(root_dir, 'code'));
addpath(genpath(fullfile(root_dir, 'common')));
if exist('OCTAVE_VERSION', 'builtin')
    pkg load control;
    pkg load statistics;
end
cfg = dataset_config(root_dir, 'bbc');
[X, gt] = load_multiview_dataset(cfg);
legacy = load(fullfile(root_dir, 'legacy_diagnostic', ...
    'bbcsport_2view_A.mat'));
Q = legacy.A;
variant = variant_config('full');
[model, history] = solve_sglog(X, Q, cfg.parameters, variant);
[runs, assignments, cluster_sizes, evaluation_diagnostics] = ...
    evaluate_repeated(model.affinity, gt, cfg.cluster_count, 20);
summary = summarize_runs(runs);
summary.iterations = history.iterations;
summary.wall_seconds = history.wall_seconds;
summary.final_primal_residual = history.primal_residual(end);
summary.final_successive_difference = history.successive_difference(end);
summary.stopping_reason = history.stopping_reason;
out_dir = fullfile(root_dir, 'results', 'legacy_diagnostic', ...
    'bbc_supplied_graph_revised_model');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
write_struct_csv(runs, fullfile(out_dir, 'metrics_all_runs.csv'));
write_struct_csv(summary, fullfile(out_dir, 'summary.csv'));
save_compat(fullfile(out_dir, 'diagnostics.mat'), 'history', 'runs', ...
    'assignments', 'cluster_sizes', 'evaluation_diagnostics', 'cfg', ...
    'variant', 'gt');
fid = fopen(fullfile(out_dir, 'NOT_FOR_REPORTING.txt'), 'w');
fprintf(fid, ['Uses the undocumented supplied bbcsport_2view_A.mat with ', ...
    'the corrected revised model solely to isolate graph-source effects.\n']);
fclose(fid);
fprintf('SUPPLIED_GRAPH_DIAGNOSTIC_DONE NMI=%.6f ACC=%.6f\n', ...
    summary.nmi_mean, summary.acc_mean);
end
