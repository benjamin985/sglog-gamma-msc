function cross_prox_bbc_diagnostic()
% Two missing cells of the 2x2 error-proximal x gamma-proximal audit.
root_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(root_dir, 'code'));
addpath(genpath(fullfile(root_dir, 'common')));
if exist('OCTAVE_VERSION', 'builtin'), pkg load control; pkg load statistics; end
cfg = dataset_config(root_dir, 'bbc');
[X, gt] = load_multiview_dataset(cfg);
legacy = load(fullfile(root_dir, 'legacy_diagnostic', 'bbcsport_2view_A.mat'));
Q = legacy.A;
names = {'elementwise_global_gamma', 'joint_legacy_gamma'};
errors = {'elementwise_log', 'joint_l2log'};
gammas = {'gamma', 'gamma_legacy'};
out_root = fullfile(root_dir, 'results', 'legacy_diagnostic', 'bbc_cross_prox');
for q = 1:2
    variant = variant_config('full');
    variant.name = names{q}; variant.error_penalty = errors{q};
    variant.tensor_penalty = gammas{q};
    [model, history] = solve_sglog(X, Q, cfg.parameters, variant);
    [runs, assignments, cluster_sizes, evaluation_diagnostics] = ...
        evaluate_repeated(model.affinity, gt, cfg.cluster_count, 20);
    summary = summarize_runs(runs);
    summary.iterations = history.iterations;
    summary.wall_seconds = history.wall_seconds;
    summary.final_primal_residual = history.primal_residual(end);
    summary.final_successive_difference = history.successive_difference(end);
    summary.stopping_reason = history.stopping_reason;
    out_dir = fullfile(out_root, names{q});
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    write_struct_csv(summary, fullfile(out_dir, 'summary.csv'));
    write_struct_csv(runs, fullfile(out_dir, 'metrics_all_runs.csv'));
    save_compat(fullfile(out_dir, 'diagnostics.mat'), 'history', 'runs', ...
        'assignments', 'cluster_sizes', 'evaluation_diagnostics', 'cfg', ...
        'variant', 'gt');
    fid=fopen(fullfile(out_dir,'NOT_FOR_REPORTING.txt'),'w');
    fprintf(fid,'Controlled legacy-proximal diagnostic; not a manuscript result.\n'); fclose(fid);
    fprintf('CROSS_PROX %s NMI=%.6f ACC=%.6f\n', names{q}, ...
        summary.nmi_mean, summary.acc_mean);
end
end
