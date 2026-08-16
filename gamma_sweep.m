function gamma_sweep()
% Post-selection gamma sensitivity analysis (label-free reporting only).
%
% The lambda1/lambda2 values are the label-free-selected, locked settings from
% TUNING_PROTOCOL.md / dataset_config.m. This script sweeps ONLY the tensor
% gamma* shape parameter on the FULL dataset under the standard 20-seed
% evaluation, to report an honest sensitivity curve. It does not re-select
% lambda1/lambda2 and never uses labels to choose a configuration; the metric
% means are written verbatim for every gamma value.

root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
addpath(genpath(fullfile(root_dir, 'common')));

run_tag = strtrim(getenv('SGLOG_RUN_TAG'));
if isempty(run_tag)
    run_tag = 'gamma_sweep_20260815';
end
results_root = fullfile(root_dir, 'results', run_tag);
if ~exist(results_root, 'dir')
    mkdir(results_root);
end

n_runs = str2double(getenv('SGLOG_RUNS'));
if isnan(n_runs) || n_runs < 2
    n_runs = 20;
end

% dataset -> gamma values to sweep (locked lambda1/lambda2 come from dataset_config)
sweeps = struct();
sweeps.scene = [12, 24, 48, 96];
sweeps.yale = [12, 24, 48];

datasets = fieldnames(sweeps);
all_rows = {};

for d = 1:numel(datasets)
    name = datasets{d};
    cfg = dataset_config(root_dir, name);
    [X, gt] = load_multiview_dataset(cfg);
    Q = cell(size(X));
    for v = 1:numel(X)
        Q{v} = construct_gradient_operator(X{v}, cfg.knn);
    end
    for g = sweeps.(name)
        fprintf('\n=== %s / gamma=%.6g ===\n', name, g);
        cfg.parameters.gamma = g;
        variant = variant_config('full');
        [model, history] = solve_sglog(X, Q, cfg.parameters, variant);
        [runs, assignments, cluster_sizes, eval_diag] = evaluate_repeated( ...
            model.affinity, gt, cfg.cluster_count, n_runs); %#ok<ASGLU>
        summary = summarize_runs(runs);
        summary.gamma = g;
        summary.iterations = history.iterations;
        summary.wall_seconds = history.wall_seconds;
        summary.stopping_reason = history.stopping_reason;
        job_dir = fullfile(results_root, sprintf('%s_g%.6g', name, g));
        if ~exist(job_dir, 'dir')
            mkdir(job_dir);
        end
        write_struct_csv(runs, fullfile(job_dir, 'metrics_all_runs.csv'));
        write_struct_csv(summary, fullfile(job_dir, 'summary.csv'));
        fid = fopen(fullfile(job_dir, 'COMPLETED.ok'), 'w');
        fprintf(fid, 'completed_at=%s\n', datestr(now, 31));
        fclose(fid);
        fprintf('   NMI=%.4f +/- %.4f  ACC=%.4f +/- %.4f  iter=%d t=%.1fs [%s]\n', ...
            summary.nmi_mean, summary.nmi_std, summary.acc_mean, summary.acc_std, ...
            summary.iterations, summary.wall_seconds, summary.stopping_reason);
        all_rows{end + 1, 1} = summary; %#ok<AGROW>
        clear model history runs assignments cluster_sizes eval_diag;
    end
end

fprintf('\nGamma sweep finished. Results: %s\n', results_root);
end
