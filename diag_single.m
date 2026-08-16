function diag_single()
% Diagnostic: run solve_sglog on one dataset/config and print the full
% per-iteration trace so the source of any NaN/divergence can be located.
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
addpath(genpath(fullfile(root_dir, 'common')));
if exist('OCTAVE_VERSION', 'builtin')
    pkg load control;
    pkg load statistics;
end

ds = getenv('DIAG_DS'); if isempty(ds), ds = 'uci'; end
l1 = str2double(getenv('DIAG_L1')); if isnan(l1), l1 = 1e-4; end
l2 = str2double(getenv('DIAG_L2')); if isnan(l2), l2 = 5e-3; end
ga = str2double(getenv('DIAG_G')); if isnan(ga), ga = 12; end
subs = str2double(getenv('DIAG_SUBSAMPLE')); if isnan(subs), subs = 1200; end

cfg = dataset_config(root_dir, ds);
cfg.parameters.lambda1 = l1;
cfg.parameters.lambda2 = l2;
cfg.parameters.gamma = ga;
cfg.parameters.max_iter = 200;

[X, gt] = load_multiview_dataset(cfg); %#ok<NASGU>
n0 = size(X{1}, 2);
if subs > 0 && n0 > subs
    set_reproducible_seed(20260813 + find(strcmp({'uci','bbc','yale','orl','scene'}, ds)));
    keep = sort(randperm(n0, subs));
    for v = 1:numel(X)
        X{v} = X{v}(:, keep);
    end
end
fprintf('dataset=%s N=%d l1=%.3g l2=%.3g g=%.3g\n', ds, size(X{1},2), l1, l2, ga);

Q = cell(size(X));
for v = 1:numel(X)
    Q{v} = construct_gradient_operator(X{v}, cfg.knn);
end
variant = variant_config('full');
[model, history] = solve_sglog(X, Q, cfg.parameters, variant);
fprintf('STOPPING_REASON=%s iterations=%d wall=%.1fs\n', ...
    history.stopping_reason, history.iterations, history.wall_seconds);
fprintf('last objective=%.8g residual=%.3e\n', ...
    history.objective(end), history.primal_residual(end));
W = model.affinity;
fprintf('affinity finite=%d nnz=%d min_degree=%.3g max_abs=%.3g\n', ...
    all(isfinite(W(:))), nnz(W), min(sum(W,2)), max(abs(W(:))));
end
