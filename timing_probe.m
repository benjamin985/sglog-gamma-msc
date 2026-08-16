function timing_probe()
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
addpath(genpath(fullfile(root_dir, 'common')));
ds = 'scene';
cfg = dataset_config(root_dir, ds);
cfg.parameters.max_iter = 100;
[X, gt] = load_multiview_dataset(cfg); %#ok<NASGU>
n0 = size(X{1}, 2);
set_reproducible_seed(20260813 + 5);
keep = sort(randperm(n0, 1200));
for v = 1:numel(X)
    X{v} = X{v}(:, keep);
end
Q = cell(size(X));
for v = 1:numel(X)
    Q{v} = construct_gradient_operator(X{v}, cfg.knn);
end
variant = variant_config('full');
t0 = tic;
[model, history] = solve_sglog(X, Q, cfg.parameters, variant);
fprintf('PROBE scene N=%d iters=%d stop=%s wall=%.1fs obj=%.6g\n', ...
    size(X{1}, 2), history.iterations, history.stopping_reason, ...
    toc(t0), history.objective(end));
end
