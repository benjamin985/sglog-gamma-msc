function diag_growth()
% Bisect the ADMM divergence: run ORL (full data, locked config) with a
% configurable penalty_growth and report one summary line.
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
addpath(genpath(fullfile(root_dir, 'common')));
if exist('OCTAVE_VERSION', 'builtin')
    pkg load control;
    pkg load statistics;
end

g = str2double(getenv('DIAG_GROWTH')); if isnan(g), g = 1.8; end
grow_mode = getenv('DIAG_GROW_MODE'); if isempty(grow_mode), grow_mode = 'all'; end

cfg = dataset_config(root_dir, 'orl');
cfg.parameters.penalty_growth = g;
cfg.parameters.max_iter = 200;
[X, gt] = load_multiview_dataset(cfg); %#ok<NASGU>
Q = cell(size(X));
for v = 1:numel(X)
    Q{v} = construct_gradient_operator(X{v}, cfg.knn);
end
variant = variant_config('full');
[model, history] = solve_sglog(X, Q, cfg.parameters, variant);
fprintf('GROWTH=%.3g MODE=%s STOP=%s iters=%d obj=%.6g res=%.3e wall=%.1f\n', ...
    g, grow_mode, history.stopping_reason, history.iterations, ...
    history.objective(end), history.primal_residual(end), history.wall_seconds);
end
