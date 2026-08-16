function smoke_test()
% Fast syntax, proximal, graph and miniature end-to-end checks.

root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
addpath(genpath(fullfile(root_dir, 'common')));
if exist('OCTAVE_VERSION', 'builtin')
    pkg load control;
    pkg load statistics;
end

for y = [0, 0.01, 0.1, 1, 10, 100]
    [x, objective, candidates] = prox_gamma_scalar(y, 12, 1.7);
    assert(x >= 0 && isfinite(objective) && any(abs(candidates - x) < 1e-12));
end
Y = reshape(linspace(-2, 2, 60), 6, 10);
E = prox_l2log_columns(Y, 0.2);
assert(isequal(size(E), size(Y)) && all(isfinite(E(:))));

sample_index = 1:30;
X = {
    reshape(sin((1:12)' * sample_index / 17) + cos((1:12)' * sample_index / 29), 12, 30), ...
    reshape(cos((1:10)' * sample_index / 13) - sin((1:10)' * sample_index / 31), 10, 30), ...
    reshape(sin((1:8)' * sample_index / 11) + 0.5 * cos((1:8)' * sample_index / 19), 8, 30)};
for v = 1:numel(X)
    X{v} = X{v} ./ max(sqrt(sum(X{v}.^2, 1)), eps);
    Q{v} = construct_gradient_operator(X{v}, 8); %#ok<AGROW>
end
parameters.max_iter = 3;
parameters.tolerance = 1e-8;
parameters.alpha0 = 1e-4;
parameters.beta0 = 1e-4;
parameters.rho0 = 1e-4;
parameters.penalty_growth = 1.8;
parameters.penalty_max = 1e10;
parameters.lambda1 = 1e-4;
parameters.lambda2 = 1e-3;
parameters.gamma = 12;
[model, history] = solve_sglog(X, Q, parameters, variant_config('full'));
assert(isequal(size(model.affinity), [30, 30]));
assert(all(isfinite(model.affinity(:))));
assert(numel(history.objective) == 3);
fprintf('SMOKE_TEST_OK runtime=%s objective=%.10g residual=%.3e\n', ...
    version, history.objective(end), history.primal_residual(end));
end
