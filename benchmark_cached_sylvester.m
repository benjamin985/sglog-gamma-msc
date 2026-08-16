function benchmark_cached_sylvester(n)
if nargin < 1, n = 500; end
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
if exist('OCTAVE_VERSION', 'builtin'), pkg load control; end
rand('state', 20260813); %#ok<RAND>
randn('state', 20260813); %#ok<RAND>
F = randn(64, n); G = F' * F;
R = sprand(n, n, min(0.02, 8 / n)); R = abs(R + R');
L = diag(sum(R, 2)) - R;
C = randn(n);
alpha = 0.037; beta = 1.9; rho = 0.004;
t = tic; cache = prepare_symmetric_sylvester(G, L); prep_seconds = toc(t);
t = tic; Zc = solve_symmetric_sylvester_cached(cache, C, alpha, beta, rho); cached_seconds = toc(t);
t = tic; Zr = sylvester(alpha * G + rho * eye(n), beta * full(L), C); reference_seconds = toc(t);
relative_error = norm(Zc - Zr, 'fro') / max(1, norm(Zr, 'fro'));
fprintf(['n=%d prep=%.6f cached_solve=%.6f reference_solve=%.6f ' ...
    'relative_error=%.3e\n'], n, prep_seconds, cached_seconds, ...
    reference_seconds, relative_error);
assert(relative_error < 1e-8);
end
