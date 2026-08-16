function test_cached_sylvester()
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
if exist('OCTAVE_VERSION', 'builtin'), pkg load control; end
rand('state', 20260813); %#ok<RAND>
randn('state', 20260813); %#ok<RAND>
for n = [8, 31, 80]
    A0 = randn(n); G = A0' * A0;
    R = sprand(n, n, min(0.2, 8 / n)); R = abs(R + R');
    L = diag(sum(R, 2)) - R;
    C = randn(n);
    alpha = 0.037; beta = 1.9; rho = 0.004;
    cache = prepare_symmetric_sylvester(G, L);
    Z_cached = solve_symmetric_sylvester_cached(cache, C, alpha, beta, rho);
    Z_reference = sylvester(alpha * G + rho * eye(n), beta * full(L), C);
    relative_error = norm(Z_cached - Z_reference, 'fro') / ...
        max(1, norm(Z_reference, 'fro'));
    equation_residual = norm((alpha * G + rho * eye(n)) * Z_cached + ...
        Z_cached * (beta * full(L)) - C, 'fro') / max(1, norm(C, 'fro'));
    fprintf('n=%d relative_error=%.3e residual=%.3e\n', ...
        n, relative_error, equation_residual);
    assert(relative_error < 1e-8);
    assert(equation_residual < 1e-8);
end
fprintf('CACHED_SYLVESTER_TEST_PASSED\n');
end
