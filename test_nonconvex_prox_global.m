function test_nonconvex_prox_global()
% Compare both nonconvex scalar proximal rules with dense one-dimensional scans.

root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));

gamma_values = [0.25, 1, 6.43, 12];
rho_values = [0.1, 1, 10, 1e3];
y_values = unique([0, logspace(-4, 3, 41)]);
worst_gamma_gap = 0;
three_real_cases = 0;
for gamma = gamma_values
    for rho = rho_values
        for y = y_values
            [x, objective, candidates] = prox_gamma_scalar(y, gamma, rho);
            upper = max([y + 5, 5, max(candidates) + 1]);
            grid = linspace(0, upper, 200001);
            values = (1 + gamma) .* grid ./ (gamma + grid) ...
                + 0.5 * rho .* (grid - y).^2;
            dense_best = min(values);
            gap = objective - dense_best;
            worst_gamma_gap = max(worst_gamma_gap, gap);
            tolerance = 5e-8 * max(1, abs(dense_best));
            assert(gap <= tolerance, ...
                'gamma* candidate rule failed at gamma=%g rho=%g y=%g', ...
                gamma, rho, y);
            coefficients = [rho, rho * (2 * gamma - y), ...
                rho * gamma * (gamma - 2 * y), ...
                gamma * (1 + gamma - rho * gamma * y)];
            roots_here = roots(coefficients);
            if sum(abs(imag(roots_here)) <= 1e-9 * (1 + abs(real(roots_here)))) == 3
                three_real_cases = three_real_cases + 1;
            end
            assert(x >= 0 && any(abs(candidates - x) <= 1e-12));
        end
    end
end
assert(three_real_cases > 0, 'The test grid did not exercise a three-real-root case.');

tau_values = [1e-4, 0.01, 0.2, 1, 10];
r_values = unique([0, logspace(-4, 2, 61)]);
worst_l2log_gap = 0;
for tau = tau_values
    for r = r_values
        x_vector = prox_l2log_columns(r, tau);
        x = abs(x_vector(1));
        objective = 0.5 * (x - r)^2 + tau * log1p(x);
        upper = max(r + 5, 5);
        grid = linspace(0, upper, 200001);
        dense_best = min(0.5 * (grid - r).^2 + tau * log1p(grid));
        gap = objective - dense_best;
        worst_l2log_gap = max(worst_l2log_gap, gap);
        tolerance = 5e-8 * max(1, abs(dense_best));
        assert(gap <= tolerance, ...
            'l2log candidate rule failed at tau=%g r=%g', tau, r);
    end
end

fprintf(['NONCONVEX_PROX_GLOBAL_OK three_real_cases=%d ' ...
    'worst_gamma_gap=%.3e worst_l2log_gap=%.3e\n'], ...
    three_real_cases, worst_gamma_gap, worst_l2log_gap);
end
