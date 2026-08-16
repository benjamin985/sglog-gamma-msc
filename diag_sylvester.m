function diag_sylvester()
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
addpath(genpath(fullfile(root_dir, 'common')));

for ds = {'yale', 'bbc', 'orl', 'scene'}
    cfg = dataset_config(root_dir, ds{1});
    [X, ~] = load_multiview_dataset(cfg);
    % Subsample scene to keep it fast and comparable to the tuning path.
    if strcmp(ds{1}, 'scene')
        set_reproducible_seed(20260813 + 5);
        keep = sort(randperm(size(X{1}, 2), 400));
        for v = 1:numel(X), X{v} = X{v}(:, keep); end
    end
    N = size(X{1}, 2);
    gram = cell(1, numel(X)); lap = cell(1, numel(X));
    for v = 1:numel(X)
        Q = construct_gradient_operator(X{v}, cfg.knn);
        gram{v} = X{v}' * X{v};
        lap{v} = Q' * Q;
    end
    alpha = 1e-4 * (1.8^10); beta = 1e-4 * (1.8^10); rho = 1e-4 * (1.8^10);
    maxdiff = 0;
    for v = 1:numel(X)
        left = alpha * gram{v} + rho * eye(N);
        right = beta * lap{v};
        forcing = randn(N);
        cache = prepare_symmetric_sylvester(gram{v}, lap{v});
        Z1 = sylvester(full(left), full(right), full(forcing));
        Z2 = solve_symmetric_sylvester_cached(cache, forcing, alpha, beta, rho);
        d = norm(Z1 - Z2, 'fro') / max(1, norm(Z1, 'fro'));
        maxdiff = max(maxdiff, d);
    end
    fprintf('%s N=%d views=%d rel_sylvester_diff=%.3e\n', ds{1}, N, numel(X), maxdiff);
end
end
