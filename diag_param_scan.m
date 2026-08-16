function diag_param_scan()
% Sweep lambda1 (self-representation sparsity) and gamma (tensor low-rank)
% on bbc/yale to find whether the rank-1 Z collapse can be reversed, and
% which parameter is the dominant lever.
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'code')));
addpath(genpath(fullfile(root, 'common')));
maxNumCompThreads(2);

ds_list = {'scene'};
l1_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e0];
g_list  = [24];

for ds = ds_list
    cfg = dataset_config(root, ds{1});
    [X, gt] = load_multiview_dataset(cfg);
    K = cfg.cluster_count;
    % Subsample large datasets so the scan stays fast (mirrors tuning path).
    n0 = size(X{1}, 2);
    if n0 > 400
        set_reproducible_seed(20260813 + find(strcmp({'uci','bbc','yale','orl','scene'}, ds{1})));
        keep = sort(randperm(n0, 400));
        for v = 1:numel(X), X{v} = X{v}(:, keep); end
        gt = gt(keep);
    end
    Q = cell(size(X));
    for v = 1:numel(X)
        Q{v} = construct_gradient_operator(X{v}, cfg.knn);
    end
    variant = variant_config('full');
    fprintf('\n===== %s scan (lambda2=%.3g fixed) =====\n', ds{1}, cfg.parameters.lambda2);
    for gi = 1:numel(g_list)
        for li = 1:numel(l1_list)
            p = cfg.parameters;
            p.lambda1 = l1_list(li);
            p.gamma = g_list(gi);
            [model, history] = solve_sglog(X, Q, p, variant);
            s = svd(model.Z{1});
            rank2 = s(2) / max(s(1), eps);
            runs = evaluate_repeated(model.affinity, gt, K, 20);
            fprintf('  l1=%.2g g=%.3g: NMI=%.4f rank2/1=%.3g ||Z||=%.3g obj=%.3g\n', ...
                p.lambda1, p.gamma, mean(runs.nmi), rank2, ...
                norm(model.Z{1}, 'fro'), history.objective(end));
        end
    end
end
end
