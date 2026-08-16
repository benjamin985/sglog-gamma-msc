function diag_eigengap_check()
% Verify the eigengap selection rule actually picks the healthy lambda1 once
% the grid is widened. For each dataset print eigengap and NMI side by side
% over the new lambda1 grid (gamma/lambda2 fixed at the locked values).
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'code')));
addpath(genpath(fullfile(root, 'common')));
maxNumCompThreads(2);

for ds = {'bbc', 'yale', 'orl'}
    cfg = dataset_config(root, ds{1});
    [X, gt] = load_multiview_dataset(cfg);
    K = cfg.cluster_count;
    Q = cell(size(X));
    for v = 1:numel(X)
        Q{v} = construct_gradient_operator(X{v}, cfg.knn);
    end
    variant = variant_config('full');
    l1_list = [1e-3, 1e-2, 1e-1, 1e0];
    fprintf('\n===== %s (K=%d lambda2=%.3g gamma=%.3g) =====\n', ...
        ds{1}, K, cfg.parameters.lambda2, cfg.parameters.gamma);
    for li = 1:numel(l1_list)
        p = cfg.parameters;
        p.lambda1 = l1_list(li);
        [model, history] = solve_sglog(X, Q, p, variant);
        W = model.affinity;
        gap = affinity_eigengap(W, K);
        runs = evaluate_repeated(W, gt, K, 20);
        fprintf('  l1=%.3g: eigengap=% .5f  NMI=%.4f\n', ...
            p.lambda1, gap, mean(runs.nmi));
    end
end
end

function gap = affinity_eigengap(W, K)
N = size(W, 1);
degree = sum(W, 2);
if min(degree) <= eps
    gap = -inf;
    return;
end
norm = spdiags(1 ./ sqrt(max(degree, eps)), 0, N, N);
nA = norm * W * norm;
nA = sparse(real((nA + nA.') / 2));
kk = min(K + 1, N - 1);
opts.issym = true; opts.isreal = true; opts.disp = 0;
try
    [~, lam] = eigs(nA, kk, 'la', opts);
    lam = sort(real(diag(lam)), 'descend');
    gap = lam(K) - lam(K + 1);
catch
    gap = -inf;
end
end
