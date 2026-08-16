function diag_singleview()
% Single-view spectral-clustering baselines for bbc/yale, to verify whether
% the multi-view fusion is actually below single-view (the anomaly hypothesis).
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'code')));
addpath(genpath(fullfile(root, 'common')));
maxNumCompThreads(2);

for ds = {'bbc', 'yale'}
    cfg = dataset_config(root, ds{1});
    [X, gt] = load_multiview_dataset(cfg);
    K = cfg.cluster_count;
    N = size(X{1}, 2);
    fprintf('\n===== %s (N=%d, K=%d, V=%d) =====\n', ds{1}, N, K, numel(X));

    for v = 1:numel(X)
        A = knn_gaussian_affinity(X{v}, cfg.knn);
        runs = evaluate_repeated(A, gt, K, 20);
        fprintf('  view%d kNN k=%d : NMI=%.4f ARI=%.4f ACC=%.4f\n', ...
            v, cfg.knn, mean(runs.nmi), mean(runs.ari), mean(runs.acc));
    end

    A = knn_gaussian_affinity(vertcat(X{:}), cfg.knn);
    runs = evaluate_repeated(A, gt, K, 20);
    fprintf('  concat kNN k=%d: NMI=%.4f ARI=%.4f ACC=%.4f\n', ...
        cfg.knn, mean(runs.nmi), mean(runs.ari), mean(runs.acc));

    for v = 1:numel(X)
        A = gaussian_affinity(X{v});
        runs = evaluate_repeated(A, gt, K, 20);
        fprintf('  view%d gaussian : NMI=%.4f ARI=%.4f ACC=%.4f\n', ...
            v, mean(runs.nmi), mean(runs.ari), mean(runs.acc));
    end

    A = gaussian_affinity(vertcat(X{:}));
    runs = evaluate_repeated(A, gt, K, 20);
    fprintf('  concat gaussian: NMI=%.4f ARI=%.4f ACC=%.4f\n', ...
        mean(runs.nmi), mean(runs.ari), mean(runs.acc));
end
end

function A = knn_gaussian_affinity(X, k)
N = size(X, 2);
k = min(k, N - 1);
D2 = pdist2(X.', X.').^2;
D2(1:N+1:end) = inf;
[~, nb] = sort(D2, 2, 'ascend');
nb = nb(:, 1:k);
sigma2 = median(D2(isfinite(D2) & D2 > 0));
if isempty(sigma2) || sigma2 <= 0, sigma2 = 1; end
W = zeros(N);
for i = 1:N
    W(i, nb(i, :)) = exp(-D2(i, nb(i, :)) / (2 * sigma2));
end
W = max(W, W.');
W(1:N+1:end) = 0;
A = sparse(W);
end

function A = gaussian_affinity(X)
N = size(X, 2);
D2 = pdist2(X.', X.').^2;
D2(1:N+1:end) = 0;
sigma2 = median(D2(isfinite(D2) & D2 > 0));
if isempty(sigma2) || sigma2 <= 0, sigma2 = 1; end
W = exp(-D2 / (2 * sigma2));
W(1:N+1:end) = 0;
A = sparse(W);
end
