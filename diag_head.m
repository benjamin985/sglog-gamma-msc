function diag_head()
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
addpath(genpath(fullfile(root_dir, 'common')));
if exist('OCTAVE_VERSION', 'builtin')
    pkg load control; pkg load statistics;
end
ds = getenv('DIAG_DS'); if isempty(ds), ds = 'bbc'; end
cfg = dataset_config(root_dir, ds);
K = cfg.cluster_count;
[X, gt] = load_multiview_dataset(cfg);
N = size(X{1}, 2);
fprintf('== dataset=%s N=%d V=%d K=%d knn=%d ==\n', ds, N, numel(X), K, cfg.knn);

% single-view RAW-data baseline
for v = 1:numel(X)
    G = X{v}' * X{v};
    G(1:N+1:end) = -inf;
    [~, nn] = sort(G, 2, 'descend');
    nn = nn(:, 1:cfg.knn);
    W = sparse(N, N);
    for i = 1:N
        W(i, nn(i, :)) = exp(G(i, nn(i,:)));
    end
    W = max(W, W');
    W(1:N+1:end) = 0;
    nmi_raw = spectral_nmi(W, gt, K);
    fprintf('  view %d raw-kNN NMI=%.4f\n', v, nmi_raw);
end

% graph connectivity
for v = 1:numel(X)
    Q = construct_gradient_operator(X{v}, cfg.knn);
    L1 = Q' * Q;
    A_adj = double(abs(L1) > 0); A_adj(1:N+1:end) = 0;
    A_adj = max(A_adj, A_adj');
    nc = conncomp_graph(A_adj);
    fprintf('  view %d mutual-kNN components=%d\n', v, nc);
end
end

function nmi = spectral_nmi(W, gt, K)
N = size(W, 1);
W = max((W + W') / 2, 0); W(1:N+1:end) = 0;
d = sum(W, 2);
if min(d) <= eps, nmi = NaN; return; end
dn = spdiags(1 ./ sqrt(max(d, eps)), 0, N, N);
nA = dn * W * dn; nA = sparse(real((nA + nA') / 2));
opts.issym = true; opts.isreal = true; opts.disp = 0;
set_reproducible_seed(20260813);
try
    [emb, ~] = eigs(nA, K, 'la', opts);
catch
    nmi = NaN; return;
end
emb = emb ./ max(sqrt(sum(emb.^2, 2)), eps);
best = 0;
for r = 1:10
    set_reproducible_seed(20260813 + r);
    lab = kmeans(emb, K, 'MaxIter', 1000, 'Replicates', 1, 'EmptyAction', 'singleton', 'Start', 'plus');
    [~, n] = compute_nmi(gt, lab);
    best = max(best, n);
end
nmi = best;
end

function c = conncomp_graph(A)
N = size(A, 1); seen = false(N, 1); c = 0;
for i = 1:N
    if ~seen(i)
        c = c + 1; stack = i; seen(i) = true;
        while ~isempty(stack)
            u = stack(end); stack(end) = [];
            nbr = find(A(:, u));
            for w = nbr'
                if ~seen(w), seen(w) = true; stack(end+1) = w; end
            end
        end
    end
end
end
