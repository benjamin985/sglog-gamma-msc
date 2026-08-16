function diag_graph()
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
variant = variant_config('full');
cfg.parameters.max_iter = 60;

% manuscript locked params (Table 3)
switch ds
    case 'bbc',  lp = [1e-4, 8.1e-5, 6.43];
    case 'yale', lp = [5e-4, 5e-2, 12];
    case 'orl',  lp = [5e-4, 1e-2, 12];
    case 'uci',  lp = [1e-4, 5e-3, 12];
    case 'scene',lp = [5e-4, 1e-2, 12];
    otherwise,  lp = [cfg.parameters.lambda1, cfg.parameters.lambda2, cfg.parameters.gamma];
end
cfg.parameters.lambda1 = lp(1);
cfg.parameters.lambda2 = lp(2);
cfg.parameters.gamma = lp(3);

configs = {...
    'mutual k=8',  8,  true; ...
    'mutual k=16', 16, true; ...
    'mutual k=32', 32, true; ...
    'full   k=8',  8,  false; ...
    'full   k=16', 16, false; ...
    'full   k=32', 32, false};
fprintf('== diag_graph %s N=%d V=%d K=%d locked=[%.0e %.0e %.2g] ==\n', ds, N, numel(X), K, lp);

for ci = 1:size(configs,1)
    name = configs{ci,1}; k = configs{ci,2}; mutual = configs{ci,3};
    Q = cell(1, numel(X)); ncomp = zeros(1, numel(X));
    for v = 1:numel(X)
        Q{v} = build_incidence(X{v}, k, mutual);
        L1 = Q{v}' * Q{v};
        Adj = double(abs(L1) > 0); Adj(1:N+1:end) = 0; Adj = max(Adj, Adj');
        ncomp(v) = conncomp(Adj);
    end
    [model, history] = solve_sglog(X, Q, cfg.parameters, variant);
    W = model.affinity;
    nmi = spectral_nmi(W, gt, K);
    Z1 = abs(model.Z{1}); off = Z1; off(1:N+1:end) = 0;
    fprintf('%-12s | comps=%s | NMI=%.3f | meanDiag=%.3g meanOff=%.3g maxOff=%.3g | %s it=%d\n', ...
        name, mat2str(ncomp), nmi, mean(diag(Z1)), mean(off(:)), max(off(:)), ...
        history.stopping_reason, history.iterations);
end
end

function Q = build_incidence(X, k, mutual)
N = size(X, 2);
k = min(k, N - 1);
gram = X.' * X;
norm2 = sum(X.^2, 1);
distance2 = max(0, norm2.' + norm2 - 2 * gram);
distance2(1:N+1:end) = inf;
[sorted_distance, neighbors] = sort(distance2, 2, 'ascend');
neighbors = neighbors(:, 1:k);
sorted_distance = sorted_distance(:, 1:k);
sigma2 = median(sorted_distance(isfinite(sorted_distance) & sorted_distance > 0));
if isempty(sigma2) || sigma2 <= 0, sigma2 = 1; end
directed_mask = sparse(N, N);
for i = 1:N
    directed_mask(i, neighbors(i, :)) = 1;
end
if mutual
    edge_mask = spones(directed_mask .* directed_mask.');
else
    edge_mask = spones(directed_mask + directed_mask.');
end
[source, target] = find(triu(edge_mask, 1));
if isempty(source)
    error('graph has no edges; increase k');
end
weights = exp(-distance2(sub2ind([N, N], source, target)) / (2 * sigma2));
weights = sqrt(max(weights, eps));
M = numel(source);
rows = repelem((1:M).', 2, 1);
cols = reshape([source, target].', [], 1);
vals = reshape([weights, -weights].', [], 1);
Q = sparse(rows, cols, vals, M, N);
end

function c = conncomp(A)
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
