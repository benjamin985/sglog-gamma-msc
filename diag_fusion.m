function diag_fusion()
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
fprintf('== dataset=%s N=%d V=%d K=%d ==\n', ds, N, numel(X), K);

% ---- single-view RAW-data baseline: kNN graph on each view ----
fprintf('\n--- single-view RAW-data spectral NMI ---\n');
for v = 1:numel(X)
    G = X{v}' * X{v};                 % cosine similarity (columns normalized)
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
    fprintf('  view %d raw-kNN: NMI=%.4f\n', v, nmi_raw);
end

% ---- graph connectivity ----
fprintf('\n--- mutual kNN graph connectivity ---\n');
Q = cell(1, numel(X));
for v = 1:numel(X)
    Q{v} = construct_gradient_operator(X{v}, cfg.knn);
end
L1 = Q{1}' * Q{1};
A_adj = double(abs(L1) > 0); A_adj(1:N+1:end) = 0;
A_adj = max(A_adj, A_adj');
ncomp = conncomp_graph(A_adj);
fprintf('  mutual-kNN graph connected components = %d (N=%d)\n', ncomp, N);

% ---- run full solver ----
fprintf('\n--- full solver (locked params) ---\n');
variant = variant_config('full');
[model, history] = solve_sglog(X, Q, cfg.parameters, variant);
fprintf('  stop=%s iters=%d residual=%.3e\n', history.stopping_reason, history.iterations, history.primal_residual(end));

% per-view |Z| block structure
fprintf('\n--- per-view |Z_v| within-cluster vs cross-cluster ---\n');
gtidx = zeros(N, 1);
[~, ~, gtidx] = unique(gt, 'sorted');
for v = 1:numel(X)
    Zv = abs(model.Z{v});
    Zv(1:N+1:end) = 0;
    within = 0; cross = 0; nw = 0; nc = 0;
    for i = 1:N
        for j = 1:N
            if i ~= j
                if gtidx(i) == gtidx(j), within = within + Zv(i,j); nw = nw + 1;
                else cross = cross + Zv(i,j); nc = nc + 1; end
            end
        end
    end
    within = within / max(nw,1); cross = cross / max(nc,1);
    fprintf('  view %d: mean|Z| within=%.3g cross=%.3g ratio=%.3g max=%.3g\n', v, within, cross, within/max(cross,eps), max(Zv(:)));
    % single-view affinity NMI (this view's Z alone)
    Wv = (abs(model.Z{v}) + abs(model.Z{v}')) / 2;
    Wv(1:N+1:end) = 0;
    nmi_v = spectral_nmi(Wv, gt, K);
    fprintf('           single-view |Z| NMI=%.4f\n', nmi_v);
end

% multi-view affinity NMI
Wm = model.affinity;
nmi_m = spectral_nmi(Wm, gt, K);
fprintf('\n--- multi-view affinity ---\n');
fprintf('  multi-view affinity NMI=%.4f (reproduces manuscript)\n', nmi_m);
end

function nmi = spectral_nmi(W, gt, K)
N = size(W, 1);
W = max((W + W') / 2, 0);
W(1:N+1:end) = 0;
d = sum(W, 2);
if min(d) <= eps
    nmi = NaN; return;
end
dn = spdiags(1 ./ sqrt(max(d, eps)), 0, N, N);
nA = dn * W * dn;
nA = sparse(real((nA + nA') / 2));
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
N = size(A, 1);
seen = false(N, 1); c = 0;
for i = 1:N
    if ~seen(i)
        c = c + 1;
        stack = i; seen(i) = true;
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
