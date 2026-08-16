function diag_sweep()
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
Q = cell(1, numel(X));
for v = 1:numel(X)
    Q{v} = construct_gradient_operator(X{v}, cfg.knn);
end
variant = variant_config('full');
cfg.parameters.max_iter = 100;

l1s = [1e-4, 5e-4, 1e-3];
l2s = [1e-5, 1e-4, 1e-3, 1e-2];
gs  = [6, 12, 24];
fprintf('== sweep %s N=%d K=%d ==\n', ds, N, K);
for a = 1:numel(l1s)
for b = 1:numel(l2s)
for c = 1:numel(gs)
    cfg.parameters.lambda1 = l1s(a);
    cfg.parameters.lambda2 = l2s(b);
    cfg.parameters.gamma = gs(c);
    [model, history] = solve_sglog(X, Q, cfg.parameters, variant);
    W = model.affinity;
    nmi = spectral_nmi(W, gt, K);
    % block structure from first view
    Zv = abs(model.Z{1}); Zv(1:N+1:end) = 0;
    [~, ~, gtidx] = unique(gt, 'sorted');
    within = 0; cross = 0; nw = 0; nc = 0;
    for i = 1:N
        for j = 1:N
            if i ~= j
                if gtidx(i) == gtidx(j), within = within + Zv(i,j); nw = nw + 1;
                else cross = cross + Zv(i,j); nc = nc + 1; end
            end
        end
    end
    ratio = (within/max(nw,1)) / max(cross/max(nc,1), eps);
    fprintf('l1=%.0e l2=%.0e g=%-3d | NMI=%.3f | Zratio=%.2f maxZ=%.3g | %s it=%d\n', ...
        l1s(a), l2s(b), gs(c), nmi, ratio, max(Zv(:)), history.stopping_reason, history.iterations);
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
