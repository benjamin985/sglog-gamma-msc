function diag_Z()
% Inspect the learned Z/E blocks directly to see whether Z collapses to a
% constant (rank-1) matrix on bbc/yale — the hypothesized root cause.
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'code')));
addpath(genpath(fullfile(root, 'common')));
maxNumCompThreads(2);

for ds = {'bbc', 'yale'}
    cfg = dataset_config(root, ds{1});
    [X, gt] = load_multiview_dataset(cfg);
    K = cfg.cluster_count;
    N = size(X{1}, 2);
    Q = cell(size(X));
    for v = 1:numel(X)
        Q{v} = construct_gradient_operator(X{v}, cfg.knn);
    end
    variant = variant_config('full');
    [model, history] = solve_sglog(X, Q, cfg.parameters, variant);
    fprintf('\n===== %s (N=%d K=%d V=%d) =====\n', ds{1}, N, K, numel(X));
    for v = 1:numel(X)
        Z = model.Z{v};
        zf = norm(Z, 'fro');
        meanZ = mean(Z(:));
        spread = norm(Z - meanZ, 'fro');
        symerr = norm(Z - Z.', 'fro') / max(1, zf);
        s = svd(Z);
        fprintf('  Z%d: ||Z||_F=%.4g mean=%.4g ||Z-mean||=%.4g symErr=%.3g top5svd=%.3g %.3g %.3g %.3g %.3g\n', ...
            v, zf, meanZ, spread, symerr, s(1), s(2), s(3), s(4), s(5));
    end
    A = model.affinity;
    meanA = mean(A(:));
    fprintf('  affinity: mean=%.4g max=%.3g nnz=%d | off-diag mean=%.4g\n', ...
        meanA, max(A(:)), nnz(A), mean(A(A ~= 0)));
    fprintf('  history objective(end)=%.6g = lam1err+lam2grad+tensor\n', ...
        history.objective(end));
    fprintf('  iters=%d stop=%s res=%.3e\n', ...
        history.iterations, history.stopping_reason, history.primal_residual(end));
end
end
