function diag_fullgrid_yale_scene()
% Label-free diagnostic scored WITH NMI (diagnosis only) over the FULL
% pre-registered 90-candidate grid for the two datasets where the eigengap
% selection rule is known to be anti-correlated with NMI (yale, scene).
% NMI is computed here ONLY to score candidate label-free criteria; the
% criterion chosen from this script must never read NMI.
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
addpath(genpath(fullfile(root_dir, 'common')));

l1_grid = [1e-3, 1e-2, 1e-1, 5e-1, 1e0];
l2_grid = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1];
g_grid  = [6, 12, 24];

specs = struct();
specs(1).name = 'yale';  specs(1).subsample = false;
specs(2).name = 'scene'; specs(2).subsample = true;
order = {'uci','bbc','yale','orl','scene'};

cols = {'dataset','lambda1','lambda2','gamma','nmi','converged','iterations', ...
    'eig_K','eig_Kp1','eigengap','diagdom','meanDiag','meanOff','svr','sv1','sv2','froZ','stopping_reason'};

for s = 1:numel(specs)
    ds = specs(s).name;
    cfg = dataset_config(root_dir, ds);
    [X, gt] = load_multiview_dataset(cfg);
    if specs(s).subsample
        di = find(strcmp(order, ds));
        set_reproducible_seed(20260813 + di);
        keep = sort(randperm(size(X{1},2), 1200));
        for v = 1:numel(X), X{v} = X{v}(:, keep); end
        gt = gt(keep);
    end
    cfg.parameters.max_iter = 100;
    Q = cell(size(X));
    for v = 1:numel(X), Q{v} = construct_gradient_operator(X{v}, cfg.knn); end
    K = cfg.cluster_count;
    N = size(X{1}, 2);
    out = fullfile(root_dir, 'tuning', sprintf('fullgrid_%s.csv', ds));
    fid = fopen(out, 'w');
    if fid < 0, error('cannot open %s', out); end
    fprintf(fid, '%s\n', strjoin(cols, ','));
    for a = 1:numel(l1_grid)
    for b = 1:numel(l2_grid)
    for c = 1:numel(g_grid)
        cfg.parameters.lambda1 = l1_grid(a);
        cfg.parameters.lambda2 = l2_grid(b);
        cfg.parameters.gamma   = g_grid(c);
        nmi = NaN; conv = 0; iters = NaN; reason = 'error';
        eig_K = NaN; eig_Kp1 = NaN;
        diagdom = NaN; meanDiag = NaN; meanOff = NaN;
        svr = NaN; sv1 = NaN; sv2 = NaN; froZ = NaN;
        try
            [model, history] = solve_sglog(X, Q, cfg.parameters, variant_config('full'));
            W = model.affinity;
            conv = double(strcmp(history.stopping_reason, 'tolerance'));
            iters = history.iterations;
            reason = history.stopping_reason;
            try
                [runs, ~, ~, ~] = evaluate_repeated(W, gt, K, 10);
                nmi = mean(runs.nmi);
            catch
                nmi = NaN;
            end
            Wn = max((W + W.')/2, 0); Wn(1:N+1:end) = 0;
            d = sum(Wn, 2);
            if min(d) > eps
                dn = spdiags(1 ./ sqrt(max(d, eps)), 0, N, N);
                nA = dn * Wn * dn; nA = sparse(real((nA + nA.')/2));
                opts.issym = true; opts.isreal = true; opts.disp = 0;
                set_reproducible_seed(20260813);
                try
                    [~, lam] = eigs(nA, min(K+1, N-1), 'la', opts);
                    lam = sort(real(diag(lam)), 'descend');
                    if numel(lam) >= K+1, eig_K = lam(K); eig_Kp1 = lam(K+1); end
                catch
                end
            end
            Z1 = abs(model.Z{1});
            off = Z1; off(1:N+1:end) = 0;
            meanDiag = mean(diag(Z1));
            meanOff = mean(off(:));
            diagdom = meanDiag / max(meanOff, eps);
            froZ = norm(model.Z{1}, 'fro');
            sv = svds(model.Z{1}, 2);
            if numel(sv) >= 2, sv1 = sv(1); sv2 = sv(2); svr = sv(2)/max(sv(1), eps); end
        catch err
            fprintf(2, '  ERR %s l1=%.2g l2=%.2g g=%g: %s\n', ds, l1_grid(a), l2_grid(b), g_grid(c), err.message);
        end
        gap = eig_K - eig_Kp1;
        vals = {ds, l1_grid(a), l2_grid(b), g_grid(c), nmi, conv, iters, ...
            eig_K, eig_Kp1, gap, diagdom, meanDiag, meanOff, svr, sv1, sv2, froZ, reason};
        fprintf(fid, '%s,%.10g,%.10g,%.10g,%.8f,%d,%g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n', ...
            vals{1}, vals{2}, vals{3}, vals{4}, vals{5}, vals{6}, vals{7}, vals{8}, ...
            vals{9}, vals{10}, vals{11}, vals{12}, vals{13}, vals{14}, vals{15}, ...
            vals{16}, vals{17}, vals{18});
        fprintf('  %s l1=%.2g l2=%.2g g=%g NMI=%.4f gap=%.4f eig_K=%.4f diagdom=%.2f svr=%.3f (%s)\n', ...
            ds, l1_grid(a), l2_grid(b), g_grid(c), nmi, gap, eig_K, diagdom, svr, reason);
    end
    end
    end
    fclose(fid);
    fprintf('wrote %s\n', out);
end
end
