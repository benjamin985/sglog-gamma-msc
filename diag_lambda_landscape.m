function diag_lambda_landscape()
% Post-hoc diagnostic used ONLY to design/validate a label-free selection
% criterion. NMI is computed here solely to score the candidate criteria; the
% criterion chosen from this script must never read NMI and is applied later
% without any label. Sweeps lambda1 x lambda2 (gamma fixed at the currently
% selected value) for all five datasets, recording NMI plus several label-free
% candidate criteria so we can find one that tracks clustering quality better
% than the raw eigengap (which is anti-correlated with NMI on yale/scene).
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
addpath(genpath(fullfile(root_dir, 'common')));

specs = struct();
% Grids sweep collapse -> healthy -> over-regularized. gamma is held at the
% eigengap-selected value (the lambda1/lambda2 axes are the mis-selected ones).
specs(1).name = 'uci';    specs(1).l1 = [1e-3, 1e-2, 1e-1, 5e-1];
specs(1).l2 = [1e-5, 1e-4, 1e-3, 1e-2]; specs(1).gamma = [12]; specs(1).subsample = true;
specs(2).name = 'bbc';    specs(2).l1 = [1e-3, 1e-2, 1e-1, 5e-1];
specs(2).l2 = [1e-5, 1e-4, 1e-3, 1e-2]; specs(2).gamma = [6];  specs(2).subsample = false;
specs(3).name = 'yale';   specs(3).l1 = [5e-2, 1e-1, 5e-1, 1e0, 2e0];
specs(3).l2 = [1e-4, 1e-3, 1e-2, 5e-2]; specs(3).gamma = [24]; specs(3).subsample = false;
specs(4).name = 'orl';    specs(4).l1 = [1e-2, 1e-1, 5e-1, 1e0];
specs(4).l2 = [1e-5, 1e-4, 1e-3, 1e-2]; specs(4).gamma = [24]; specs(4).subsample = false;
specs(5).name = 'scene';  specs(5).l1 = [1e-4, 5e-4, 1e-3, 1e-2, 1e-1];
specs(5).l2 = [1e-5, 1e-4, 1e-3, 1e-2]; specs(5).gamma = [12]; specs(5).subsample = true;

order = {'uci','bbc','yale','orl','scene'};
cols = {'dataset','lambda1','lambda2','gamma','nmi','converged','iterations', ...
    'final_primal_residual','eig_K','eig_Kp1','eigengap','eig_ratio','eig1', ...
    'norm_gap','diagdom','meanDiag','meanOff','svr','sv1','sv2','froZ','stopping_reason'};

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
    out = fullfile(root_dir, 'tuning', sprintf('diag_landscape_%s.csv', ds));
    fid = fopen(out, 'w');
    if fid < 0, error('cannot open %s', out); end
    fprintf(fid, '%s\n', strjoin(cols, ','));
    for a = 1:numel(specs(s).l1)
    for b = 1:numel(specs(s).l2)
    for c = 1:numel(specs(s).gamma)
        cfg.parameters.lambda1 = specs(s).l1(a);
        cfg.parameters.lambda2 = specs(s).l2(b);
        cfg.parameters.gamma   = specs(s).gamma(c);
        nmi = NaN; conv = 0; iters = NaN; resid = NaN; reason = 'error';
        eig_K = NaN; eig_Kp1 = NaN; eig1 = NaN;
        diagdom = NaN; meanDiag = NaN; meanOff = NaN;
        svr = NaN; sv1 = NaN; sv2 = NaN; froZ = NaN;
        try
            [model, history] = solve_sglog(X, Q, cfg.parameters, variant_config('full'));
            W = model.affinity;
            conv = double(strcmp(history.stopping_reason, 'tolerance'));
            iters = history.iterations;
            resid = history.primal_residual(end);
            reason = history.stopping_reason;
            % NMI (diagnosis only).
            try
                [runs, ~, ~, ~] = evaluate_repeated(W, gt, K, 10);
                nmi = mean(runs.nmi);
            catch
                nmi = NaN;
            end
            % Label-free eigenvalue criteria on the normalized affinity.
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
                    if numel(lam) >= 1, eig1 = lam(1); end
                    if numel(lam) >= K+1, eig_K = lam(K); eig_Kp1 = lam(K+1); end
                catch
                end
            end
            % Z-structure criteria (first view).
            Z1 = abs(model.Z{1});
            off = Z1; off(1:N+1:end) = 0;
            meanDiag = mean(diag(Z1));
            meanOff = mean(off(:));
            diagdom = meanDiag / max(meanOff, eps);
            froZ = norm(model.Z{1}, 'fro');
            sv = svds(model.Z{1}, 2);
            if numel(sv) >= 2, sv1 = sv(1); sv2 = sv(2); svr = sv(2)/max(sv(1), eps); end
        catch err
            fprintf(2, '  ERR %s l1=%.2g l2=%.2g: %s\n', ds, specs(s).l1(a), specs(s).l2(b), err.message);
        end
        gap = eig_K - eig_Kp1;
        ratio = eig_K / max(eig_Kp1, eps);
        normgap = gap / max(eig1, eps);
        vals = {ds, specs(s).l1(a), specs(s).l2(b), specs(s).gamma(c), ...
            nmi, conv, iters, resid, eig_K, eig_Kp1, gap, ratio, eig1, ...
            normgap, diagdom, meanDiag, meanOff, svr, sv1, sv2, froZ, reason};
        fprintf(fid, '%s,%.10g,%.10g,%.10g,%.8f,%d,%g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n', ...
            vals{1}, vals{2}, vals{3}, vals{4}, vals{5}, vals{6}, vals{7}, vals{8}, ...
            vals{9}, vals{10}, vals{11}, vals{12}, vals{13}, vals{14}, vals{15}, ...
            vals{16}, vals{17}, vals{18}, vals{19}, vals{20}, vals{21}, vals{22});
        fprintf('  %s l1=%.2g l2=%.2g g=%g NMI=%.4f gap=%.4f ratio=%.4f diagdom=%.3f svr=%.3f (%s)\n', ...
            ds, specs(s).l1(a), specs(s).l2(b), specs(s).gamma(c), nmi, gap, ratio, diagdom, svr, reason);
    end
    end
    end
    fclose(fid);
    fprintf('wrote %s\n', out);
end
end
