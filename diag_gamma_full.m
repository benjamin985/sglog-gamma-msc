function diag_gamma_full()
% Full-dataset gamma diagnostic (label-free). Sweeps ONLY the tensor gamma*
% shape parameter at the label-free-selected (locked) lambda1/lambda2, on the
% FULL dataset (no subsampling), to determine whether a label-free diagnostic
% can rank gamma consistently with clustering quality. NMI/ACC are computed
% here ONLY to score candidate criteria; the criterion finally adopted must
% never read them at selection time.
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
addpath(genpath(fullfile(root_dir, 'common')));

specs = struct();
specs(1).name = 'scene'; specs(1).gamma_grid = [6, 12, 24, 48, 96];
specs(2).name = 'yale';  specs(2).gamma_grid = [6, 12, 24, 48];

cols = {'dataset','gamma','nmi','acc','converged','iterations', ...
    'eig_K','eig_Kp1','eigengap','diagdom','meanDiag','meanOff','svr','sv1','sv2','froZ','stopping_reason'};

for s = 1:numel(specs)
    ds = specs(s).name;
    cfg = dataset_config(root_dir, ds);
    [X, gt] = load_multiview_dataset(cfg);
    Q = cell(size(X));
    for v = 1:numel(X), Q{v} = construct_gradient_operator(X{v}, cfg.knn); end
    K = cfg.cluster_count;
    N = size(X{1}, 2);
    out = fullfile(root_dir, 'tuning', sprintf('gamma_full_%s.csv', ds));
    fid = fopen(out, 'w');
    fprintf(fid, '%s\n', strjoin(cols, ','));
    for g = specs(s).gamma_grid
        cfg.parameters.gamma = g;
        fprintf('\n=== %s / gamma=%.6g (full) ===\n', ds, g);
        [model, history] = solve_sglog(X, Q, cfg.parameters, variant_config('full'));
        W = model.affinity;
        [runs, ~, ~, ~] = evaluate_repeated(W, gt, K, 20);
        nmi = mean(runs.nmi); acc = mean(runs.acc);
        conv = double(strcmp(history.stopping_reason, 'tolerance'));
        Wn = max((W + W.')/2, 0); Wn(1:N+1:end) = 0;
        d = sum(Wn, 2);
        eig_K = NaN; eig_Kp1 = NaN;
        if min(d) > eps
            dn = spdiags(1 ./ sqrt(max(d, eps)), 0, N, N);
            nA = dn * Wn * dn; nA = sparse(real((nA + nA.')/2));
            opts.issym = true; opts.isreal = true; opts.disp = 0;
            set_reproducible_seed(20260813);
            [~, lam] = eigs(nA, min(K+1, N-1), 'la', opts);
            lam = sort(real(diag(lam)), 'descend');
            if numel(lam) >= K+1, eig_K = lam(K); eig_Kp1 = lam(K+1); end
        end
        Z1 = abs(model.Z{1});
        off = Z1; off(1:N+1:end) = 0;
        meanDiag = mean(diag(Z1));
        meanOff = mean(off(:));
        diagdom = meanDiag / max(meanOff, eps);
        froZ = norm(model.Z{1}, 'fro');
        sv = svds(model.Z{1}, 2);
        sv1 = sv(1); sv2 = sv(2); svr = sv(2)/max(sv(1), eps);
        gap = eig_K - eig_Kp1;
        fprintf('   NMI=%.4f ACC=%.4f eig_K=%.4f gap=%.4f meanDiag=%.4f svr=%.4f diagdom=%.2f (%s)\n', ...
            nmi, acc, eig_K, gap, meanDiag, svr, diagdom, history.stopping_reason);
        vals = {ds, g, nmi, acc, conv, history.iterations, eig_K, eig_Kp1, gap, ...
            diagdom, meanDiag, meanOff, svr, sv1, sv2, froZ, history.stopping_reason};
        fprintf(fid, '%s,%.10g,%.8f,%.8f,%d,%g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n', ...
            vals{:});
    end
    fclose(fid);
    fprintf('wrote %s\n', out);
end
end
