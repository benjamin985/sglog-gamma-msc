function diag_affinity()
% Post-process the already-solved bbc/yale affinity (from diagnostics.mat) and
% test whether diffusion / reweighting recovers the lost cluster structure.
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'code')));
addpath(genpath(fullfile(root, 'common')));
maxNumCompThreads(2);

for ds = {'bbc', 'yale'}
    f = fullfile(root, 'results', 'final_20260814_retuned', ds{1}, 'full', 'diagnostics.mat');
    d = load(f);
    A = full(d.evaluation_diagnostics.affinity);
    K = d.cfg.cluster_count;
    gt = d.gt;
    N = numel(gt);
    fprintf('\n===== %s (N=%d, K=%d) =====\n', ds{1}, N, K);

    % Reported eigenvalue spectrum (normalized affinity, largest first).
    ev = d.evaluation_diagnostics.eigenvalues;
    ev = sort(real(ev(:)), 'descend');
    fprintf('  top eigenvalues (first %d): ', min(K + 2, numel(ev)));
    fprintf('%.4f ', ev(1:min(K + 2, numel(ev))));
    fprintf('\n');

    % Baseline: exact affinity from the solver.
    runs = evaluate_repeated(A, gt, K, 20);
    fprintf('  A        NMI=%.4f ARI=%.4f\n', mean(runs.nmi), mean(runs.ari));

    % Matrix-power diffusion A^t.
    At = A;
    for t = 1:3
        At = At * A;
        runs = evaluate_repeated(At, gt, K, 20);
        fprintf('  A^%d      NMI=%.4f ARI=%.4f\n', t + 1, mean(runs.nmi), mean(runs.ari));
    end

    % Row-normalized random-walk diffusion (D^-1 A)^t, symmetrized.
    D = sum(A, 2);
    P = A ./ max(D, eps);
    Pt = P;
    for t = 1:3
        Pt = Pt * P;
        At = (Pt + Pt.') / 2;
        runs = evaluate_repeated(At, gt, K, 20);
        fprintf('  (D^-1A)^%d NMI=%.4f ARI=%.4f\n', t + 1, mean(runs.nmi), mean(runs.ari));
    end

    % Entry-wise power reweighting A.^p (sharpens large affinities).
    for p = [2, 3]
        runs = evaluate_repeated(A.^p, gt, K, 20);
        fprintf('  A.^%d     NMI=%.4f ARI=%.4f\n', p, mean(runs.nmi), mean(runs.ari));
    end
end
end
