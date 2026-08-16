function diag_twist()
% A/B the tensor twist convention [2,3,1] (transposed) vs [1,3,2] (untransposed)
% on the two anomalous datasets, re-solving with the eigengap-selected params.
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'code')));
addpath(genpath(fullfile(root, 'common')));
maxNumCompThreads(2);

for ds = {'bbc', 'yale'}
    cfg = dataset_config(root, ds{1});
    [X, gt] = load_multiview_dataset(cfg);
    K = cfg.cluster_count;
    Q = cell(size(X));
    for v = 1:numel(X)
        Q{v} = construct_gradient_operator(X{v}, cfg.knn);
    end
    variant = variant_config('full');
    for tw = {'[2,3,1]', '[1,3,2]', '[1,2,3]'}
        variant.twist = str2num(tw{1}); %#ok<ST2NM>
        [model, history] = solve_sglog(X, Q, cfg.parameters, variant);
        runs = evaluate_repeated(model.affinity, gt, K, 20);
        fprintf('%s twist=%s NMI=%.4f ARI=%.4f ACC=%.4f iters=%d\n', ...
            ds{1}, tw{1}, mean(runs.nmi), mean(runs.ari), ...
            mean(runs.acc), history.iterations);
    end
end
end
