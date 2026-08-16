function audit_recompute()
% Recompute every reported mean/std via code/summarize_runs.m from the saved
% per-run metrics_all_runs.csv, and compare against the recorded summary.csv.
% Prints the maximum discrepancy (should reproduce the 3.33e-16 figure cited in
% the response) and writes results/final_20260815_retuned_v2/audit_comparison.csv.
    addpath('code');
    datasets = {'uci','bbc','orl','scene','yale'};
    variants = {'full','no_gradient','no_l2log','tnn'};
    metrics  = {'nmi','ari','acc','recall','precision','fscore'};
    base = 'results/final_20260815_retuned_v2';

    maxd = 0.0;
    worst = '';
    out_rows = {};

    for i = 1:numel(datasets)
        for j = 1:numel(variants)
            dsname = datasets{i}; v = variants{j};
            d = fullfile(base, dsname, v);
            T = readtable(fullfile(d, 'metrics_all_runs.csv'));
            runs = struct();
            for k = 1:numel(metrics)
                runs.(metrics{k}) = T.(metrics{k});
            end
            s = summarize_runs(runs);
            R = readtable(fullfile(d, 'summary.csv'));
            for k = 1:numel(metrics)
                m = metrics{k};
                rm = s.([m '_mean']); rs = s.([m '_std']);
                cm = R.([m '_mean'])(1); cs = R.([m '_std'])(1);
                dmean = abs(rm - cm); dstd = abs(rs - cs);
                out_rows(end+1, :) = {dsname, v, m, rm, cm, dmean, rs, cs, dstd}; %#ok<AGROW>
                if dmean > maxd, maxd = dmean; worst = sprintf('%s/%s %s mean', dsname, v, m); end
                if dstd  > maxd, maxd = dstd;  worst = sprintf('%s/%s %s std', dsname, v, m); end
            end
        end
    end

    T = cell2table(out_rows, 'VariableNames', ...
        {'dataset','variant','metric','recomputed_mean','recorded_mean','d_mean', ...
         'recomputed_std','recorded_std','d_std'});
    writetable(T, fullfile(base, 'audit_comparison.csv'));

    fprintf('MAX_DISCREPANCY=%.17g\n', maxd);
    fprintf('WORST=%s\n', worst);
end
