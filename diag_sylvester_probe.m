function diag_sylvester_probe()
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
addpath(genpath(fullfile(root_dir, 'common')));
if exist('OCTAVE_VERSION', 'builtin')
    pkg load control;
    pkg load statistics;
end

ds = getenv('DIAG_DS'); if isempty(ds), ds = 'orl'; end
cfg = dataset_config(root_dir, ds);
[X, gt] = load_multiview_dataset(cfg); %#ok<NASGU>
N = size(X{1}, 2);
alpha = 1e-4; rho = 1e-4; beta = 1e-4;

for v = 1:numel(X)
    Q = construct_gradient_operator(X{v}, cfg.knn);
    gram = X{v}' * X{v};
    L = Q' * Q;
    gram = (full(gram) + full(gram')) / 2;
    L = (full(L) + full(L')) / 2;

    % SVD-based (robust orthogonal) diagonalization
    [U, Sg] = svd(gram); ge = diag(Sg);
    [V, Sl] = svd(L);    le = diag(Sl);
    orthU = norm(U'*U - eye(N), 'fro');
    recg = norm(gram - U*diag(ge)*U', 'fro') / max(1, norm(gram, 'fro'));
    fprintf('view %d: SVD orthU=%.3g recgram=%.3g gram_eig[min,max]=[%.4g, %.4g]\n', ...
        v, orthU, recg, min(ge), max(ge));

    left_values = alpha*ge + rho;
    right_values = beta*le;
    denom = bsxfun(@plus, left_values, right_values');
    forcing = alpha * gram;
    Zs = U * ((U' * forcing * V) ./ denom) * V';
    rs = norm((alpha*gram+rho*eye(N))*Zs + Zs*(beta*L) - forcing, 'fro');
    fprintf('          SVD spectral residual=%.4g |Z|max=%.4g\n', rs, max(abs(Zs(:))));
end
end
