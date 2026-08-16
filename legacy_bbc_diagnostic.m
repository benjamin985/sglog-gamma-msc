function legacy_bbc_diagnostic()
% Faithful BBCSport legacy-path diagnostic. NOT FOR MANUSCRIPT REPORTING.
root_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(root_dir, 'code'));
addpath(genpath(fullfile(root_dir, 'common')));
if exist('OCTAVE_VERSION', 'builtin')
    pkg load control;
    pkg load statistics;
end

data = load(fullfile(root_dir, 'datasets', 'bbcsport_2view.mat'));
graph_data = load(fullfile(root_dir, 'legacy_diagnostic', ...
    'bbcsport_2view_A.mat'));
X = {normalize_columns(double(data.X1)), normalize_columns(double(data.X2))};
gt = double(data.gt(:));
A = graph_data.A;

parameters.lambda1 = 1e-4;
parameters.lambda2 = 8.1e-5;
parameters.gamma = 6.43;
parameters.mu10 = 1e-4;
parameters.mu20 = 1e-4;
parameters.rho0 = 1e-4;
parameters.penalty_growth = 1.8;
parameters.penalty_max = 1e11;
parameters.tolerance = 1e-5;
parameters.max_iter = 201;

[model, history] = legacy_solve(X, A, parameters);
[runs, assignments, cluster_sizes, evaluation_diagnostics] = ...
    evaluate_repeated(model.affinity, gt, numel(unique(gt)), 20);
summary = summarize_runs(runs);
summary.iterations = history.iterations;
summary.wall_seconds = history.wall_seconds;
summary.final_primal_residual = history.primal_residual(end);
summary.final_successive_difference = history.successive_difference(end);
summary.stopping_reason = history.stopping_reason;

legacy_labels = legacy_spectral(model.affinity, numel(unique(gt)));
[~, legacy_nmi] = compute_nmi(gt, legacy_labels);
legacy_acc = Accuracy(legacy_labels, gt);
[legacy_f, legacy_precision, legacy_recall] = compute_f(gt, legacy_labels);
legacy_ari = RandIndex(gt, legacy_labels);

out_dir = fullfile(root_dir, 'results', 'legacy_diagnostic', 'bbc');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
write_struct_csv(runs, fullfile(out_dir, 'fixed_seed_metrics.csv'));
write_struct_csv(summary, fullfile(out_dir, 'fixed_seed_summary.csv'));
legacy_metrics = struct('nmi', legacy_nmi, 'ari', legacy_ari, ...
    'acc', legacy_acc, 'recall', legacy_recall, ...
    'precision', legacy_precision, 'fscore', legacy_f);
write_struct_csv(legacy_metrics, fullfile(out_dir, 'legacy_replicates20.csv'));
save_compat(fullfile(out_dir, 'diagnostics.mat'), 'model', 'history', ...
    'runs', 'assignments', 'cluster_sizes', 'evaluation_diagnostics', ...
    'legacy_labels', 'legacy_metrics', 'parameters', 'gt');
fid = fopen(fullfile(out_dir, 'NOT_FOR_REPORTING.txt'), 'w');
fprintf(fid, ['This diagnostic intentionally uses the undocumented legacy ', ...
    'bbcsport_2view_A.mat graph, per-view elementwise log shrinkage, the ', ...
    'legacy gamma root heuristic, and the legacy stopping rule. It is not ', ...
    'eligible for the revised manuscript.\n']);
fclose(fid);
fprintf('LEGACY_DIAGNOSTIC_DONE NMI=%.6f ACC=%.6f iterations=%d\n', ...
    legacy_nmi, legacy_acc, history.iterations);
end

function [model, history] = legacy_solve(X, A, p)
V = numel(X); N = size(X{1}, 2);
Z = cell(1,V); G = cell(1,V); W = cell(1,V); E = cell(1,V);
Y = cell(1,V); J = cell(1,V); B = cell(1,V);
for v = 1:V
    Z{v}=zeros(N); G{v}=zeros(N); W{v}=zeros(N);
    E{v}=zeros(size(X{v})); Y{v}=zeros(size(X{v}));
    J{v}=zeros(N,size(A{v},1)); B{v}=zeros(size(J{v}));
end
mu1=p.mu10; mu2=p.mu20; rho=p.rho0;
history.reconstruction_residual=nan(p.max_iter,1);
history.gradient_residual=nan(p.max_iter,1);
history.tensor_residual=nan(p.max_iter,1);
history.successive_difference=nan(p.max_iter,1);
start_time=tic; converged=false;
for iteration=1:p.max_iter
    Z_previous=Z;
    for v=1:V
        left=mu1*(X{v}'*X{v})+rho*eye(N);
        right=mu2*(A{v}'*A{v});
        forcing=(B{v}+mu2*J{v})*A{v}+X{v}'*Y{v} ...
            +mu1*X{v}'*(X{v}-E{v})-W{v}+rho*G{v};
        Z{v}=sylvester(full(left),full(right),full(forcing));
    end
    for v=1:V
        E{v}=legacy_log_shrink(X{v}-X{v}*Z{v}+Y{v}/mu1,p.lambda1/mu1);
        J{v}=soft_threshold(Z{v}*A{v}'-B{v}/mu2,p.lambda2/mu2);
    end
    % Exact legacy shiftdim(X,1) convention: T(i,v,j)=Z_v(j,i).
    Z_tensor=permute(cat(3,Z{:}),[2,3,1]);
    W_tensor=permute(cat(3,W{:}),[2,3,1]);
    G_tensor=legacy_gamma_tensor(Z_tensor+W_tensor/rho,p.gamma,rho);
    G_slices=permute(G_tensor,[3,1,2]);
    max_rec=0; max_grad=0; max_tensor=0;
    for v=1:V
        G{v}=G_slices(:,:,v);
        rec=X{v}-X{v}*Z{v}-E{v};
        grad=J{v}-Z{v}*A{v}'; tensor=Z{v}-G{v};
        Y{v}=Y{v}+mu1*rec; B{v}=B{v}+mu2*grad; W{v}=W{v}+rho*tensor;
        max_rec=max(max_rec,norm(rec,inf));
        max_grad=max(max_grad,norm(grad,inf));
        max_tensor=max(max_tensor,norm(tensor,inf));
    end
    change=max(cellfun(@(a,b)norm(a-b,'fro')/max(1,norm(b,'fro')),Z,Z_previous));
    history.reconstruction_residual(iteration)=max_rec;
    history.gradient_residual(iteration)=max_grad;
    history.tensor_residual(iteration)=max_tensor;
    history.successive_difference(iteration)=change;
    fprintf('legacy_iter=%d rec=%.3e grad=%.3e tensor=%.3e change=%.3e\n', ...
        iteration,max_rec,max_grad,max_tensor,change);
    if max_rec<p.tolerance, converged=true; break; end
    mu1=min(mu1*p.penalty_growth,p.penalty_max);
    mu2=min(mu2*p.penalty_growth,p.penalty_max);
    rho=min(rho*p.penalty_growth,p.penalty_max);
end
fields=fieldnames(history);
for i=1:numel(fields), history.(fields{i})=history.(fields{i})(1:iteration); end
history.primal_residual=max([history.reconstruction_residual, ...
    history.gradient_residual,history.tensor_residual],[],2);
history.objective=nan(iteration,1);
history.iterations=iteration; history.wall_seconds=toc(start_time);
if converged, history.stopping_reason='legacy_reconstruction_only';
else, history.stopping_reason='maximum_iterations'; end
affinity=zeros(N);
for v=1:V, affinity=affinity+abs(Z{v})+abs(Z{v}'); end
affinity(1:N+1:end)=0;
model=struct('Z',{Z},'E',{E},'G',{G},'affinity',affinity);
end

function X = legacy_log_shrink(Y,lambda)
X=zeros(size(Y)); delta=(1+abs(Y)).^2-4*lambda;
id=find(delta>0); Y1=Y(id);
X0=(abs(Y1)-1)/2+sqrt(delta(id))/2.*sign(Y1);
keep=((X0-Y1).^2/2+lambda*log(1+abs(X0)))<Y1.^2/2;
id=id(keep); X0=X0(keep); same=sign(X0)==sign(Y(id));
X(id(same))=X0(same);
end

function G = legacy_gamma_tensor(M,gamma,rho)
Mhat=fft(M,[],3); Ghat=complex(zeros(size(Mhat)));
n3=size(Mhat,3); last=floor(n3/2)+1;
for k=1:last
    [U,S,V]=svd(full(Mhat(:,:,k)),'econ'); s=diag(S);
    for j=1:numel(s), s(j)=legacy_gamma_scalar(s(j),gamma,rho); end
    slice=U*diag(s)*V'; Ghat(:,:,k)=slice;
    if k>1, Ghat(:,:,n3-k+2)=conj(slice); end
end
G=real(ifft(Ghat,[],3));
end

function x = legacy_gamma_scalar(y,gamma,rho)
if y==0, x=0; return; end
r=roots([rho,rho*(2*gamma-y),rho*gamma*(gamma-2*y), ...
    gamma*(gamma+1-rho*y*gamma)]);
if all(abs(imag(r))<1e-12)
    r=real(r);
    if max(r)>0, x=max(r); else, x=y; end
else
    x=0;
end
end

function labels = legacy_spectral(W,K)
N=size(W,1); D=spdiags(1./sqrt(sum(W,2)+eps),0,N,N);
L=eye(N)-full(D*W*D); [~,~,V]=svd(L);
embedding=V(:,N-K+1:N);
for i=1:N, embedding(i,:)=embedding(i,:)/norm(embedding(i,:)+eps); end
set_reproducible_seed(1234567890);
labels=kmeans(embedding,K,'MaxIter',1000,'Replicates',20, ...
    'EmptyAction','singleton','Start','plus');
end

function Y = normalize_columns(X)
Y=bsxfun(@rdivide,X,max(sqrt(sum(X.^2,1)),1e-12));
end

function X = soft_threshold(Y,t)
X=sign(Y).*max(abs(Y)-t,0);
end
