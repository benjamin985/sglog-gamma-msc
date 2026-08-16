nf = @(X) sqrt(sum(sum(X.^2)));
A = randn(200); A = A'*A; A = (A+A')/2;
for trial = 1:3
    [U, D] = eig(A);
    printf('trial %d: eig orthU=%.4g rec=%.4g\n', trial, nf(U'*U-eye(200)), nf(A-U*diag(diag(D))*U')/nf(A));
end
% diagonal matrix test (known eigendecomposition)
B = diag(1:200);
[U2, D2] = eig(B);
printf('diag(1:200): eig orthU=%.4g rec=%.4g\n', nf(U2'*U2-eye(200)), nf(B-U2*diag(diag(D2))*U2')/nf(B));
