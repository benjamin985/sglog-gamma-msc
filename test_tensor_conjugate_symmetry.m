function test_tensor_conjugate_symmetry()
root_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root_dir, 'code')));
rand('state', 20260813); %#ok<RAND>
randn('state', 20260813); %#ok<RAND>
for n3 = [7, 8]
    M = randn(19, 3, n3);
    for kind = {'gamma', 'tnn'}
        rho = 2.3; gamma = 6.43;
        [G, value] = prox_tensor_penalty(M, rho, gamma, kind{1});
        [Gref, valueref] = reference_full_fft(M, rho, gamma, kind{1});
        rel = norm(G(:) - Gref(:)) / max(1, norm(Gref(:)));
        value_error = abs(value - valueref) / max(1, abs(valueref));
        fprintf('n3=%d type=%s relative_error=%.3e value_error=%.3e\n', ...
            n3, kind{1}, rel, value_error);
        assert(rel < 1e-10); assert(value_error < 1e-10);
    end
end
fprintf('TENSOR_CONJUGATE_TEST_PASSED\n');
end

function [G, penalty_value] = reference_full_fft(M, rho, gamma, penalty_type)
Mhat = fft(M, [], 3); Ghat = complex(zeros(size(Mhat))); total = 0;
for k = 1:size(Mhat, 3)
    [U, S, V] = svd(Mhat(:, :, k), 'econ'); singular_values = diag(S);
    if strcmpi(penalty_type, 'gamma')
        shrunk = zeros(size(singular_values));
        for j = 1:numel(singular_values)
            shrunk(j) = prox_gamma_scalar(singular_values(j), gamma, rho);
        end
        total = total + sum((1 + gamma) .* shrunk ./ (gamma + shrunk));
    else
        shrunk = max(singular_values - 1 / rho, 0);
        total = total + sum(shrunk);
    end
    Ghat(:, :, k) = U * diag(shrunk) * V';
end
G = real(ifft(Ghat, [], 3)); penalty_value = total / size(Mhat, 3);
end
