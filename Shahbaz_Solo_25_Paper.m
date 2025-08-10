clear;
close all;
clc;

% File path
%filePath = '494_bus.mtx'; % Ensure this file exists in the current folder
%filePath = '685_bus.mtx'; % Ensure this file exists in the current folder
%filePath = '1138_bus.mtx'; % Ensure this file exists in the current folder
%filePath = 'bcsstk04.mtx'; % Ensure this file exists in the current folder
%filePath = 'bcsstk07.mtx'; % Ensure this file exists in the current folder

%filePath = 'fidap028.mtx'; % Ensure this file exists in the current folder
%filePath = 'pde900.mtx'; % Ensure this file exists in the current folder
filePath = 'arc130.mtx'; % Ensure this file exists in the current folder
filePath = 'ck104.mtx'; % Ensure this file exists in the current folder


% Check if the file exists
if ~isfile(filePath)
    error('File "%s" does not exist. Check the file path.', filePath);
end

% Open the file
fid = fopen(filePath, 'r');
if fid == -1
    error('Failed to open the file "%s". Check permissions and file path.', filePath);
end

% Skip the header lines starting with %
line = fgetl(fid);
while ischar(line) && startsWith(line, '%')
    line = fgetl(fid);
end

% Check if file ended unexpectedly
if ~ischar(line)
    error('File ended unexpectedly. No size information found.');
end

% Read the size of the matrix
sizeData = sscanf(line, '%d %d %d');
rows = sizeData(1);
cols = sizeData(2);
nz = sizeData(3);

% Read the non-zero entries
data = fscanf(fid, '%d %d %f', [3, nz]);
if size(data, 2) ~= nz
    error('Number of non-zero entries does not match specified size.');
end

row_indices = data(1, :);
col_indices = data(2, :);
values = data(3, :);

% Close the file
fclose(fid);

% Construct the sparse matrix
A = sparse(row_indices, col_indices, values, rows, cols);

% Display the matrix
% disp('Matrix A:');
% disp(A);

% Visualize sparsity pattern
spy(A);
title('Sparsity Pattern of the Matrix ck104');
[rA cA]=size(A);
b=rand(rA,1);
x0=zeros(rA,1);


%%%%%%%%%%%%%%%%
tol = 1e-6;       % Tolerance for convergence
K = 10;           % Maximum iterations

[x_1, iter_1, residuals_1] = Algo1(A, b, x0, tol, K);
[x_2, iter_2, residuals_2] = Algo2(A, b, x0, tol, K);
[x_3, iter_3, residuals_3] = Algo3(A, b, x0, tol, K);
[x_4, iter_4, residuals_4] = Algo4(A, b, x0, tol, K);

m = 5;
alpha_bar = 1e-2; % Regularization parameter

% Define spectral filter (example: Tikhonov-type)
phi_filter = @(sigma, alpha) sigma ./ (sigma.^2 + alpha^2);

[x_5, converged, residuals_5] = AMVBiCG_RR(A, b, x0, K, tol, m);
M = diag(diag(A));          % Diagonal part of A
M_inv = @(x) M \ x;
[x_6, ~, residuals_6] = AMVBiCG_RR_RG(A, b, x0, K, tol, m, alpha_bar, true, M_inv);


% Plot residuals for comparison
figure;
semilogy(0:length(residuals_1)-1, residuals_1, '-o', 'LineWidth', 2, 'DisplayName', 'Algorithm 1');
hold on;
semilogy(0:length(residuals_2)-1, residuals_2, '-o', 'LineWidth', 2, 'DisplayName', 'Algorithm 2');
hold on;
semilogy(0:length(residuals_3)-1, residuals_3, '-o', 'LineWidth', 2, 'DisplayName', 'Algorithm 3');
hold on;
semilogy(0:length(residuals_4)-1, residuals_4, '-o', 'LineWidth', 2, 'DisplayName', 'Algorithm 4');
hold on;
semilogy(0:length(residuals_5)-1, residuals_5, '-s', 'LineWidth', 2, 'DisplayName', 'Algorithm 5');
hold on;
semilogy(0:length(residuals_6)-1, residuals_6, '-s', 'LineWidth', 2, 'DisplayName', 'Algorithm 6');


legend show;
grid on;
xlabel('Iteration');
ylabel('Residual Norm');
title('Comparison of Residuals');

%%%%%%%%%%%%%%%%%%%%Algorithms%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x, iter, residuals] = Algo1(A, b, x0, tol, K)
    % Input:
    % A      - Coefficient matrix
    % b      - Right-hand side vector
    % x0     - Initial guess
    % tol    - Convergence tolerance
    % K      - Maximum allowable iterations
    %
    % Output:
    % x      - Approximate solution
    % iter   - Number of iterations performed
    % residuals - Norm of residuals at each iteration

    % Initialization
    x = x0;
    r = A * x - b;
    hat_r = r;
    hat_R = hat_r;
    P = r;
    R = r;
    hat_P = hat_R;
    s = 1;
    S = s;
    iter = 0;
    residuals = [];

    % Main iteration loop
    while S <= K
        % Compute W and hat_W
        W = A * P;
        hat_W = A' * hat_P;

        % Compute sigma and gamma
        sigma = hat_P' * W;
        gamma = hat_P' * r;

        % Solve for alpha
        alpha = gamma / sigma;

        % Calculate epsilon from Table 2 (assume it is 1 for simplicity here)
        epsilon = 1; % Replace with actual epsilon calculation if known

        % Update x, r, and hat_r
        x = x - epsilon * P * alpha;
        r = r - epsilon * W * alpha;
        hat_r = hat_r - epsilon * hat_W * alpha;

        % Check convergence
        residual_norm = norm(r, 2);
        residuals = [residuals; residual_norm];
        if residual_norm < tol
            break;
        end

        % Choose s_{i+1} (keep s=1 for simplicity, update if needed)
        s_next = 1; % Modify based on specific s-step logic

        % Update R and hat_R
        R_next = r;
        for k = 2:s_next
            R_next = [R_next, A^(k-1) * r];
        end

        hat_R_next = hat_r;
        for k = 2:s_next
            hat_R_next = [hat_R_next, (A')^(k-1) * hat_r];
        end

        % Compute omega
        omega = hat_W' * R_next;

        % Solve for beta
        beta = omega / sigma;

        % Calculate xi from Table 2 (assume it is 1 for simplicity here)
        xi = 1; % Replace with actual xi calculation if known

        % Update P and hat_P
        P = R_next - xi * P * beta;
        hat_P = hat_R_next - xi * hat_P * beta;

        % Update iteration counters
        S = S + s;
        iter = iter + 1;
    end
end

function [x, iter, residuals_6] = Algo2(A, b, x0, tol, K)
% --- Algorithm 6 ---
% Initialization
x = x0;
r = A * x - b;
hat_r = A' * r;
hat_R = hat_r;
U = r;
P = r;
R = r;
V = A * U;
s = 1;
S = s;
i = 0;
norm_r = norm(r);
residuals_6 = norm_r; % Store residual norm for plotting

while S <= K
    sigma = hat_R' * V;
    gamma = hat_R' * R;
    alpha = gamma / sigma;
    Q = U - V * alpha;
    x = x - (U + Q) * alpha(:, 1);
    r = r - A * (U + Q) * alpha(:, 1);
    norm_r = norm(r);
    residuals_6 = [residuals_6, norm_r]; % Append residual norm
    if norm_r < tol
        break;
    end
    hat_gamma = hat_R' * R;
    hat_r = A' * r;
    s = 1;
    hat_R = [hat_r, A' * hat_r];
    R = [r, A * r];
    omega = R(:, 1)' * R(:, 2:end);
    beta = omega ./ hat_gamma;
    xi = 1; % Placeholder
    U = R + xi * Q * beta;
    P = U + xi * (Q + P) * beta;
    V = A * P;
    S = S + s;
    i = i + 1;
end
iter = i;
end

function [x, iter, residuals_7] = Algo3(A, b, x0, tol, K)
% --- Algorithm 7: Variable s-step BiCGStab ---
% Initialization
x = x0;
r = A * x - b;
hat_r = r;
hat_R = hat_r;
P = r;
R = r;
s = 1;
S = s;
i = 0;
norm_r = norm(r);
residuals_7 = norm_r; % Store residual norm for plotting

while S <= K
    W = A * P;
    hat_W = A * hat_R;
    sigma = hat_W' * W;
    gamma = hat_R' * R;
    alpha = gamma / sigma;
    epsilon = 1; % Placeholder for epsilon_i calculation
    T = R - epsilon * W * alpha;
    delta = (W' * T) / sigma;
    x = x + epsilon * P * alpha(:, 1) + T * delta(:, 1);
    r = r - epsilon * W * alpha(:, 1) - A * T * delta(:, 1);
    norm_r = norm(r);
    residuals_7 = [residuals_7, norm_r]; % Append residual norm
    if norm_r < tol
        break;
    end
    s = 1; % Placeholder for s_(i+1)
    hat_R = [hat_r, A' * hat_r];
    R = [r, A * r];
    omega = R(:, 1)' * R(:, 2:end);
    beta = omega ./ sigma;
    xi = 1; % Placeholder for xi_i calculation
    P = R + xi * (P - W * delta) * beta;
    S = S + s;
    i = i + 1;
end
iter = i;
end


function [x, iter, residuals] = Algo4(A, b, x0, tol, K)
    % Variable s-step hybrid BiCGStab Function
    % Inputs:
    % A: Matrix
    % b: Right-hand side vector
    % x0: Initial guess
    % tol: Tolerance for convergence
    % K: Maximum allowable iterations

    % Initialize variables
    r0 = b - A * x0;
    hat_r0 = r0;
    R0 = r0;
    hat_R0 = hat_r0;
    P0 = r0;
    W0 = A * P0;
    Omega0 = 0;
    s0 = 1;
    S0 = s0;
    i = 0;
    U = P0;
    % Residual history
    residuals = norm(r0);

    % Start iteration
    while S0 <= K
        % Compute scalars
        sigma = W0' * W0; % Scalar

        % Solve for alpha
        alpha = (P0' * P0) / sigma;

        % Calculate epsilon from table 2 row 2 (set to 1 as placeholder)
        epsilon = 1; % Replace with appropriate logic if needed

        % Update T and Y
        T = R0 - epsilon * W0 * alpha;
        Y = T - R0 - epsilon * Omega0 * alpha + epsilon * W0 * alpha;

        if mod(i, 2) == 1
            % Odd iterations: compute intermediate scalars
            tilde_Y = Y' * Y;
            tilde_W = W0' * Y;
            tilde_T = Y' * T;

            % Solve for delta and eta
            delta = (tilde_Y * tilde_W - tilde_T * tilde_W) / (sigma * tilde_Y - tilde_Y * tilde_W);
            eta = (sigma * tilde_T - tilde_Y * tilde_W) / (sigma * tilde_Y - tilde_Y * tilde_W);
        else
            % Even iterations: simplified update
            delta = (W0' * T) / sigma;
            eta = 0;
        end

        % Update U, Z, x, and r
        U = W0 * delta + (T - R0 + U * delta) * eta;
        Z = R0 * delta - epsilon * U * alpha;
        x = x0 + epsilon * P0 * alpha + Z;
        r = R0 - epsilon * W0 * alpha - A * Z;

        % Store residual norm
        residuals(end + 1) = norm(r, 2);

        % Check convergence
        if residuals(end) < tol
            fprintf('Converged at iteration %d\n', i);
            break;
        end

        % Construct R_(i+1)
        R = r; % Simplified for s_next = 1

        % Solve for beta
        beta = (R0' * R) / sigma; % Scalar division

        % Calculate xi (set to 1 as placeholder)
        xi = 1; % Replace with appropriate calculation if needed

        % Update P, W, T, and Omega
        P = R + xi * (P0 - U) * beta;
        W0 = A * P;
        T = R - W0;
        Omega0 = A * T + xi * W0 * norm(beta, 2);

        % Update variables for next iteration
        x0 = x;
        R0 = R;
        S0 = S0 + s0;
        i = i + 1;
    end

    % Return results
    iter = i;
end

function [x, converged, residuals] = AMVBiCG_RR(A, b, x0, K_max, tol, m, M_inv)
    % Algorithm 5 (AMVBiCG_RR)
    % Inputs:
    %   A       - Matrix or function handle
    %   b       - Right-hand side vector
    %   x0      - Initial guess
    %   K_max   - Maximum iterations
    %   tol     - Tolerance
    %   m       - Residual refinement period
    %   M_inv   - Preconditioner (optional)
    %
    % Outputs:
    %   x         - Approximate solution
    %   converged - Boolean (true if converged)
    %   residuals - Residual norms at each iteration

    % Initialize
    x = x0;
    r = b - A * x;
    r_hat = r;  % Shadow residual (for BiCG)
    p = r;
    p_hat = r_hat;

    residuals = [norm(r)];
    converged = false;

    % Adaptive subspace parameters (from Algorithm 5)
    s = 1;      % Initial subspace size
    xi = 1e-8;  % Orthogonalization threshold

    for j = 0:K_max-1
        % Standard BiCG steps (Algorithm 1)
        Ap = A * p;
        alpha = (r_hat' * r) / (r_hat' * Ap);
        x = x + alpha * p;
        r_new = r - alpha * Ap;

        % Check convergence
        residuals = [residuals; norm(r_new)];
        if residuals(end) < tol
            converged = true;
            break;
        end

        % BiCG residual update
        A_p_hat = A' * p_hat;
        beta = (r_hat' * r_new) / (r_hat' * r);
        p = r_new + beta * p;
        p_hat = r_hat + beta * p_hat;
        r = r_new;

        % --- Algorithm 5 Enhancements ---
        % Adaptive subspace sizing
        if residuals(end) < 0.5 * residuals(end-1)
            s = min(s + 1, 5);  % Gradually increase subspace size (max 5)
        else
            s = max(s - 1, 1);  % Decrease if progress stalls
        end

        % Periodic residual refinement (every m steps)
        if mod(j+1, m) == 0
            r = b - A * x;  % Explicit recalculation
            if exist('M_inv', 'var') && ~isempty(M_inv)
                x = x + M_inv(r);  % Preconditioned refinement
            end
            residuals(end) = norm(r);
        end

        % Stabilized orthogonalization (if s > 1)
        if s > 1
            % Build Krylov block
            K = r;
            for k = 1:s-1
                K = [K, A * K(:, end)];
            end

            % Modified Gram-Schmidt with thresholding
            [Q, ~] = mgs_orth([], K, xi);

            % Update search direction
            p = Q(:, end);  % Use last vector in the block
            p_hat = Q(:, end);  % Same for shadow direction
        end
    end
end

function [x, converged, residuals] = AMVBiCG_RR_RG(A, b, x0, K_max, tol, m, alpha_bar, use_spectral_filter, M_inv)
    % Algorithm 6 (AMVBiCG-RR with regularization)
    %
    % Inputs:
    %   A                   - Matrix or function handle
    %   b                   - Right-hand side vector
    %   x0                  - Initial guess
    %   K_max               - Maximum iterations
    %   tol                 - Tolerance
    %   m                   - Residual refinement period
    %   alpha_bar           - Regularization parameter (Tikhonov)
    %   use_spectral_filter - Boolean (true to enable spectral filtering)
    %   M_inv               - Preconditioner (optional)
    %
    % Outputs:
    %   x         - Approximate solution
    %   converged - Boolean (true if converged)
    %   residuals - Residual norms at each iteration

    % Initialize
    x = x0;
    r = b - A * x;
    r_hat = r;  % Shadow residual
    p = r;
    p_hat = r_hat;

    residuals = [norm(r)];
    converged = false;

    for j = 0:K_max-1
        % --- Standard BiCG steps (Algorithm 1) ---
        Ap = A * p;
        A_p_hat = A' * p_hat;

        % Regularized solve for alpha (Algorithm 6)
        sigma = p_hat' * Ap;
        if cond(sigma) > 1e8
            alpha = (sigma + alpha_bar) \ (p_hat' * r);  % Tikhonov regularization
        else
            alpha = sigma \ (p_hat' * r);
        end

        % Update solution and residuals
        x = x + alpha * p;
        r_new = r - alpha * Ap;
        r_hat_new = r_hat - alpha * A_p_hat;

        % --- Spectral filtering (optional, from Algorithm 6) ---
        if use_spectral_filter
            T = [r_new, A * r_new]' * [r_hat_new, A' * r_hat_new];  % Small projected matrix
            [U, S, V] = svd(T);
            S_tilde = diag(S) ./ (diag(S).^2 + alpha_bar^2);  % Tikhonov filter
            T_tilde = U * diag(S_tilde) * V';
            r_new = r_new * T_tilde(1,1);  % Damp unstable modes
        end

        % Check convergence
        residuals = [residuals; norm(r_new)];
        if residuals(end) < tol
            converged = true;
            break;
        end

        % BiCG residual update
        beta = (r_hat_new' * r_new) / (r_hat' * r);
        p = r_new + beta * p;
        p_hat = r_hat_new + beta * p_hat;
        r = r_new;
        r_hat = r_hat_new;

        % --- Residual refinement (every m steps, from Algorithm 6) ---
        if mod(j+1, m) == 0
            r = b - A * x;  % Explicit recalculation
            if exist('M_inv', 'var') && ~isempty(M_inv)
                x = x + M_inv(r);  % Preconditioned refinement
            end
            residuals(end) = norm(r);
        end
    end
end
function [Q, R] = mgs_orth(Q_prev, X, threshold)
    % Modified Gram-Schmidt with thresholding
    [n, k] = size(X);
    Q = zeros(n, k);
    
    for i = 1:k
        v = X(:,i);
        for j = 1:size(Q_prev,2)
            dot_prod = Q_prev(:,j)'*v;
            if abs(dot_prod) > threshold
                v = v - dot_prod*Q_prev(:,j);
            end
        end
        norm_v = norm(v);
        if norm_v > threshold
            Q(:,i) = v/norm_v;
        else
            Q(:,i) = zeros(n,1);
        end
    end
    R = Q'*X;
end




