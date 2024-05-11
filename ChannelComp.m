% ChannelComp Main Script

% Parameters
K = 4; % Number of nodes
modulation_type = 'BPSK'; % Choose modulation: 'BPSK', 'QPSK', '16QAM', etc.
SNR = 30; % Signal-to-noise ratio in dB
h = generate_channel_coefficients(K); % Generate channel coefficients for each node
q = 2;
p_init = ones(1, K); % Initial power allocation
f = @(x) sum(x); % Example function (sum)
data = randi([0 q-1], K, 1); % Generate random data for each node
% Get modulation symbols
symbols = get_modulation_symbols(modulation_type,q);

modulated_data = symbols(data + 1);
% Solve G-ChannelComp (P3/P4)
[x_opt, s_opt, A, gamma_i_j] = solve_channelcomp_p3_p4(K, f, modulated_data, q);

% Solve E-ChannelComp (P6)
[p_opt, s_opt_e] = solve_channelcomp_p6(x_opt, s_opt, K, modulated_data, h, q, f, A, gamma_i_j);

% Transmission Over MAC
[transmitted_signal, received_signal, s_opt_e] = transmit_over_mac(x_opt, s_opt_e, K, modulated_data, h, p_opt, SNR,q);

% Decode signals using Voronoi diagram and MLE
decoded_values = decode_channelcomp_voronoi(received_signal, s_opt_e, K, modulated_data, f,q);

% Display decoded function values
disp('Decoded function values:');
disp(decoded_values);

% Plot constellation points and decision boundaries
plot_constellation_and_boundaries(received_signal, s_opt_e, decoded_values, K, symbols, modulation_type);

%===========================================function=================================================================%
function h = generate_channel_coefficients(K)
    % Generate random channel coefficients (between 0.5 and 1.5) for each node
    % h = 0.5 + rand(1, K);
    h = randn(K,1) + 1i*randn(K,1);% Random complex channel coeffs

end
function symbols = get_modulation_symbols(modulation_type, q)
    % Get modulation symbols based on the specified type
    switch modulation_type
        case 'BPSK'
            symbols = pskmod(0:q-1, q);
        case 'QPSK'
            symbols = pskmod(0:q-1, q); % 4-PSK (QPSK)
        case 'QAM'
            if mod(log2(q), 1) == 0 % Ensure q is a power of 2
                symbols = qammod(0:q-1, q);
            else
                error('QAM requires q to be a power of 2');
            end
        case 'FSK'
            symbols = exp(1j * (0:q-1) * 2 * pi / q); % FSK with q tones
        otherwise
            error('Unknown modulation type');
    end
    % Normalize symbols for unit average power
    symbols = symbols / sqrt(mean(abs(symbols).^2));
end

function [x_opt, s_opt, A, gamma_i_j] = solve_channelcomp_p3_p4(K, f, modulated_data, q)
    % Solve the P3/P4 optimization problem for ChannelComp

    % Generate matrix A
    M = q^K; % Total number of received signal combinations
    N = K * q; % Total number of modulation states
    num_bits = ceil(log2(q)); % Necessary bit width for each node's states
    index_combinations = de2bi(0:M-1, num_bits * K, 'left-msb');

    A = zeros(M, N);
    for i = 1:M
        for k = 1:K
            base_idx = (k-1) * q;
            mapped_idx = sum(index_combinations(i, (k-1)*num_bits+1:k*num_bits) .* (2.^((num_bits-1):-1:0))) + 1;
            A(i, base_idx + mapped_idx) = 1;
        end
    end

    % Define the modulation vector x
    % x = repmat(symbols, K, 1);

    % Calculate the vector s
    % s = A * x;

    % Dynamic epsilon calculation based on the function differences
    epsilon = calculate_epsilon(A, f, index_combinations);

    % Construct Bi_j and γi_j matrices
    [Bi_j, gamma_i_j] = construct_bi_j_gamma(A, epsilon, f, index_combinations);

    % Solve Problem P3 using CVX (Convex Optimization Library)
    x_opt = solve_optimization_problem(N, Bi_j, gamma_i_j);

    % Calculate the resulting vector s after applying x_opt
    s_opt = A * x_opt;
    disp('Optimized vector s (G-ChannelComp):');
    disp(s_opt);
end

function epsilon = calculate_epsilon(A, f, index_combinations)
    % Calculate epsilon dynamically based on the function differences
    M = size(A, 1);
    M_pairs = nchoosek(1:M, 2);
    num_pairs = size(M_pairs, 1);
    max_f_diff_sq = 0;

    for idx = 1:num_pairs
        i = M_pairs(idx, 1);
        j = M_pairs(idx, 2);
        f_diff_sq = abs(f(index_combinations(i, :) + 1) - f(index_combinations(j, :) + 1))^2;
        max_f_diff_sq = max(max_f_diff_sq, f_diff_sq);
    end

    epsilon = 1 / max_f_diff_sq;
end

function [Bi_j, gamma_i_j] = construct_bi_j_gamma(A, epsilon, f, index_combinations)
    % Construct Bi,j and γi,j matrices
    M = size(A, 1);
    N = size(A, 2);
    M_pairs = nchoosek(1:M, 2);
    num_pairs = size(M_pairs, 1);

    Bi_j = zeros(N, N, num_pairs);
    gamma_i_j = zeros(num_pairs, 1);

    for idx = 1:num_pairs
        i = M_pairs(idx, 1);
        j = M_pairs(idx, 2);
        a_i = A(i, :);
        a_j = A(j, :);
        Bi_j(:, :, idx) = (a_i - a_j)' * (a_i - a_j);
        gamma_i_j(idx) = epsilon * abs(f(index_combinations(i, :) + 1) - f(index_combinations(j, :) + 1))^2;
    end
end

function x_opt = solve_optimization_problem(N, Bi_j, gamma_i_j)
    % Solve the P3 optimization problem using CVX
    cvx_begin sdp
        variable X(N, N) semidefinite
        minimize(trace(X))
        subject to
            for idx = 1:size(Bi_j, 3)
                trace(X * Bi_j(:, :, idx)) >= gamma_i_j(idx);
            end
            trace(X) <= 1;
            X >= 0;
    cvx_end

    % Check if X is rank-one
    [U, S, V] = svd(X);
    rank_X = sum(diag(S) > 1e-6);

    if rank_X == 1
        % Optimal solution found with P3
        x_opt = U(:, 1) * sqrt(S(1, 1));
        disp('Optimal vector x from P3:');
        disp(x_opt);
    else
        % Solution X is not rank-one, solve P4 using DC Programming
        disp('Solution X is not rank-one, moving to solve P4...');
        
        % Initialize X for P4
        X = eye(N);

        % DC Programming Parameters
        tolerance = 1e-4;
        max_iterations = 50;
        iteration = 0;
        converged = false;

        while ~converged && iteration < max_iterations
            % Compute the leading eigenvector of X
            [U, S, V] = svd(X);
            u1 = U(:, 1);

            % Solve the convex approximation problem using CVX
            cvx_begin sdp
                variable X_new(N, N) semidefinite
                minimize(trace(X_new) - u1' * X_new * u1)
                subject to
                    for idx = 1:size(Bi_j, 3)
                        trace(X_new * Bi_j(:, :, idx)) >= gamma_i_j(idx);
                    end
                    trace(X_new) <= 1;
                    X_new >= 0;
            cvx_end

            % Check for convergence
            if norm(X_new - X, 'fro') < tolerance
                converged = true;
            end

            % Update X
            X = X_new;
            iteration = iteration + 1;
        end

        % Check if X is rank-one after P4
        [U, S, V] = svd(X);
        rank_X = sum(diag(S) > 1e-6);

        if rank_X == 1
            % Optimal solution found after solving P4
            x_opt = U(:, 1) * sqrt(S(1, 1));
            disp('Optimal vector x from P4:');
            disp(x_opt);
        else
            % Suboptimal solution, apply Gaussian randomization
            disp('Solution X is still not rank-one, applying Gaussian randomization...');
            num_randomizations = 10;
            best_obj = inf;
            best_x = zeros(N, 1);

            for iter = 1:num_randomizations
                z = randn(N, 1);
                x_trial = U * sqrt(S) * z;
                obj_trial = trace(x_trial' * x_trial);

                if obj_trial < best_obj
                    best_obj = obj_trial;
                    best_x = x_trial;
                end
            end

            x_opt = best_x;
            disp('Randomized optimal vector x:');
            disp(x_opt);
        end
    end
end

function [p_opt, s_opt_e] = solve_channelcomp_p6(x_opt, s_opt, K, modulated_data, h, q, f, A, gamma_i_j)
    % Calculate the operator Hq
    Hq = kron(diag(h), eye(q));

    % Create matrix C
    C = A * Hq * diag(x_opt) * kron(eye(K), ones(q, 1));

    % Generate index combinations
    M = q^K;
    num_bits = ceil(log2(q));
    index_combinations = de2bi(0:M-1, num_bits * K, 'left-msb');

    % Construct Ci_j matrices
    Ci_j = construct_ci_j_gamma(C, index_combinations);

    % Solve Problem P6 using CVX
    p_opt = solve_p6_optimization(K, Ci_j, gamma_i_j);

    % Calculate the resulting vector s after applying p_opt
    s_opt_e = C * p_opt;
    disp('Optimized vector s (E-ChannelComp):');
    disp(s_opt_e);
end

function Ci_j = construct_ci_j_gamma(C, index_combinations)
    % Construct C_{i,j} matrices
    M = size(C, 1);
    M_pairs = nchoosek(1:M, 2);
    num_pairs = size(M_pairs, 1);

    Ci_j = zeros(size(C, 2), size(C, 2), num_pairs);

    for idx = 1:num_pairs
        i = M_pairs(idx, 1);
        j = M_pairs(idx, 2);
        diff_vector = (C(i, :) - C(j, :))';
        Ci_j(:, :, idx) = diff_vector * diff_vector';
    end
end


function p_opt = solve_p6_optimization(K, Ci_j, gamma_i_j)
    % Solve the P6 optimization problem using CVX
    cvx_begin sdp
        variable P(K, K) semidefinite
        minimize(trace(P))
        subject to
            for idx = 1:size(Ci_j, 3)
                trace(P * Ci_j(:, :, idx)) >= gamma_i_j(idx);
            end
            P >= 0;
    cvx_end

    % Check if P is rank-one
    [U, S, ~] = svd(P);
    rank_P = sum(diag(S) > 1e-6);

    if rank_P == 1
        % Optimal solution found
        p_opt = U(:, 1) * sqrt(S(1, 1));
        disp('Optimal vector p from P6:');
        disp(p_opt);
    else
        % Suboptimal solution, apply Gaussian randomization
        disp('Solution P is not rank-one, applying Gaussian randomization...');
        num_randomizations = 10;
        best_obj = inf;
        best_p = zeros(K, 1);

        for iter = 1:num_randomizations
            z = randn(K, 1);
            p_trial = U * sqrt(S) * z;
            obj_trial = trace(p_trial' * p_trial);

            if obj_trial < best_obj
                best_obj = obj_trial;
                best_p = p_trial;
            end
        end

        p_opt = best_p;
        disp('Randomized optimal vector p:');
        disp(p_opt);
    end
end




function [transmitted_signal, received_signal, s_opt] = transmit_over_mac(x_opt, s_opt, K, modulated_data, h, p, SNR,q)
    % Transmit the signals over MAC with noise and channel effects

    
    N = K*q;
    transmitted_signal = zeros(N, 1);
    for k = 1:K
        transmitted_signal((k-1)*q + 1:k*q) = h(k) * sqrt(p(k)) * x_opt((k-1)*q + 1:k*q);
    end

    % Add noise to the signal (AWGN)
    noise_variance = 10^(-SNR / 10);
    z = sqrt(noise_variance) * (randn(length(s_opt), 1) + 1i * randn(length(s_opt), 1)) / sqrt(2);

    % Simulate received signal
    received_signal = s_opt + z;
end

function decoded_values = decode_channelcomp_voronoi(received_signal, s_opt, K, modulated_data, f,q)
    % Decode the received signals using the Voronoi diagram and MLE

    
    M = q^K;
    num_bits = ceil(log2(q)); % Necessary bit width for each node's states
    index_combinations = de2bi(0:M-1, num_bits * K, 'left-msb');
    expected_values = arrayfun(@(i) f(index_combinations(i, :)), 1:M);

    % Maximum Likelihood Estimator (MLE)
    Pr = @(y, g, sigma) 1 / sqrt(2 * pi * sigma^2) * exp(-norm(y - g, 2)^2 / (2 * sigma^2));

    % Decision Rule with Voronoi Diagram
    decoded_values = zeros(size(received_signal));
    likelihoods = zeros(length(received_signal), M);
    for i = 1:length(received_signal)
        for j = 1:M
            likelihoods(i, j) = Pr(received_signal(i), s_opt(j), 1);
        end
        [~, idx] = max(likelihoods(i, :));
        decoded_values(i) = expected_values(idx);
    end
end

function plot_constellation_and_boundaries(received_signal, s_opt, decoded_values, K, modulated_data, modulation_type)
    % Plot the constellation points and decision boundaries
    
    % Generate matrix A
    q = length(modulated_data);
    M = q^K; % Total number of received signal combinations
    num_bits = ceil(log2(q)); % Necessary bit width for each node's states
    index_combinations = de2bi(0:M-1, num_bits * K, 'left-msb');

    % Function Value Mapping
    expected_values = arrayfun(@(i) sum(index_combinations(i, :)), 1:M);

    % Plot constellation points and decision boundaries
    figure;
    scatter(real(received_signal), imag(received_signal), 'o');
    hold on;
    scatter(real(s_opt), imag(s_opt), 'x', 'LineWidth', 2);
    for i = 1:M
        text(real(s_opt(i)), imag(s_opt(i)), ['  ', num2str(expected_values(i))], 'FontSize', 10, 'FontWeight', 'bold');
    end
    for i = 1:length(received_signal)
        text(real(received_signal(i)), imag(received_signal(i)), ['  ', num2str(decoded_values(i))], 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'blue');
    end
    xlabel('Real Part');
    ylabel('Imaginary Part');
    title(['Constellation Points and Decision Boundaries for ', modulation_type]);
    legend({'Received', 'Constellation'});
    grid on;
end
