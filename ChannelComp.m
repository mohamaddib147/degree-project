% ChannelComp Main Script

% Parameters
K = 2; % Number of nodes
modulation_type = 'QPSK'; % Choose modulation: 'BPSK', 'QPSK', '16QAM', etc.
q = 8; % Modulation states for QAM (8-QAM)
SNR = 30; % Signal-to-noise ratio in dB
h = [1, 0.8, 0.6, 0.4]; % Channel coefficients for each node
p = [1, 1, 1, 1]; % Transmit power for each node

% Define the desired function f and its range Rf
f = @(x) sum(x); % Example function (sum)

% Get the modulation symbols
symbols = get_modulation_symbols(modulation_type, q);

% Get the optimization results (P3/P4)
[x_opt, s_opt] = solve_channelcomp_p3_p4(K, f, symbols, q);

% Transmission Over MAC
% Generate the transmitted and received signals
[transmitted_signal, received_signal, s_opt] = transmit_over_mac(x_opt, s_opt, K, symbols, h, p, SNR);

% Decode the received signals using the Voronoi diagram and MLE
decoded_values = decode_channelcomp_voronoi(received_signal, s_opt, K, symbols, f);

% Display decoded function values
disp('Decoded function values:');
disp(decoded_values);

% Plot the constellation points and decision boundaries
plot_constellation_and_boundaries(received_signal, s_opt, decoded_values, K, symbols, modulation_type);

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

function [x_opt, s_opt] = solve_channelcomp_p3_p4(K, f, symbols, q)
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
    x = repmat(symbols, K, 1);

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
    disp('Optimized vector s:');
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
    % Construct Bi_j and γi_j matrices
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

function [transmitted_signal, received_signal, s_opt] = transmit_over_mac(x_opt, s_opt, K, symbols, h, p, SNR)
    % Transmit the signals over MAC with noise and channel effects

    q = length(symbols);
    N = length(x_opt);
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

function decoded_values = decode_channelcomp_voronoi(received_signal, s_opt, K, symbols, f)
    % Decode the received signals using the Voronoi diagram and MLE

    q = length(symbols);
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

function plot_constellation_and_boundaries(received_signal, s_opt, decoded_values, K, symbols, modulation_type)
    % Plot the constellation points and decision boundaries
    
    % Generate matrix A
    q = length(symbols);
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
