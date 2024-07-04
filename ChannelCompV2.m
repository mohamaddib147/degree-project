% Main script
K = 2; % Number of nodes
q = 8; % Number of quantization levels
N = K * q; % Dimension of X

% Generate random data for each node
x = generateRandomData(K, q);
disp('Generated Data:');
disp(x);

% Quantize the generated data
quantizedX = quantizeData(x, q);
disp('Quantized Data:');
disp(quantizedX);

% Generate matrix A
A = generateA(K, q);
disp('Matrix A:');
disp(A);

% Define the desired function f
f = @(x) max(x); 
f_name = 'f(x) = max(x)';
disp(['Desired Function: ', f_name]);

% Compute function values f for each row of A
f_values = computeFunctionValues(A, f);
disp('Function Values:');
disp(f_values);

% Calculate epsilon dynamically
% epsilon = calculateEpsilon(f_values);
epsilon = 1e-4;
disp(['Epsilon: ', num2str(epsilon)]);

% Compute B and gamma
[B, gamma] = computeBgamma(A, epsilon, f_values);

% Solve P3
num_pairs = length(B); % Number of pairs
[X_opt, fval] = solveP3(B, gamma, N, num_pairs);
disp('Optimal Solution X_opt:');
disp(X_opt);

% Extract modulation vector from the SDP solution
[V, D] = eig(X_opt);
rank_X_opt = rank(X_opt, 1e-6); % Use a small tolerance to check the rank
disp(['Rank of X_opt: ', num2str(rank_X_opt)]);

if rank_X_opt == 1
    disp('Optimal Modulation Vector using Cholesky Decomposition:');
    modulation_vector = chol(X_opt, 'lower') * V(:, 1);
else
    disp('Sub-optimal Modulation Vector using Gaussian Randomization:');
    modulation_vector = gaussianRandomization(X_opt, N);
end
disp('Modulation Vector:');
disp(modulation_vector);

% Encode and transmit signals
encoded_signals = cell(1, K);
for k = 1:K
    encoded_signals{k} = encoder(quantizedX(k), modulation_vector((k-1)*q+1:k*q), q);
end
disp('Encoded Signals:');
disp(encoded_signals);

% Combine transmitted signals (simulating transmission over a MAC)
combined_signal = sum(cat(2, encoded_signals{:}), 2);
disp('Combined Signal:');
disp(combined_signal);

% Receiver decoding
decoded_value = receiverDecoding(combined_signal, modulation_vector, f_values, q, K);
disp('Decoded Value:');
disp(decoded_value);

% Plot the constellation diagram
plotConstellationDiagram(modulation_vector, K, q, f_name);

% Functions

function x = generateRandomData(K, q)
    % Generate random data for each node
    x = randi([0 q-1], 1, K);
end

function quantizedData = quantizeData(x, q)
    % Ensure data is within the quantization levels (0 to q-1)
    quantizedData = min(max(round(x), 0), q-1);
end

function A = generateA(K, q)
    % Calculate M and N
    M = q^K; % Number of possible combinations of quantized values
    N = K * q; % Number of nodes times the number of quantization levels
    
    % Generate all possible index combinations
    index_combinations = generateCombinations(K, q); % M x K matrix
    
    % Initialize matrix A
    A = zeros(M, N);
    
    % Fill in matrix A
    for i = 1:M
        for k = 1:K
            % Get the state of the k-th node in the i-th combination
            state = index_combinations(i, k);
            % Convert state to one-hot encoding
            one_hot = zeros(1, q);
            one_hot(state + 1) = 1; % +1 because MATLAB indices start at 1
            % Place the one-hot vector in the corresponding location in A
            A(i, (k-1)*q + 1:k*q) = one_hot;
        end
    end
end

function combinations = generateCombinations(K, q)
    % Generate all possible combinations of states
    M = q^K;
    combinations = zeros(M, K);
    
    for i = 1:M
        val = i - 1;
        for k = 1:K
            combinations(i, k) = mod(val, q);
            val = floor(val / q);
        end
    end
end

function f_values = computeFunctionValues(A, f)
    % Compute function values for each row of A
    M = size(A, 1);
    f_values = zeros(M, 1);
    for i = 1:M
        % Convert one-hot encoded row back to original values
        values = oneHotToValues(A(i, :), size(A, 2) / 2);
        f_values(i) = f(values);
    end
end

function values = oneHotToValues(row, q)
    % Convert one-hot encoded row to original values
    K = length(row) / q;
    values = zeros(1, K);
    for k = 1:K
        values(k) = find(row((k-1)*q + 1:k*q)) - 1;
    end
end

function [B, gamma] = computeBgamma(A, epsilon, f_values)
    % Compute B and gamma matrices for the SDP
    M = size(A, 1);
    num_pairs = M * (M - 1) / 2;
    B = cell(num_pairs, 1);
    gamma = zeros(num_pairs, 1);
    pair_index = 1;
    for i = 1:M
        for j = i+1:M
            a_diff = (A(i, :) - A(j, :))';
            B{pair_index} = a_diff * a_diff';
            gamma(pair_index) = epsilon * (f_values(i) - f_values(j))^2;
            pair_index = pair_index + 1;
        end
    end
end

function [X_opt, fval] = solveP3(B, gamma, N, num_pairs)
    % Solve the semidefinite programming problem P3
    cvx_begin sdp
        variable X(N, N) hermitian semidefinite 
        minimize( trace(X) )
        subject to
            for pair_index = 1:num_pairs
                if ~isempty(B{pair_index})
                    trace(X * B{pair_index}) >= gamma(pair_index);
                end
            end
            trace(X) <= 1;
            X >= 0;
    cvx_end
    X_opt = X;
    fval = cvx_optval;
end

function x_opt = gaussianRandomization(X_opt, N)
    % Gaussian randomization method to find a suboptimal solution
    num_trials = 100000;
    best_obj = inf;
    x_opt = zeros(N, 1);
    
    for trial = 1:num_trials
        x_rand = (mvnrnd(zeros(N, 1), X_opt) + 1i * mvnrnd(zeros(N, 1), X_opt))';
        x_rand = x_rand / norm(x_rand); % Normalize to unit norm
        
        obj_val = x_rand' * X_opt * x_rand;
        if obj_val < best_obj
            best_obj = obj_val;
            x_opt = x_rand;
        end
    end
end

function encoded_signal = encoder(x_quantized, modulation_vector, q)
    % Encoder function to map quantized values to modulation symbols
    encoded_signal = modulation_vector(x_quantized + 1); % +1 for MATLAB 1-based indexing
end

function decoded_value = receiverDecoding(received_signal, modulation_vector, f_values, q, K)
    % Generate the constellation points from the modulation vector
    constellation_points = generateConstellationPoints(modulation_vector, q, K);
    
    % Find the closest constellation point
    [~, idx] = min(abs(received_signal - constellation_points));
    
    % Map the index to the corresponding function value
    decoded_value = f_values(idx;
end

function constellation_points = generateConstellationPoints(modulation_vector, q, K)
    % Generate all possible combinations of the modulation symbols
    M = q^K;
    index_combinations = generateCombinations(K, q);
    
    % Initialize constellation points
    constellation_points = zeros(M, 1);
    
    % Generate constellation points by summing the modulation vectors
    for i = 1:M
        for k = 1:K
            state = index_combinations(i, k);
            constellation_points(i) = constellation_points(i) + modulation_vector((k-1)*q + state + 1);
        end
    end
end

function plotConstellationDiagram(modulation_vector, K, q, f_name)
    % Plot the constellation diagram of the modulation vector
    % modulation_vector: Modulation vector
    % K: Number of nodes
    % q: Number of quantization levels
    % f_name: Name of the function for the plot title

    % Prepare the constellation points
    constellation_points = reshape(modulation_vector, q, K);

    % Generate different markers for each node's constellation points
    markers = {'o', '^', 's', 'd', 'v', '>', '<', 'p', 'h', '+', '*', 'x'};
    colors = lines(K); % Generate different colors for each node

    figure;
    hold on;
    grid on;

    for k = 1:K
        real_parts = real(constellation_points(:, k));
        imag_parts = imag(constellation_points(:, k));
        
        % Set small imaginary parts to zero
        imag_parts(abs(imag_parts) < 1e-6) = 0;
        
        % Plot the constellation points
        scatter(real_parts, imag_parts, 'Marker', markers{k}, ...
                'MarkerEdgeColor', 'k', 'MarkerFaceColor', colors(k, :), ...
                'DisplayName', ['x_' num2str(k)]);
        
        % Connect points with lines
        plot(real_parts, imag_parts, 'Color', colors(k, :), 'LineWidth', 1.5);
    end

    % Set axis labels
    xlabel('Real(x)');
    ylabel('Imag(x)');

    % Set axis limits for better visualization
    axis([-0.5 0.5 -0.5 0.5]);

    % Plot title and legend
    title(['Constellation Diagram for ', f_name, ' with K=', num2str(K), ' nodes and q=', num2str(q)]);
    legend('show');

    % Set plot properties for better visualization
    set(gca, 'FontSize', 12);
    set(gca, 'LineWidth', 1.5);

    hold off;
end

function epsilon = calculateEpsilon(f_values)
    % Calculate epsilon  based on Lemma 1
    max_diff = max(max(abs(f_values - f_values.')));
    epsilon = 1 / max_diff;
end
