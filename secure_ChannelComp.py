import numpy as np
import cvxpy as cp
from scipy.stats import rayleigh
from itertools import product, combinations
from numpy.linalg import svd, norm

def generate_channel_coefficients(num_users, sigma_h, sigma_g):
    h = rayleigh.rvs(loc=0, scale=sigma_h, size=num_users)
    h[0] = np.sqrt(np.pi - np.sqrt(4 - np.pi)) / np.sqrt(2)
    g = rayleigh.rvs(loc=0, scale=sigma_g, size=num_users)
    h[1:] = np.maximum(h[1:], h[0])
    return h, g

def quantize(x, num_bits):
    return np.round(x * (2**num_bits - 1)) / (2**num_bits - 1)

def modulate(tilde_x, modulation_vector):
    return modulation_vector[int(tilde_x)]

def evaluate_function(values, function_name):
    if function_name == 'frac1':
        pairwise_product = values[0]*values[1] + values[1]*values[2] + values[2]*values[0]
        square_values = values[2]**2 + values[0]**2 + values[1]**2
        return square_values + 3*pairwise_product
    elif function_name == 'frac2':
        ratio1 = (values[0]+1) / (values[1]+1)
        ratio2 = (values[2]+3) / (values[3]+1)
        return ratio1 + ratio2
    elif function_name == 'frac3':
        return (values[0]+1) / (values[1]+1)
    elif function_name == 'prod':
        return np.prod(values)
    elif function_name == 'sum':
        return np.sum(values)
    elif function_name == 'max':
        return np.max(values)
    elif function_name == 'norm':
        return norm(values)

def encoder(x_quantized, modulation_vector, q):
    return modulation_vector[int(x_quantized)]

def generate_combinations(K, q):
    M = q**K
    combinations = np.zeros((M, K), dtype=int)
    for i in range(M):
        val = i
        for k in range(K):
            combinations[i, k] = val % q
            val //= q
    return combinations

def generate_constellation_points(modulation_vector, q, num_users):
    M = q**num_users
    index_combinations = generate_combinations(num_users, q)
    constellation_points = np.zeros(M, dtype=complex)
    for i in range(M):
        for k in range(num_users):
            state = index_combinations[i, k]
            constellation_points[i] += modulation_vector[k*q + state]
    return constellation_points

def MLE_with_artificial_noise(y, modulation_vector, noise_vector, sigma_z, function_range):
    if sigma_z == 0:
        raise ValueError("sigma_z must be greater than 0 to avoid division by zero.")
    constellation_points = generate_constellation_points(modulation_vector, num_states, num_users)
    likelihoods = []
    for i, g_i in enumerate(constellation_points):
        w_i = noise_vector[i % len(noise_vector)]  # Ensure index within bounds
        likelihood = np.exp(-np.linalg.norm(y - (g_i + w_i))**2 / (2 * sigma_z**2)) / np.sqrt(2 * np.pi * sigma_z**2)
        likelihoods.append(likelihood)
    idx = np.argmax(likelihoods)
    return function_range[idx]  # Return the value corresponding to the most likely constellation point

def voronoi_decoder(y, modulation_vector, noise_vector, function_range):
    constellation_points = generate_constellation_points(modulation_vector, num_states, num_users)
    distances = [np.linalg.norm(y - (g_i + w_i))**2 for g_i, w_i in zip(constellation_points, noise_vector)]
    idx = np.argmin(distances)
    return function_range[idx]  # Return the value corresponding to the closest constellation point

# Parameters
num_users = 2
num_bits = 3
num_states = 2**num_bits
num_messages = num_states**num_users
num_channels = num_states * num_users
P = 1
sigma_y = 0.1
sigma_z = 1  # Ensure sigma_z is non-zero to avoid division by zero
sigma_h = 1
sigma_g = 1
SNR_dB = np.arange(0, 15, 2)
function_name = 'sum'
loss_factor = 1

def main():
    # Generate channel coefficients
    h, g = generate_channel_coefficients(num_users, sigma_h, sigma_g)

    # Calculate c values
    c_values = [np.sqrt(10**(snr_db / 10) * sigma_y / num_users) for snr_db in SNR_dB]
    c = c_values[0]  # Choose an example c for demonstration

    # Quantization and Modulation
    num_cases = int(num_messages * (num_messages - 1) / 2)
    input_domain = [n for n in range(num_states)]
    domain_values = [ele for ele in product(input_domain, repeat=num_users)]
    function_range = np.zeros(num_messages)

    for idx in range(num_messages):
        function_range[idx] = evaluate_function(domain_values[idx], function_name)

    matrix_A = np.zeros((num_messages, num_channels))
    count = 0
    for ele in product(range(num_states), repeat=num_users):
        for idx in range(num_users):
            matrix_A[count, idx * num_states + ele[idx]] = 1
        count += 1

    distance_function = [loss_factor * abs(f[0] - f[1]) ** 2 for f in combinations(function_range, 2)]
    distance_matrix_A = [np.outer(matrix_A[ele[0]] - matrix_A[ele[1]], matrix_A[ele[0]] - matrix_A[ele[1]]) for ele in combinations(range(num_messages), 2)]
    
    distance_function = []
    distance_matrix_A = []
    count = 0
    combinations_counter = [ele for ele in combinations(range(num_messages), 2)]
    for f in combinations(function_range, 2):
        if abs(f[0] - f[1]) != 0:
            temp_f = loss_factor * abs(f[0] - f[1]) ** 2
            distance_function.append(temp_f)
            ele = combinations_counter[count]
            temp_A = np.outer(matrix_A[ele[0]] - matrix_A[ele[1]], matrix_A[ele[0]] - matrix_A[ele[1]])
            distance_matrix_A.append(temp_A)
        count += 1
    length = len(distance_function)

    # Unified optimization problem
    X = cp.Variable((num_channels, num_channels), PSD=True)
    d_squared = cp.Variable(num_users - 1)

    modulation_objective = cp.trace(X)
    noise_objective = cp.sum(cp.multiply(d_squared, ((h[:num_users-1]**2 / h[num_users-1]**2) * g[num_users-1]**2 - 2 * (h[:num_users-1] / h[num_users-1]) * g[:num_users-1] * g[num_users-1] + g[:num_users-1]**2)))

    unified_objective = cp.Minimize(modulation_objective - noise_objective)
    modulation_constraints = [cp.trace(B @ X) >= distance_function[idx] for idx, B in enumerate(distance_matrix_A)]
    noise_constraints = [
        d_squared <= P - (c**2 / h[:num_users-1]**2),
        cp.sum(cp.multiply(d_squared, (h[:num_users-1]**2 / h[num_users-1]**2))) <= P - (c**2 / h[num_users-1]**2)
    ]

    constraints = modulation_constraints + noise_constraints

    prob = cp.Problem(unified_objective, constraints)
    result = prob.solve(solver=cp.SCS)

    # Extract singular vectors and noise precoding matrix
    X = X.value
    d = np.sqrt(d_squared.value)
    U, s, vh = svd(X, full_matrices=True)
    print("Rank of X using SVD:", len([x for x in s if x > 1e-10]))

    u0 = loss_factor * U[:, 0]
    u1 = loss_factor * U[:, 1]
    x = []
    y = []
    for idx in range(num_users):
        x.append(u0[idx * num_states:(idx + 1) * num_states])
        y.append(u1[idx * num_states:(idx + 1) * num_states])

    modulation_vector = np.concatenate(x)
    input_values = [np.random.randint(0, num_states) for _ in range(num_users)]
    encoded_signals = [encoder(val, modulation_vector, num_states) for val in input_values]

    A_prime = np.zeros((num_users - 1, num_users))
    for m in range(num_users - 1):
        A_prime[m, m] = 1
    A_prime[:, num_users - 1] = -h[:num_users - 1] / h[num_users - 1]

    precoding_matrix = np.diag(d) @ A_prime
    V = np.random.randn(num_users, num_users - 1)
    w = np.dot(precoding_matrix.T, V.T).T

    print(f'Input Values: {input_values}')
    print(f'Encoded Signals: {encoded_signals}')
    print(f'Modulation Vector: {modulation_vector}')
    print(f'Artificial Noise: {w}')
    print(f'Function Range: {function_range}')

    transmitted_signals = [c * (1/h_k) * x_k + w_k for h_k, x_k, w_k in zip(h, encoded_signals, w)]
    print(f'Transmitted Signals: {transmitted_signals}')

    y = sum(transmitted_signals) + sigma_y
    print(f'Received Signal at Legitimate Receiver: {y}')

    y_e = sum(g_k * (np.conj(g_k) / np.abs(g_k)**2) * (c * (1/g_k) * x_k + w_k) for g_k, x_k, w_k in zip(g, encoded_signals, w))
    y_e += sigma_z
    print(f'Received Signal at Eavesdropper: {y_e}')

    decoded_value_MLE = MLE_with_artificial_noise(y, modulation_vector, w, sigma_z, function_range)
    print(f'Decoded Value at CP (MLE): {decoded_value_MLE}')

    voronoi_decodere = voronoi_decoder(y, modulation_vector, w, function_range)
    print(f'Voronoi Decoder: {voronoi_decodere}')

    print(f'Constellation Points: {generate_constellation_points(modulation_vector, num_states, num_users)}')

if __name__ == "__main__":
    main()
