import numpy as np
import cvxpy as cp
from itertools import combinations, product
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt
import pandas as pd

# Configuration
num_bits = 3
num_states = 2**num_bits  # Number of states
num_users = 2  # Number of users
num_messages = num_states**num_users  # Number of messages
num_channels = num_states * num_users  # Number of channels
loss_factor = 1  # Loss factor
num_time_slots = 2  # Number of time slots

# Channel coefficients (example)
np.random.seed(42)  # For reproducibility
channel_coeffs = np.random.randn(num_users) + 1j * np.random.randn(num_users)
channel_powers = np.abs(channel_coeffs)**2

# Power control
transmit_powers = np.conj(channel_coeffs) / channel_powers

# Function selection
function_name = 'prod'  # Choose the function: 'prod', 'max', 'norm', 'max', 'frac1', 'frac2','frac3'

def evaluate_function(values, function_name):
    # Ensure values has enough elements for the requested operation
    if function_name == 'frac1' and len(values) < 3:
        raise ValueError("Not enough values for function 'frac1'; at least 3 are required.")
    elif function_name == 'frac2' and len(values) < 4:
        raise ValueError("Not enough values for function 'frac2'; at least 4 are required.")
    elif function_name == 'frac3' and len(values) < 2:
        raise ValueError("Not enough values for function 'frac3'; at least 2 are required.")
    
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

# Calculate the number of all possible cases
num_cases = int(num_messages * (num_messages - 1) / 2)

# Generate input domain values
input_domain = [n for n in range(num_states)]

# Generate all values in the domain
domain_values = [ele for ele in product(input_domain, repeat=num_users)]

# Initialize function range
function_range = np.zeros(num_messages)

# Calculate function range
for idx in range(num_messages):
    function_range[idx] = evaluate_function(domain_values[idx], function_name)

# Generate matrix A
matrix_A = np.zeros((num_messages, num_channels))
count = 0
for ele in product(range(num_states), repeat=num_users):
    for idx in range(num_users):
        matrix_A[count, idx * num_states + ele[idx]] = 1
    count += 1

# Generate distance matrices
distance_function = [loss_factor * abs(f[0] - f[1]) ** 2 for f in combinations(function_range, 2)]
distance_matrix_A = [np.outer(matrix_A[ele[0]] - matrix_A[ele[1]], matrix_A[ele[0]] - matrix_A[ele[1]]) for ele in combinations(range(num_messages), 2)]

# Remove zero input values from distance matrices
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

# Construct the problem.
X = cp.Variable((num_channels, num_channels), PSD=True)
objective = cp.Minimize(cp.trace(X))
all_B = [cp.trace(B @ X) for B in distance_matrix_A]
constraints = [all_B[idx] >= distance_function[idx] for idx in range(length)]
prob = cp.Problem(objective, constraints)

# Solve the problem
result = prob.solve('SCS', use_indirect=True, verbose=True)

# Extract singular vectors
'''
if rank_X_opt == 1:
    # Obtain optimal modulation vector solution via Cholesky decomposition
    print("Obtain optimal modulation vector solution via Cholesky decomposition")
    modulation_vector = np.linalg.cholesky(X).T.flatten()
else:
    print(f'Recover sub-optimal solution using Gaussian randomization method')
    # Recover sub-optimal solution using Gaussian randomization method
    num_samples = 1000  # Number of samples for randomization
    suboptimal_modulation_vectors = []
    for _ in range(num_samples):
        try:
            gaussian_noise = np.random.normal(0, 1, X.shape)
            random_matrix = X + gaussian_noise
            suboptimal_modulation_vector = np.linalg.cholesky(random_matrix).T.flatten()
            suboptimal_modulation_vectors.append(suboptimal_modulation_vector)
        except np.linalg.LinAlgError:
            continue  # Skip if Cholesky decomposition fails
    if suboptimal_modulation_vectors:
        modulation_vector = np.mean(suboptimal_modulation_vectors, axis=0)
    else:
        modulation_vector = np.zeros(X.shape[0])  # Fallback if all decompositions fail
'''
X = X.value
U, s, vh = svd(X, full_matrices=True)
print("Rank of X using SVD:", len([x for x in s if x > 1e-10]))  # Print the rank of X using SVD
u0 = loss_factor * U[:, 0]
u1 = loss_factor * U[:, 1]
x = []
y = []
for idx in range(num_users):
    x.append(u0[idx * num_states:(idx + 1) * num_states])
    y.append(u1[idx * num_states:(idx + 1) * num_states])



# Print the desired function
print(f'Desired Function: {function_name}')

# Encode and transmit signals
def encoder(x_quantized, modulation_vector, q):
    return modulation_vector[x_quantized]

# Ensure that q is defined in the context
q = num_states
input_values = [np.random.randint(0, q) for _ in range(num_users)]

encoded_signals = []
for k in range(num_users):
    encoded_signals.append(encoder(input_values[k], x[k], q))

# Print the transmitted values
print('Transmitted Values:')
for k in range(num_users):
    print(f'Node {k+1}: {input_values[k]}')

# Combine transmitted signals (simulating transmission over a MAC)
combined_signal = sum(encoded_signals)

# Generate all possible combinations of the modulation symbols
def generate_combinations(K, q):
    M = q**K
    combinations = np.zeros((M, K), dtype=int)
    for i in range(M):
        val = i
        for k in range(K):
            combinations[i, k] = val % q
            val //= q
    return combinations

# Receiver decoding using Maximum Likelihood Estimation (MLE)
def receiver_decoding(received_signal, modulation_vector, f_values, q, K):
    constellation_points = generate_constellation_points(modulation_vector, q, K)
    idx = np.argmin(np.abs(received_signal - constellation_points))
    return f_values[idx]

# Generate constellation points
def generate_constellation_points(modulation_vector, q, K):
    M = q**K
    index_combinations = generate_combinations(K, q)
    constellation_points = np.zeros(M, dtype=complex)
    for i in range(M):
        for k in range(K):
            state = index_combinations[i, k]
            constellation_points[i] += modulation_vector[k*q + state]
    return constellation_points

# Decode the received signal
decoded_value = receiver_decoding(combined_signal, np.concatenate(x), function_range, q, num_users)

# Print the decoded value at CP
print(f'Decoded Value at CP: {decoded_value}')

import matplotlib.pyplot as plt

def plotFunction(num_users, function_name, x, y):
    legend = [f'$\\vec{{x}}_{idx+1}$' for idx in range(num_users)]
    
    # Define markers and colors
    markers = ['o', '^', 's', 'p', '*', 'x', 'D', 'v', 'h']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    plt.figure(figsize=(6, 6))
    for idx in range(num_users):
        plt.scatter(x[idx], y[idx], s=100, alpha=0.7, marker=markers[idx % len(markers)], color=colors[idx % len(colors)], label=legend[idx])
    
    plt.xlabel('Real$(x)$')
    plt.ylabel('Imag$(x)$')
    plt.legend(loc='upper left', fontsize='medium')
    plt.title(f'$f(\\mathbf{{x}}) = {function_name}(x_1, x_2)$', fontsize=14)
    plt.grid(color=(0.6, 0.73, 0.89), linestyle='--', linewidth=1)
    plt.savefig('Summation.png')
    plt.show()
    plt.close()

plotFunction(num_users, function_name, x, y)

# Save the modulation vectors to a file
MaxModulation = pd.DataFrame({'Y1': y[0], 'Y2': y[1], 'X1': x[0], 'X2': x[1]})
MaxModulation.set_index('Y1')
MaxModulation.to_csv('SumPam.dat', index=False, sep=' ')
MaxModulation = pd.DataFrame({'Y1': y[0], 'Y2': y[1], 'X1': x[0], 'X2': x[1]})
MaxModulation.set_index('Y1')
print(MaxModulation)


