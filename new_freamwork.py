import numpy as np
import cvxpy as cp
from scipy.stats import rayleigh
from itertools import product, combinations
from numpy.linalg import svd, norm
# Generate channel coefficients
def generate_channel_coefficients(num_users, sigma_h, sigma_g):
    h = rayleigh.rvs(loc=0, scale=sigma_h, size=num_users)
    h[0] = np.sqrt(np.pi - np.sqrt(4 - np.pi)) / np.sqrt(2)
    g = rayleigh.rvs(loc=0, scale=sigma_g, size=num_users)
    h[1:] = np.maximum(h[1:], h[0])
    return h, g

# Optimize the precoding matrix
def optimize_precoding_matrix(num_users, h, g, c_squared, P, with_CSI):
    d_squared = cp.Variable(num_users - 1)
    if with_CSI:
        objective = cp.Maximize(cp.sum(cp.multiply(d_squared, ((h[:num_users-1]**2 / h[num_users-1]**2) * g[num_users-1]**2 - 2 * (h[:num_users-1] / h[num_users-1]) * g[:num_users-1] * g[num_users-1] + g[:num_users-1]**2))))
    else:
        objective = cp.Maximize(cp.sum(cp.multiply(d_squared, (2 * (h[:num_users-1]**2 / h[num_users-1]**2) - (np.pi / 2) * (h[:num_users-1] / h[num_users-1]) + 1))))
    
    constraints = [
        d_squared <= P - (c_squared / h[:num_users-1]**2),
        cp.sum(cp.multiply(d_squared, (h[:num_users-1]**2 / h[num_users-1]**2))) <= P - (c_squared / h[num_users-1]**2)
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if prob.status not in ["infeasible", "unbounded"]:
        d_squared_value = d_squared.value
        d_squared_value[d_squared_value < 0] = 0
        d = np.sqrt(d_squared_value)
    else:
        d = np.zeros(num_users - 1)
    
    return d

# Create reduced row echelon form
def create_reduced_row_echelon_form(h, num_users):
    A_prime = np.zeros((num_users - 1, num_users))
    for m in range(num_users - 1):
        A_prime[m, m] = 1
    A_prime[:, num_users - 1] = -h[:num_users - 1] / h[num_users - 1]
    return A_prime

def create_precoding_matrix(A_prime, d, num_users):
    # Ensure A_prime has the correct dimensions
    A_prime = A_prime[:, :num_users-1]
    return np.diag(d) @ A_prime

# Evaluate the function based on the function name
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

# Encode the quantized data
def encoder(x_quantized, modulation_vector, q):
    return modulation_vector[int(x_quantized)]

# Generate combinations of the input domain
def generate_combinations(K, q):
    M = q**K
    combinations = np.zeros((M, K), dtype=int)
    for i in range(M):
        val = i
        for k in range(K):
            combinations[i, k] = val % q
            val //= q
    return combinations

# Generate constellation points based on the modulation vector
def generate_constellation_points(modulation_vector, q, K):
    M = q**K
    index_combinations = generate_combinations(K, q)
    constellation_points = np.zeros(M, dtype=complex)
    for i in range(M):
        for k in range(K):
            state = index_combinations[i, k]
            constellation_points[i] += modulation_vector[k*q + state]
    return constellation_points

def compute_precoding_matrices(num_users, h, g, c_squared, P):
    # Optimize the precoding matrices without and with CSI
    d_without_CSI = optimize_precoding_matrix(num_users, h, g, c_squared, P, with_CSI=False)
    d_with_CSI = optimize_precoding_matrix(num_users, h, g, c_squared, P, with_CSI=True)

    # Create reduced row echelon form matrices
    A_prime_without_CSI = create_reduced_row_echelon_form(h, num_users)
    A_prime_with_CSI = create_reduced_row_echelon_form(h, num_users)

    # Create the precoding matrices
    A_without_CSI = create_precoding_matrix(A_prime_without_CSI, d_without_CSI, num_users)
    A_with_CSI = create_precoding_matrix(A_prime_with_CSI, d_with_CSI, num_users)

    V_without_CSI = np.random.randn(num_users - 1, 1)
    V_with_CSI = np.random.randn(num_users - 1, 1)
    
    # Calculate noise vectors w
    try:
        print(f"A_without_CSI shape: {A_without_CSI.shape}")
        print(f"V_without_CSI shape: {V_without_CSI.shape}")
        w_without_CSI = A_without_CSI @ V_without_CSI
        print(f"w_without_CSI shape: {w_without_CSI.shape}")
    except ValueError as e:
        print(f"Error in w_without_CSI calculation: {e}")
        w_without_CSI = None

    try:
        print(f"A_with_CSI shape: {A_with_CSI.shape}")
        print(f"V_with_CSI shape: {V_with_CSI.shape}")
        w_with_CSI = A_with_CSI @ V_with_CSI
        print(f"w_with_CSI shape: {w_with_CSI.shape}")
            
    except ValueError as e:
        print(f"Error in w_with_CSI calculation: {e}")
        w_with_CSI = None

    return w_without_CSI, w_with_CSI

# Maximum Likelihood Estimator (MLE) decoder
def maximum_likelihood_decoder(y, constellation_points, w_with_CSI, sigma_y):
    """
    Maximum Likelihood Estimator (MLE) decoder for ChannelComp with artificial noise.

    Parameters:
    y (np.ndarray): Received signal vector.
    constellation_points (np.ndarray): Array of possible constellation points (g_i).
    w_with_CSI (np.ndarray): Artificial noise vector.
    sigma_y (float): Total noise variance (including artificial noise).

    Returns:
    int: Index of the estimated constellation point.
    """
    num_points = constellation_points.shape[0]
    distances = np.zeros(num_points)

    # Subtract the artificial noise from the received signal
    y_adjusted = y - w_with_CSI

  
    idx = np.argmin(np.abs(y_adjusted - constellation_points))
    
    return idx

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
    
    # Ensure that q is defined in the context
    q = num_states
    input_values = [np.random.randint(0, q) for _ in range(num_users)]
    #Modulate the  data using the modulathion vector 
    encoded_signals = []
    for k in range(num_users):
        encoded_signals.append(encoder(input_values[k], x[k], q))

   
    print(f'Desired Function: {function_name}')
    # Print the transmitted values
    print('Transmitted Values:')
    for k in range(num_users):
        print(f'Node {k+1}: {input_values[k]}')

    # Generate channel coefficients
    h, g = generate_channel_coefficients(num_users, sigma_h, sigma_g)

    # Calculate c values
    c_values = [np.sqrt(10**(snr_db / 10) * sigma_y / num_users) for snr_db in SNR_dB]
    c = c_values[0]  # Choose an example c for demonstration

    # Optimize the precoding matrices and compute noise
    c_squared = c**2
    w_without_CSI, w_with_CSI = compute_precoding_matrices(num_users, h, g, c_squared, P)

    # Quantization and Modulation
    modulation_vector = np.concatenate(x)

   
    vec_t_k = np.sum(encoded_signals, axis=0) + np.sum(w_with_CSI, axis=0)
   
    print("Transmitted signals (vec_t_k):")
    print(vec_t_k)
    
    # Combine transmitted signals (simulating transmission over a MAC)
    combined_signal = np.sum(vec_t_k)
    print("combined_signal (combined_signal):")
    print(combined_signal)

    # Define vec_z as the noise vector at the receiver (assuming Gaussian noise for example)
    vec_z = np.random.normal(0, sigma_z, combined_signal.shape)
    # Aggregate the signals at the receiver
    vec_y = combined_signal + vec_z
    print("Aggregated received signal (vec_y):")
    print(vec_y)
    # Aggregate the signals at Eavesdropper
    vec_y_e = np.sum(g * combined_signal) + vec_z
    print("Aggregated received signal (vec_y_e) at the evesdropper:")
    print(vec_y_e)

    # Create all possible combinations of the modulation symbols
    constellation_points = generate_constellation_points(modulation_vector, q, num_users)

    # Decode the received signal at the legitimate receiver
    idx = maximum_likelihood_decoder(combined_signal,  constellation_points, w_with_CSI, sigma_y)
    decoded= function_range[idx]
    print(f"decoded Value: {decoded}")
    print("Estimated indexs:", idx)


    # attemp to Decode the received signal at Eavesdropper
    idx_eavesdropper = maximum_likelihood_decoder(vec_y_e, constellation_points, w_without_CSI, sigma_y)
    decoded_eavesdropper = function_range[idx_eavesdropper]
    print(f"Eavesdropper decoded Value: {decoded_eavesdropper}")
    print("Eavesdropper estimated index:", idx_eavesdropper)
if __name__ == "__main__":
    main()


    
