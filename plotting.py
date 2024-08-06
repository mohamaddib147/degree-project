from matplotlib import pyplot as plt
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
def calculate_MSE_new(num_users, c, h, g, A_with_CSI, A_without_CSI, sigma_y, sigma_z):
    """
    Calculates the MSE for the legitimate receiver and the eavesdropper.
    """
    D = num_users - (c**2 * num_users**2) / (c**2 * num_users + 1 * sigma_y)
    
    def calculate_S(A):
        numerator = c**2 * (np.sum(g / h))**2
        denominator = c**2 * np.sum((g**2) / (h**2)) + 1 * (np.linalg.norm(A)**2 + sigma_z)
        return num_users - numerator / denominator
    
    S_with_CSI = calculate_S(A_with_CSI)
    S_without_CSI = calculate_S(A_without_CSI)
    S_noiseless = calculate_S(0)
    
    
    return D, S_with_CSI, S_without_CSI, S_noiseless
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
num_runs = 10
def main():
    print('Starting simulations...')
MSE_legitimate_avg = np.zeros(len(SNR_dB))
MSE_eavesdropper_with_CSI_avg = np.zeros(len(SNR_dB))
MSE_eavesdropper_without_CSI_avg = np.zeros(len(SNR_dB))
MSE_eavesdropper_noiseless_avg = np.zeros(len(SNR_dB))


for idx, snr_db in enumerate(SNR_dB):
    print(f'Processing SNR = {snr_db} dB')
    MSE_legitimate = np.zeros(num_runs)
    MSE_eavesdropper_with_CSI = np.zeros(num_runs)
    MSE_eavesdropper_without_CSI = np.zeros(num_runs)
    MSE_eavesdropper_noiseless = np.zeros(num_runs)
    

    for run in range(num_runs):
        if run % 100 == 0:
            print(f'  Run {run} of {num_runs}')

        # Generate channel coefficients
        h, g = generate_channel_coefficients(num_users, sigma_h, sigma_g)

        # Calculate c values
        c = np.sqrt(10**(snr_db / 10) * sigma_y / num_users)
        c_squared = c**2

        # Optimize the precoding matrices and compute noise
        w_without_CSI, w_with_CSI = compute_precoding_matrices(num_users, h, g, c_squared, P)

        # Generate the precoding matrices
        A_prime_without_CSI = create_reduced_row_echelon_form(h, num_users)
        A_prime_with_CSI = create_reduced_row_echelon_form(h, num_users)

        A_without_CSI = create_precoding_matrix(A_prime_without_CSI, w_without_CSI, num_users)
        A_with_CSI = create_precoding_matrix(A_prime_with_CSI, w_with_CSI, num_users)

        # Calculate MSE
        D, S_with_CSI, S_without_CSI, S_noiseless = calculate_MSE_new(
            num_users, c, h, g, A_with_CSI, A_without_CSI, sigma_y, sigma_z
        )

        

        # Store MSE results
        MSE_legitimate[run] = D
        MSE_eavesdropper_with_CSI[run] = S_with_CSI
        MSE_eavesdropper_without_CSI[run] = S_without_CSI
        MSE_eavesdropper_noiseless[run] = S_noiseless
        

    MSE_legitimate_avg[idx] = np.mean(MSE_legitimate)
    MSE_eavesdropper_with_CSI_avg[idx] = np.mean(MSE_eavesdropper_with_CSI)
    MSE_eavesdropper_without_CSI_avg[idx] = np.mean(MSE_eavesdropper_without_CSI)
    MSE_eavesdropper_noiseless_avg[idx] = np.mean(MSE_eavesdropper_noiseless)
    

# Plotting results
plt.figure()
plt.plot(SNR_dB, MSE_legitimate_avg, 'b-o', label='Legitimate Receiver')
plt.plot(SNR_dB, MSE_eavesdropper_with_CSI_avg, 'r-s', label='Eavesdropper with CSI')
plt.plot(SNR_dB, MSE_eavesdropper_without_CSI_avg, 'g-^', label='Eavesdropper without CSI')
plt.plot(SNR_dB, MSE_eavesdropper_noiseless_avg, 'm-*', label='Eavesdropper Noise-less')

plt.xlabel('SNR (dB)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE vs. SNR for Legitimate Receiver and Eavesdropper')
plt.legend()
plt.grid(True)
plt.show()

print('Simulation completed.')
if __name__ == "__main__":
    main()


    
