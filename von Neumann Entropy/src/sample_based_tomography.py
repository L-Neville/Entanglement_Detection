"""
This module focuses on function sampled_based_tomography. 
It takes probability distribution of all states in every measurement bases as
inputs, and return the reconstructed density matrix. The input should have the 
type dict[str, np.ndarray], e.g. for a 2-qubit system, 
samples['ZI'] = np.array([.1, .2, .3, .4]), reflecting probabilities for states 
00, 01, 10 and 11. Keys 'ZZ', 'ZI', 'IZ', 'II' should have same values.
Definition of main function starts from Line 121.
"""

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.quantum_info import Statevector, DensityMatrix, Pauli, Operator, random_statevector, partial_trace
from qiskit_aer import AerSimulator, Aer
import numpy as np
from scipy.linalg import sqrtm
import cvxpy as cp
import itertools
from IPython.display import clear_output
from winsound import Beep

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.quantum_info import Statevector, DensityMatrix, Pauli, Operator, random_statevector
from qiskit_aer import AerSimulator, Aer
import numpy as np
from scipy.linalg import sqrtm
import cvxpy as cp
from IPython.display import clear_output
from winsound import Beep

def tensor_prod(*tensors):
    if len(tensors) == 2:
        return np.kron(tensors[0], tensors[1])
    else:
        return np.kron(tensors[0], tensor_prod(*tensors[1:]))
    
def hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)

def trace_one(matrix):
    return np.isclose(np.trace(matrix), 1)

def positive_semi_definite(matrix, tol=1e-8):
    return np.all(np.linalg.eigvals(matrix) + tol >= 0)

def is_legal(matrix):
    return hermitian(matrix) and trace_one(matrix) and positive_semi_definite(matrix)

def check_legal(matrix, print_errors=True, name=None, indent=False):
    errors, legal = [], True
    if not hermitian(matrix):
        errors.append('not hermitian')
    if not trace_one(matrix):
        errors.append('trace not equal to one')
    if not positive_semi_definite(matrix):
        errors.append('not positive semidefinite')
    if len(errors) > 0:
        legal = False
    if print_errors:
        msg = '    ' if indent else ''
        msg += 'input' if (name is None) else name
        msg += ' '
        if not legal:
            print(msg + '; '.join(errors))
        else: 
            print(msg + 'is a legal density matrix')
    return legal
        
def generate_prob_lst(num_states):
    prob_lst = np.array([np.random.random() for _ in range(num_states)])
    prob_lst /= np.sum(prob_lst)
    return prob_lst

def get_rank(dm, tol=1e-10):
    return int(np.sum(np.linalg.eigvalsh(dm) > tol))

def get_fidelity(dm1, dm2, tol=1e-5):
    # assert is_legal(dm1) and is_legal(dm2), 'inputs are not legal density matrices'
    if not is_legal(dm1) and is_legal(dm2):
        print("Warning: inputs are not legal density matrices")
    try: 
        fidelity = (np.trace(sqrtm(sqrtm(dm1) @ dm2 @ sqrtm(dm1)))) ** 2
    except ValueError:
        print('fidelity cannot be computed for given inputs')
    if np.abs(np.imag(fidelity)) > tol: 
        print('Warning: fidelity is not real within tol')
    return fidelity.real

def generate_dm(num_qubits, num_states, state_lst=None, prob_lst=None):
    if state_lst is None:
        state_lst = [random_statevector(2**num_qubits) for _ in range(num_states)]
    if prob_lst is None:
        prob_lst = generate_prob_lst(num_states)
    density_matrix = sum([DensityMatrix(state_lst[i]).data * prob_lst[i] for i in range(num_states)])
    return density_matrix

def generate_Pauli_expectations(dm, obsv):
    return np.trace(dm @ Pauli(obsv).to_matrix()).real

def get_trace_norm(dm):
    return np.sum(np.linalg.svd(dm, compute_uv=False))

single_states = {'0': np.array([[1], [0]]), '1': np.array([[0], [1]])}

class State():
    def __init__(self, strings):
        if len(strings) == 1:
            self.state = single_states[strings]
        else:
            singles = [single_states[s] for s in strings]
            self.state = tensor_prod(*singles)
    def to_vector(self):
        return self.state
    
Hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

Phase = np.array([[1, 0], [0, 1j]], dtype=np.complex128)

def expct_based_tomography(num_qubits:int, Pauli_expectations:dict[str, np.float32])->np.ndarray:
    '''
    Given expectations of all Pauli obervables, return the reconstructed density matrix.
    '''
    return sum([Pauli(obsv).to_matrix() * expct for obsv, expct in Pauli_expectations.items()]) / 2 ** num_qubits

def sample_based_tomography(num_qubits:int, samples:dict[str, np.ndarray], output_expct=False)->np.ndarray:
    '''
    Given probability distribution of all states in every measurement bases,
    return the reconstructed density matrix.
    '''
    Pauli_expectations = dict()
    for observable, distribution in samples.items():
        all_states = [''.join(state) for state in list(itertools.product(*['01' for _ in range(num_qubits)]))]
        indices = [i for i in range(len(observable)) if not observable[i] == 'I']
        parities = [(-1) ** (sum([1 for i in indices if state[i] == '1'])) for state in all_states]
        Pauli_expectations[observable] = sum([prob * parity for prob, parity in zip(distribution, parities)])
    if output_expct:
        return Pauli_expectations
    return expct_based_tomography(num_qubits, Pauli_expectations)

def generate_Pauli_expectations(num_qubits:int, dm:np.ndarray)->dict[str, np.float32]:
    """
    Given the density matrix, generate expectations of all Pauli observables.
    """
    return {observable: np.trace(dm @ Pauli(observable).to_matrix()).real 
            for observable in [''.join(obsv) for obsv in list(itertools.product(*['IXYZ' for _ in range(num_qubits)]))]}
    
def generate_distribution(num_qubits:int, dm:np.ndarray)->dict[str, np.ndarray]:
    """
    Given the density matrix, generate distribution of all states in every measurement bases.
    """
    converter = {
        'X': Hadamard, 
        'Y': Phase @ Hadamard, 
        'Z': Pauli('I').to_matrix(),
        'I': Pauli('I').to_matrix()
    }
    all_states = [''.join(state) for state in list(itertools.product(*['01' for _ in range(num_qubits)]))]
    all_distributions = dict()
    all_bases = [''.join(obsv) for obsv in list(itertools.product(*['IXYZ' for _ in range(num_qubits)]))]
    for basis in all_bases:
        overall_converter = tensor_prod(*[converter[s] for s in basis])
        converted_dm = overall_converter.conj().T @ dm @ overall_converter
        all_distributions[basis] = np.array([(State(state).to_vector().conj().T @ converted_dm @ State(state).to_vector())[0][0].real for state in all_states])
    return all_distributions

# Test of tomography and generation functions
if __name__ == "__main__":
    num_qubits = 3
    rho = generate_dm(num_qubits, 2 ** num_qubits)
    print("1. fidelity test:")
    rho1 = expct_based_tomography(num_qubits, generate_Pauli_expectations(num_qubits, rho))
    check_legal(rho1, name='rho1', indent=True)
    print(f"    fidelity of expectation-based tomography: {get_fidelity(rho, rho1)}")
    print(f"    trace of expectation-based result: {np.trace(rho1)}")
    print(f"    eigenvalues of expectation-based result: {np.round(np.linalg.eigvals(rho1), 2)}")
    rho2 = sample_based_tomography(num_qubits, generate_distribution(num_qubits, rho))
    check_legal(rho2, name='rho2', indent=True)
    print(f"    fidelity of samples-based tomography: {get_fidelity(rho, rho2)}")
    print(f"    trace of expectation-based result: {np.trace(rho2)}")
    print(f"    eigenvalues of expectation-based result: {np.round(np.linalg.eigvals(rho2), 2)}")
    print("2. expectation test:")
    observables = [''.join(obsv) for obsv in list(itertools.product(*['IXYZ' for _ in range(num_qubits)]))]
    selected_observables = np.random.choice(observables, 2 ** num_qubits, replace=False)
    tomography_results = sample_based_tomography(num_qubits, generate_distribution(num_qubits, rho), output_expct=True)
    for observable in selected_observables:
        print(f"    observable {observable}: tomography {tomography_results[observable]}, true value {np.trace(rho @ Pauli(observable).to_matrix()).real}")
        
        