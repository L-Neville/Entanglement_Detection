"""
This module implements the broadcast of Pauli observables. For Pauli observables that commute, they share
same set of eigenstates, and their expectations can be evaluated from the same set of samples, a process named 
broadcast. In this module, firstly non-commuting observables are sampled on their bases, and then their exectations are 
broadcasted to all observables that commute with them, and finally all observables' expectations are evaluated. 
Currently, it only uses sampling, and no approximation method is used.
"""

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.quantum_info import Statevector, DensityMatrix, Pauli, random_clifford, random_statevector, partial_trace
from qiskit_aer import AerSimulator, Aer
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, logm
import cvxpy as cp
import pickle
import itertools, functools, collections, warnings, copy, winsound, os
from IPython.display import clear_output

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
    
def sample_from_dict(d, n_samples):
    def alias_setup(probs):
        n = len(probs)
        alias = np.zeros(n, dtype=int)
        prob = np.zeros(n, dtype=np.float64)
        scaled_probs = np.array(probs) * n
        small = []
        large = []
        for i, sp in enumerate(scaled_probs):
            if sp < 1.0:
                small.append(i)
            else:
                large.append(i)
        while small and large:
            small_idx = small.pop()
            large_idx = large.pop()

            prob[small_idx] = scaled_probs[small_idx]
            alias[small_idx] = large_idx

            scaled_probs[large_idx] = scaled_probs[large_idx] + scaled_probs[small_idx] - 1.0

            if scaled_probs[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)
        while large:
            large_idx = large.pop()
            prob[large_idx] = 1.0
        while small:
            small_idx = small.pop()
            prob[small_idx] = 1.0
        return alias, prob
    def alias_draw(alias, prob):
        n = len(alias)
        i = np.random.randint(n)
        if np.random.rand() < prob[i]:
            return i
        else:
            return alias[i]
    keys = list(d.keys())
    probs = list(d.values())
    alias, prob = alias_setup(probs)
    samples = [keys[alias_draw(alias, prob)] for _ in range(n_samples)]
    return samples

def tensor_prod(*tensors):
    if len(tensors) == 2:
        return np.kron(tensors[0], tensors[1])
    else:
        return np.kron(tensors[0], tensor_prod(*tensors[1:]))

def broadcast_string(s):
    def helper(s, index, current, results):
        if index == len(s):
            results.add(current)
            return
        # Option 1: Keep the current character
        helper(s, index + 1, current + s[index], results)
        # Option 2: Replace the current character with 'I'
        helper(s, index + 1, current + 'I', results)

    results = set()
    helper(s, 0, '', results)
    return results

def broadcast_all_strings(strings):
    all_broadcasts = set()
    for s in strings:
        all_broadcasts.update(broadcast_string(s))
    return all_broadcasts
    
def estimate_Pauli_expectations(dm, obsv, num_samples, simulation=False):
    num_samples = int(num_samples)
    if simulation: # simulate the process of sampling
        exp = np.real(np.trace(dm @ Pauli(obsv).to_matrix()))
        prob_p1 = (1 + exp) / 2
        prob_m1 = 1 - prob_p1
        samples = np.random.choice([+1, -1], size=num_samples, p=[prob_p1, prob_m1])
        return np.mean(samples)
    else: # use the approximate distribution instead
        exp = np.real(np.trace(dm @ Pauli(obsv).to_matrix()))
        num_samples_root = num_samples ** .5
        std_dev = (1 - exp ** 2) ** .5 / num_samples_root
        return np.random.normal(exp, std_dev)
    
def broadcast_Pauli_expectations(dm, num_qubits, obserables, num_samples):
    repetition = num_samples // len(obserables)
    original = set(obserables)
    broadcasted = broadcast_all_strings(obserables).difference(original)
    expectations = {key: [] for key in broadcasted.union(original)}
    remaining = copy.deepcopy(broadcasted)
    converter = {
        'X': np.array([[1, 1], [1, -1]]) / np.sqrt(2), 
        'Y': np.array([[1, 0], [0, 1j]], dtype=np.complex128) @ (np.array([[1, 1], [1, -1]]) / np.sqrt(2)), 
        'Z': np.array([[1, 0], [0, 1]])
    }
    all_states = [''.join(state) for state in list(itertools.product(*['01' for _ in range(num_qubits)]))]
    for obsv in original:
        overall_converter = tensor_prod(*[converter[s] for s in obsv])
        converted_dm = overall_converter.conj().T @ dm @ overall_converter
        distributions = {state: (State(state).to_vector().conj().T @ converted_dm @ State(state).to_vector())[0][0].real for state in all_states}
        samples = dict(collections.Counter(sample_from_dict(distributions, repetition)))
        probabilities = [samples[state] / repetition if state in samples.keys() else 0 for state in all_states]
        indices = range(len(obsv))
        parities = [(-1) ** (sum([1 for i in indices if state[i] == '1'])) for state in all_states]
        expectations[obsv] = sum([prob * parity for prob, parity in zip(probabilities, parities)])
        if len(remaining) > 0:
            for obs in broadcast_all_strings([obsv]).difference(obsv):
                if obs in remaining:
                    indices = [i for i in range(len(obs)) if not obs[i] == 'I']
                    parities = [(-1) ** (sum([1 for i in indices if state[i] == '1'])) for state in all_states]
                    expectations[obs] = sum([prob * parity for prob, parity in zip(probabilities, parities)])
                    remaining.remove(obs)
    return expectations