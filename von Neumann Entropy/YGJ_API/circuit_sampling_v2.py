from scipy.linalg import sqrtm
from scipy.linalg import logm
from scipy.linalg import eigvals

import deepquantum as dq
import deepquantum.photonic as dqp

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import copy


def noise(sigma=0.1):
    return np.random.normal(0, sigma)

class DVCircuit():

    def __init__(self, ifnoise=False, 
                 param = {
                    'theta1': 1,
                    'theta2': 1,
                    'MZI1' : [1, 1],
                    'MZI2' : [1, 1],
                    'MZI3' : [1, 1],
                    'MZI4' : [1, 1],
                 },
                 repeatation = 1,
                 sigma = 0.1):

        num_mode = 10

        self.Qa = [4, 5]
        self.Q1 = [2, 3]
        self.Q2 = [6, 7]
        self.E1 = 1
        self.E2 = 8
        self.e1 = 0
        self.e2 = 9

        # |0>_1 \otimes (|00>_2a + |11>_2a) / sqrt(2)
        state1 = np.zeros(num_mode, dtype=int)
        state1[self.Q1[0]] = 1
        state1[self.Qa[0]] = 1
        state1[self.Q2[0]] = 1

        state2 = np.zeros(num_mode, dtype=int)
        state2[self.Q1[0]] = 1
        state2[self.Qa[1]] = 1
        state2[self.Q2[1]] = 1

        qstate = dq.FockState(  state=[(1/np.sqrt(2), state1), 
                                    (1/np.sqrt(2), state2)], 
                                basis=False, cutoff=2)

        self.cir = dq.QumodeCircuit(nmode=num_mode, init_state=qstate, 
                            cutoff=2, backend='fock', basis=False)

        if not ifnoise:
            sigma = 0
            
        # rotate the input state
        theta1 = param['theta1']
        theta2 = param['theta2']
        self.cir.bs_rx(wires=self.Q1, inputs=-theta1+noise(sigma))
        self.cir.bs_rx(wires=self.Q2, inputs=-theta2+noise(sigma))

        for _ in range(repeatation):

            # swap
            self.cir.mzi(wires=[self.Q1[1], self.Q2[0]], inputs=[0+noise(sigma), 0+noise(sigma)])
            self.cir.ps(wires=self.Q1[1], inputs=-np.pi/2+noise(sigma))
            self.cir.ps(wires=self.Q2[0], inputs=-np.pi/2+noise(sigma))

            # collision
            self.cir.mzi(wires=[self.E1, self.Q1[0]], inputs=[param['MZI1'][0]+noise(sigma), param['MZI1'][1]+noise(sigma)])
            self.cir.mzi(wires=[self.Q2[1], self.E2], inputs=[param['MZI2'][0]+noise(sigma), param['MZI2'][1]+noise(sigma)])
            self.cir.mzi(wires=[self.e1, self.E1], inputs=[param['MZI3'][0]+noise(sigma), param['MZI3'][1]+noise(sigma)])
            self.cir.mzi(wires=[self.E2, self.e2], inputs=[param['MZI4'][0]+noise(sigma), param['MZI4'][1]+noise(sigma)])

    def get_circuit(self):
        return self.cir

    def normal_sampling(self, shots=1000):
        cir = self.cir
        sample = cir.measure(shots=shots,wires=self.Q1+self.Qa+self.Q2)
        return sample

    def basis_sampling(self, basis='iii', shots=1000):

        cir = copy.deepcopy(self.cir)

        # measurement
        if basis[0] == 'x':
            cir.bs_h(wires=self.Q1, inputs=np.pi)
        elif basis[0] == 'y':
            cir.mzi(wires=self.Q1, inputs=[0, np.pi])
            cir.ps(wires=self.Q1[0], inputs=np.pi)
            cir.ps(wires=self.Q1[1], inputs=np.pi)
        elif basis[0] == 'z':
            cir.bs_h(wires=self.Q1, inputs=0)
        
        if basis[1] == 'x':
            cir.bs_h(wires=self.Qa, inputs=np.pi)
        elif basis[1] == 'y':
            cir.mzi(wires=self.Qa, inputs=[0, np.pi])
            cir.ps(wires=self.Qa[0], inputs=np.pi)
            cir.ps(wires=self.Qa[1], inputs=np.pi)
        elif basis[1] == 'z':
            cir.bs_h(wires=self.Qa, inputs=0)

        if basis[2] == 'x':
            cir.bs_h(wires=self.Q2, inputs=np.pi)
        elif basis[2] == 'y':
            cir.mzi(wires=self.Q2, inputs=[0, np.pi])
            cir.ps(wires=self.Q2[0], inputs=np.pi)
            cir.ps(wires=self.Q2[1], inputs=np.pi)
        elif basis[2] == 'z':
            cir.bs_h(wires=self.Q2, inputs=0)

        # measurement
        cir()

        sample = cir.measure(shots=shots,wires=self.Q1+self.Qa+self.Q2)

        correct_sample = []
        wanted_state = [dq.FockState([1,0,1,0,1,0]), 
                        dq.FockState([1,0,1,0,0,1]),
                        dq.FockState([1,0,0,1,1,0]),
                        dq.FockState([1,0,0,1,0,1]),
                        dq.FockState([0,1,1,0,1,0]),
                        dq.FockState([0,1,1,0,0,1]),
                        dq.FockState([0,1,0,1,1,0]),
                        dq.FockState([0,1,0,1,0,1])]
        for w_state in wanted_state:
            try:
                correct_sample.append(sample[w_state])
            except:
                correct_sample.append(0)

        correct_sample = np.array(correct_sample)
        pdf = correct_sample / np.sum(correct_sample)
        # import time
        # self.cir.draw('pic/circuit{}.svg'.format(time.time()))
        return pdf
    
    def basis_output_prob(self, basis='iii', shots=10):
        
        cir = copy.deepcopy(self.cir)

        # measurement
        if basis[0] == 'x':
            cir.bs_h(wires=self.Q1, inputs=np.pi)
        elif basis[0] == 'y':
            cir.mzi(wires=self.Q1, inputs=[0, np.pi])
            cir.ps(wires=self.Q1[0], inputs=np.pi)
            cir.ps(wires=self.Q1[1], inputs=np.pi)
        elif basis[0] == 'z':
            cir.bs_h(wires=self.Q1, inputs=0)
        
        if basis[1] == 'x':
            cir.bs_h(wires=self.Qa, inputs=np.pi)
        elif basis[1] == 'y':
            cir.mzi(wires=self.Qa, inputs=[0, np.pi])
            cir.ps(wires=self.Qa[0], inputs=np.pi)
            cir.ps(wires=self.Qa[1], inputs=np.pi)
        elif basis[1] == 'z':
            cir.bs_h(wires=self.Qa, inputs=0)

        if basis[2] == 'x':
            cir.bs_h(wires=self.Q2, inputs=np.pi)
        elif basis[2] == 'y':
            cir.mzi(wires=self.Q2, inputs=[0, np.pi])
            cir.ps(wires=self.Q2[0], inputs=np.pi)
            cir.ps(wires=self.Q2[1], inputs=np.pi)
        elif basis[2] == 'z':
            cir.bs_h(wires=self.Q2, inputs=0)

        # measurement
        cir()

        sample = cir.measure(shots=shots,wires=self.Q1+self.Qa+self.Q2, with_prob=True)

        prob = []
        wanted_state = [dq.FockState([1,0,1,0,1,0]), 
                        dq.FockState([1,0,1,0,0,1]),
                        dq.FockState([1,0,0,1,1,0]),
                        dq.FockState([1,0,0,1,0,1]),
                        dq.FockState([0,1,1,0,1,0]),
                        dq.FockState([0,1,1,0,0,1]),
                        dq.FockState([0,1,0,1,1,0]),
                        dq.FockState([0,1,0,1,0,1])]
        for w_state in wanted_state:
            try:
                prob.append(sample[w_state][1])
            except:
                prob.append(0)

        prob = np.array(prob)
        prob = prob / np.sum(prob)
        return prob

    def sampling_all_observable(self, shots=1000):

        observable_list = []
        for i in ['i','x','y','z']:
            for j in ['i','x','y','z']:
                for k in ['i','x','y','z']:
                    observable_list.append(i+j+k)

        sampling_prob_all = np.zeros((64, 8))

        for i in range(64):
            observable = observable_list[i]
            sampling_prob_all[i] = self.basis_sampling(basis=observable, shots=shots)

        return sampling_prob_all
    
    def prob_all_observable(self, shots=1000):

        observable_list = []
        for i in ['i','x','y','z']:
            for j in ['i','x','y','z']:
                for k in ['i','x','y','z']:
                    observable_list.append(i+j+k)

        prob_all = np.zeros((64, 8))

        for i in range(64):
            observable = observable_list[i]
            prob_all[i] = self.basis_output_prob(basis=observable, shots=shots)

        return prob_all
    
    def tomography(self, if_sampling=False, shots=1000):

        if not if_sampling:
            prob_all = self.prob_all_observable()
        else:
            prob_all = self.sampling_all_observable(shots=shots)

        # average of observable
        # <O> = p_i <i|O|i>

        sigma_x = np.array([[0, 1],
                            [1, 0]])
        sigma_y = np.array([[0, -1j],
                            [1j, 0]])
        sigma_z = np.array([[1, 0],
                            [0, -1]])
        sigma_i = np.array([[1, 0],
                            [0, 1]])
        
        order = {'i':0, 'x':1, 'y':2, 'z':3}
        pauli = {'x': sigma_x, 'y': sigma_y, 'z': sigma_z, 'i': sigma_i}
        rho_hat = np.zeros((8, 8), dtype=complex)
        state = ['000', '001', '010', '011', '100', '101', '110', '111']

        for i in ['i','x','y','z']:
            for j in ['i','x','y','z']:
                for k in ['i','x','y','z']:

                    observable = np.kron(pauli[i],np.kron(pauli[j],pauli[k]))
                    
                    ave_observable = 0
                    for m in range(8):
                        pm = prob_all[order[k]+4*order[j]+16*order[i]][m]
                        ave_observable += pm * eigenvalues(state[m], i+j+k)
                    
                    # print(ave_observable, observable)
                    rho_hat += ave_observable * observable

        rho_hat = rho_hat / (2**3)
        self.rho = rho_hat
                    
        return rho_hat


    def mutual_information(self):

        rho = self.rho

        rho_A = partial_trace_BC(rho, 2,2,2)
        rho_B = partial_trace_AC(rho, 2,2,2)
        rho_C = partial_trace_AB(rho, 2,2,2)
        rho_AB = partial_trace_C(rho, 2,2,2)
        rho_BC = partial_trace_A(rho, 2,2,2)
        rho_AC = partial_trace_B(rho, 2,2,2)
        rho_ABC = rho

        I2_A_B = I2(rho_AB, rho_A, rho_B)
        I2_A_C = I2(rho_AC, rho_A, rho_C)
        I2_B_C = I2(rho_BC, rho_B, rho_C)
        I3_A_B_C = I3(rho_ABC, rho_AB, rho_AC, rho_BC, rho_A, rho_B, rho_C)

        return I2_A_B, I2_A_C, I2_B_C, I3_A_B_C
       
# 可观测量特征值
def eigenvalues(state, observable):
    eigen = 1
    for i in range(3):
        if state[i] == '0':
            eigen *= 1
        elif observable[i] == 'i':
            eigen *= 1
        else:
            eigen *= -1
    return eigen

# 美化输出
def table(matrix):
    return pd.DataFrame(matrix)

# 计算 rho 的平方根
def sqrt_matrix(matrix):
    sqrt_rho = sqrtm(matrix)
    return sqrt_rho

# 计算两个密度矩阵的保真度
def compute_fidelity(rho, sigma):
    sqrt_rho = sqrt_matrix(rho)
    return np.trace(sqrtm(sqrt_rho @ sigma @ sqrt_rho)).real

# 生成高斯随机矩阵，并修改为哈密顿矩阵
def random_hermitian_matrix(N, sigma=1):
    A = np.random.normal(0, sigma, (N, N)) + 1j * np.random.normal(0, sigma, (N, N))
    return np.matrix(A + A.conj().T)/2

# 生成 sigma, 在目标密度矩阵附近生成一定保真度的随机密度矩阵
def generate_sigma(rho, ratio=0.1, fidelity=0.99):
    
    from scipy.linalg import expm

    H = random_hermitian_matrix(rho.shape[0]) * ratio
    sigma = expm(-1j*H*ratio) @ rho @ expm(1j*H*ratio)

    while compute_fidelity(rho, sigma) < fidelity:
        H = random_hermitian_matrix(rho.shape[0]) * ratio
        sigma = expm(-1j*H*ratio) @ rho @ expm(1j*H*ratio)
    
    return compute_fidelity(rho, sigma), sigma

# ------------------------------------------
# 求偏迹
def partial_trace_AB(rho_ABC, dim_A, dim_B, dim_C):

    rho_C = np.zeros((dim_C, dim_C), dtype=complex)
    for i in range(dim_A):
        for j in range(dim_B):
            # |i> |j>
            ket_i = np.zeros((dim_A,1), dtype=complex)
            ket_i[i] = 1
            ket_j = np.zeros((dim_B,1), dtype=complex)
            ket_j[j] = 1
            I_C = np.eye(dim_C)

            rho_C += np.kron(ket_i.conj().T, np.kron(ket_j.conj().T, I_C)) @ rho_ABC @ np.kron(ket_i, np.kron(ket_j, I_C))
    
    return rho_C

def partial_trace_BC(rho_ABC, dim_A, dim_B, dim_C):
    rho_A = np.zeros((dim_A, dim_A), dtype=complex)
    for i in range(dim_B):
        for j in range(dim_C):
            ket_i = np.zeros((dim_B, 1), dtype=complex)
            ket_i[i] = 1
            ket_j = np.zeros((dim_C, 1), dtype=complex)
            ket_j[j] = 1
            I_A = np.eye(dim_A)  # I_A 是 A 系统的单位矩阵
            
            a = np.kron(I_A, np.kron(ket_i.conj().T, ket_j.conj().T)) @ rho_ABC @ np.kron(I_A, np.kron(ket_i, ket_j))
            # print(a.shape)
            # 跟踪 B 和 C
            rho_A += np.kron(I_A, np.kron(ket_i.conj().T, ket_j.conj().T)) @ rho_ABC @ np.kron(I_A, np.kron(ket_i, ket_j))
    
    return rho_A

def partial_trace_AC(rho_ABC, dim_A, dim_B, dim_C):
    rho_B = np.zeros((dim_B, dim_B), dtype=complex)
    for i in range(dim_A):
        for j in range(dim_C):
            ket_i = np.zeros((dim_A, 1), dtype=complex)
            ket_i[i] = 1
            ket_j = np.zeros((dim_C, 1), dtype=complex)
            ket_j[j] = 1
            I_B = np.eye(dim_B)  # I_B 是 B 系统的单位矩阵
            
            # 跟踪 A 和 C
            rho_B += np.kron(ket_i.conj().T, np.kron(I_B, ket_j.conj().T)) @ rho_ABC @ np.kron(ket_i, np.kron(I_B, ket_j))
    
    return rho_B

def partial_trace_C(rho_ABC, dim_A, dim_B, dim_C):
    rho_AB = np.zeros((dim_A * dim_B, dim_A * dim_B), dtype=complex)
    for i in range(dim_C):
        ket_i = np.zeros((dim_C, 1), dtype=complex)
        ket_i[i] = 1
        I_A = np.eye(dim_A)
        I_B = np.eye(dim_B)
        
        # 跟踪 C
        rho_AB += np.kron(I_A, np.kron(I_B, ket_i.conj().T)) @ rho_ABC @ np.kron(I_A, np.kron(I_B, ket_i))
    
    return rho_AB

def partial_trace_B(rho_ABC, dim_A, dim_B, dim_C):
    rho_AC = np.zeros((dim_A * dim_C, dim_A * dim_C), dtype=complex)
    for i in range(dim_B):
        ket_i = np.zeros((dim_B, 1), dtype=complex)
        ket_i[i] = 1
        I_A = np.eye(dim_A)
        I_C = np.eye(dim_C)
        
        # 跟踪 B
        rho_AC += np.kron(I_A, np.kron(ket_i.conj().T, I_C)) @ rho_ABC @ np.kron(I_A, np.kron(ket_i, I_C))
    
    return rho_AC

def partial_trace_A(rho_ABC, dim_A, dim_B, dim_C):
    rho_BC = np.zeros((dim_B * dim_C, dim_B * dim_C), dtype=complex)
    for i in range(dim_A):
        ket_i = np.zeros((dim_A, 1), dtype=complex)
        ket_i[i] = 1
        I_B = np.eye(dim_B)
        I_C = np.eye(dim_C)

        # 跟踪 A
        rho_BC += np.kron(ket_i.conj().T, np.kron(I_B, I_C)) @ rho_ABC @ np.kron(ket_i, np.kron(I_B, I_C))

    return rho_BC

# 计算 rho 的熵
def compute_entropy(rho):

    # 计算 rho 的特征值
    eigenvalues = eigvals(rho)

    # 计算 rho 的对数
    log_rho = logm(rho)

    # 计算 rho 的熵
    entropy = -np.trace(np.dot(rho, log_rho))

    return entropy

# 计算二方互信息
def I2(rho_AB, rho_A, rho_B):
    # I_2(S:M)=H(S)+H(M)-H(S,M)
    I2_A_B = compute_entropy(rho_A) + compute_entropy(rho_B) - compute_entropy(rho_AB)

    return I2_A_B.real

# 计算三方互信息
def I3(rho_ABC, rho_AB, rho_AC, rho_BC,
       rho_A, rho_B, rho_C):
    
    I2_A_B = compute_entropy(rho_A) + compute_entropy(rho_B) - compute_entropy(rho_AB)

    I2_A_C = compute_entropy(rho_A) + compute_entropy(rho_C) - compute_entropy(rho_AC)

    I2_A_BC = compute_entropy(rho_A) + compute_entropy(rho_BC) - compute_entropy(rho_ABC)

    # I_3(S:M_1:M_2)=I_2(S:M_1)+I_2(S:M_2)-I_2(S:M)

    I3_A_B_C = I2_A_B + I2_A_C - I2_A_BC

    return I3_A_B_C.real