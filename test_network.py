import numpy as np

import numpy as np

from matplotlib import pyplot as plt

def init_random_vec(d):
    a_vec = ((np.random.rand(d) * 2) - 1) * np.pi
    c_vec = np.exp(a_vec*1j)
    return c_vec

def init_random_mat(d1, d2):
    a_vec = ((np.random.rand(d1, d2) * 2) - 1) * np.pi
    c_vec = np.exp(a_vec*1j)
    return c_vec


def s_opp(mat1, mat2):
    d = mat1.shape[0]
    return np.matmul(mat1, np.conj(mat2))/d

def similarity(vec1, vec2):
    d = vec1.shape[0]
    return np.real(np.dot(vec1, np.conj(vec2))/d)

def bind(vec1, vec2):
    return vec1 * vec2

def inverse(vec):
    return normalize(np.conj(vec)) / np.absolute(vec)

def unbind(vec1, vec2):
    return bind(vec1, inverse(vec2))

def bundle(vec1, *args):
    new_vec = vec1
    for vec in args:
        new_vec = new_vec + vec
    #new_vec = new_vec / (len(args)+1)
    return new_vec
    
def normalize(vec):
    a_vec = np.angle(vec)
    return np.exp(a_vec*1j)

def frac_power(c_vec, x):
    a_vec = np.angle(c_vec)
    a_vec = a_vec * x
    c_vec = np.exp(a_vec*1j)
    return c_vec