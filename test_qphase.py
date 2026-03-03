import numpy as np
from matplotlib import pyplot as plt

def init_random_vec(d):
    a_vec = ((np.random.rand(d) * 2) - 1) * np.pi
    c_vec = np.exp(a_vec*1j)
    return c_vec
amp_list = []
for x in range(100):
    overall_amp = 0
    for case in range(1000):
        base_vec = init_random_vec(1)
        influence_vec = init_random_vec(1)*0
        for _ in range(x):
            influence_vec += init_random_vec(1)
        overall_vec = base_vec + influence_vec
        overall_amp += (np.absolute(overall_vec)[0] **2)/(x+1)
    overall_amp /= 1000
    amp_list.append(overall_amp)

plt.plot(amp_list)
plt.show()