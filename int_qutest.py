import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate


functions = []
results = []
def debug(x):
    print(x)
    return 0
def f(*x):
    if len(x) == 1:
        return np.sqrt(2 - 2*np.cos(x[0]))
    else:
        return np.sqrt(1 + (f(*x[1:])**2) - 2*f(*x[1:])*np.cos(x[0]))

for i in range(10):
    # if len(functions) == 0:
    #     functions.append(lambda x: debug(0) + np.sqrt(2 - 2*np.cos(x)))
    # else:
    #     f_last = functions[i-1]
    #     f_new = lambda *x: debug(x) + np.sqrt(1 + (f_last(*x[1:])**2) - 2*f_last(*x[1:])*np.cos(x[0]))
    #     functions.append(f_new)
    #     #functions.append(lambda *x: np.sqrt(1 + (functions[i-1](*x[1:])**2) - 2*functions[i-1](*x[1:])*np.cos(x[0])))
    ranges = [[0, 2*np.pi] for _ in range(i+1)]
    test_input = [0 for _ in range(i+1)]
    #print(functions[-1](*test_input))
    result, error = integrate.nquad(lambda *x: f(*x), ranges)
    results.append(result/((2*np.pi)**(i+1)))
    print(results[-1])

plt.plot(results)
plt.show()