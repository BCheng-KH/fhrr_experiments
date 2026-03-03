import numpy as np

from matplotlib import pyplot as plt

def init_random_vec(d):
    a_vec = ((np.random.rand(d) * 2) - 1) * np.pi
    c_vec = np.exp(a_vec*1j)
    return c_vec


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


def binary_search(vec, x_vec, p, mid=0, step=1, found=False):
    
    low_x = mid-step
    mid_x = mid
    high_x = mid+step
    low_s = similarity(vec, frac_power(x_vec, low_x))
    mid_s = similarity(vec, frac_power(x_vec, mid_x))
    high_s = similarity(vec, frac_power(x_vec, high_x))
    if not found:
        if mid_s > low_s and mid_s > high_s:
            if p > 0:
                return binary_search(vec, x_vec, p-1, mid_x, step/2, True)
            else:
                return mid_x, mid_s
        elif low_s > high_s:
            return binary_search(vec, x_vec, p, low_x, step, False)
        else:
            return binary_search(vec, x_vec, p, high_x, step, False)
    else:
        if mid_s > low_s and mid_s > high_s:
            if p > 0:
                return binary_search(vec, x_vec, p-1, mid_x, step/2, True)
            else:
                return mid_x, mid_s
        elif low_s > high_s:
            if p > 0:
                return binary_search(vec, x_vec, p-1, low_x, step/2, True)
            else:
                return low_x, low_s
        else:
            if p > 0:
                return binary_search(vec, x_vec, p-1, high_x, step/2, True)
            else:
                return high_x, high_s



def grid_search(vec, x_vec, values):
    x_high = None
    s_high = None
    for v in values:
        s = similarity(vec, frac_power(x_vec, v))
        if x_high == None:
            x_high = v
            s_high = s
        else:
            if s > s_high:
                x_high = v
                s_high = s
    return x_high, s_high


def test_bundle(f, bundle, x_vec, y_vec, test_values):
    error = 0
    f_guess = []
    for i, x in enumerate(test_values):
        bundle_slice = unbind(bundle, frac_power(x_vec, x))
        grid_values = list(np.arange(-(round(abs(f(x)*2))+1), (round(abs(f(x)*2))+1), 0.1))
        x_guess, s = grid_search(bundle_slice, y_vec, grid_values)
        x_guess, s = binary_search(bundle_slice, y_vec, 10, mid=x_guess, step=0.5)
        f_guess.append(x_guess)
        if i == 0:
            error = (f(x) - x_guess)**2
        else:
            error = (i/(i+1))*error + (1/(i+1))*((f(x) - x_guess)**2)
    return error, f_guess




dim = 1000

x_vec = init_random_vec(dim)
y_vec = init_random_vec(dim)

def f(x):
    return np.sin(x) + np.sin(2*x+1)

x_interval = [0, 10]

sample_interval_1 = 0.1
sample_interval_2 = 0.05

sample_1 = [bind(frac_power(x_vec, x), frac_power(y_vec, f(x))) for x in np.arange(*x_interval, sample_interval_1)]
bundle_1 = normalize(bundle(*sample_1))

sample_2 = [bind(frac_power(x_vec, x), frac_power(y_vec, f(x))) for x in np.arange(*x_interval, sample_interval_2)]
bundle_2 = normalize(bundle(*sample_2))

print(similarity(bundle_1, bundle_2))


test_interval = 0.05
test_values = np.arange(*x_interval, test_interval)
true_sample = [f(x) for x in test_values]

bundle_error_1, bundle_sample_1 = test_bundle(f, bundle_1, x_vec, y_vec, test_values)
bundle_error_2, bundle_sample_2 = test_bundle(f, bundle_2, x_vec, y_vec, test_values)

print(bundle_error_1, bundle_error_2)

plt.plot(test_values, true_sample)
plt.show()
plt.plot(test_values, bundle_sample_1)
plt.show()
plt.plot(test_values, bundle_sample_2)
plt.show()


