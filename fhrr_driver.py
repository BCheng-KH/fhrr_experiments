import numpy as np
import tqdm


SPT = np.sqrt(np.pi)/2

def general_normalize(mat):
    a_mat = np.angle(mat)
    return np.exp(a_mat*1j)

class fhrr_space:
    def __init__(self, dims):
        self.dims = dims
        self.ones = np.exp(np.zeros(shape=(self.dims, 1))*1j)
        self.zeros = np.zeros_like(self.ones)
    def init_ones_vec(self):
        return self.ones.copy()
    def init_zeros_vec(self):
        return self.zeros.copy()
    def init_random_vec(self):
        a_vec = ((np.random.rand(self.dims, 1) * 2) - 1) * np.pi
        c_vec = np.exp(a_vec*1j)
        return c_vec
    def a_to_c(self, vec):
        c_vec = np.exp(vec*1j)
        return c_vec
    def c_to_a(self, vec):
        a_vec = np.angle(vec)
        return a_vec
    def normalize(self, vec):
        a_vec = np.angle(vec)
        return np.exp(a_vec*1j)
    def similarity_R(self, vec1, vec2):
        return np.real((vec1.conj().T @ vec2)/self.dims)
    def similarity_C(self, vec1, vec2):
        return ((vec1.conj().T @ vec2)/self.dims)
    def norm_similarity_R(self, vec1, vec2):
        return np.real((self.normalize(vec1).conj().T @ self.normalize(vec2))/self.dims)
    def norm_similarity_C(self, vec1, vec2):
        return ((self.normalize(vec1).conj().T @ self.normalize(vec2))/self.dims)
    def bind(self, *vecs):
        result = self.init_ones_vec()
        for vec in vecs:
            result = result * vec
        return result
    def bundle(self, *vecs):
        result = self.init_zeros_vec()
        for vec in vecs:
            result += vec
        return result
    def bind_and_bundle(self, vec_set):
        bundle_vecs = []
        for vecs in vec_set:
            bound_vec = self.bind(*vecs)
            bundle_vecs.append(bound_vec)
        result = self.bundle(*bundle_vecs)
        return result
    def inverse(self, vec):
        return self.normalize(np.conj(vec)) / np.absolute(vec)
    def norm_inverse(self, vec):
        return self.normalize(np.conj(vec))
    def unbind(self, vec1, vec2):
        return self.bind(vec1, self.inverse(vec2))
    def norm_unbind(self, vec1, vec2):
        return self.bind(vec1, self.norm_inverse(vec2))
    def frac_power(self, vec, x):
        a_vec = np.angle(vec)
        a_vec = a_vec * x
        c_vec = np.exp(a_vec*1j)
        return c_vec
    def get_pdf(self, d_vec, e_vecs):
        p_unnormalized = self.similarity_R(d_vec, e_vecs)
        return p_unnormalized/np.sum(p_unnormalized)


def make_orthogonal_map(space1, space2, passes = 3, verbose=False):
    M = np.exp((((np.random.rand(space2.dims, space1.dims) * 2) - 1) * np.pi)*1j)
    if verbose:
        total_steps = ((space1.dims-1)*passes*(space1.dims-2))//2
        pbar = tqdm.tqdm(total = total_steps)
    for i in range(1, space1.dims):
        i_vec = M[:, i:i+1].copy()
        for p in range(passes):
            #print(i, p, i_vec.shape)
            for j in range(1, i):
                sim = space2.norm_similarity_C(M[:, j:j+1], i_vec)
                #print(sim)
                i_vec = space2.normalize(i_vec - (M[:, j:j+1]*sim))
                #print(i, p, j, i_vec.shape)
                pbar.update(1)
        M[:, i:i+1] = i_vec
    pbar.close()
    return M

def make_inverse(M):
    MTM = M.conj().T @ M
    MTM_inverse = np.linalg.inv(MTM)
    M_inverse = MTM_inverse @ M.conj().T
    return general_normalize(M_inverse)

class fhrr_map:
    def __init__(self, space1, space2, maps=None, orthogonal = False):
        self.spaces = [space1, space2]
        self.dims = (space1.dims, space2.dims)
        if maps == None:
            if orthogonal == False:
                self.F_map = np.exp((((np.random.rand(self.dims[1], self.dims[0]) * 2) - 1) * np.pi)*1j)
                self.B_map = make_inverse(self.F_map)
            else:
                if self.dims[0] <= self.dims[1]:
                    self.F_map = make_orthogonal_map(space1, space2)
                    self.B_map = make_inverse(self.F_map)
                else:
                    self.B_map = make_orthogonal_map(space2, space1)
                    self.F_map = make_inverse(self.B_map)
        else:
            self.F_map = maps[0]
            self.B_map = maps[1]
        self.F_multiplier = SPT*np.sqrt(space1.dims)
        self.B_multiplier = SPT*np.sqrt(space2.dims)
    def directed_map(self, vec, direction=0, pre_norm = False, post_norm = False):
        D_map = self.F_map if direction == 0 else self.B_map
        D_multiplier = self.F_multiplier if direction == 0 else self.B_multiplier
        vec1 = vec if not pre_norm else self.spaces[direction].normalize(vec)
        vec2 = (D_map*D_multiplier)@vec1
        vec3 = vec2 if not post_norm else self.spaces[1-direction].normalize(vec2)
        return vec3
    def forwards(self, vec):
        return self.directed_map(vec, 0, False, False)
    def backwards(self, vec):
        return self.directed_map(vec, 1, False, False)
    def norm_forwards(self, vec):
        return self.directed_map(vec, 0, True, False)
    def norm_backwards(self, vec):
        return self.directed_map(vec, 1, True, False)
    def forwards_norm(self, vec):
        return self.directed_map(vec, 0, False, True)
    def backwards_norm(self, vec):
        return self.directed_map(vec, 1, False, True)
    def norm_forwards_norm(self, vec):
        return self.directed_map(vec, 0, True, True)
    def norm_backwards_norm(self, vec):
        return self.directed_map(vec, 1, True, True)
    def inverse(self):
        return fhrr_map(self.spaces[1], self.spaces[0], [self.B_map, self.F_map])
    def copy(self):
        return fhrr_map(self.spaces[0], self.spaces[1], [self.F_map.copy(), self.B_map.copy()])


def test_map(map, space1, space2, noise_level):
    target = space1.init_random_vec()
    encoded = map.forwards(target)
    noise = space2.init_zeros_vec()
    for _ in range(noise_level):
        noise = space2.bundle(noise, space2.bind(map.forwards(space1.init_random_vec()), map.norm_forwards(space1.init_random_vec())))
    bun = space2.bundle(encoded, noise)
    retrieved = map.norm_backwards(bun)
    result = np.abs(space1.norm_similarity_C(target, retrieved))
    angle = np.angle(space1.norm_similarity_C(target, retrieved))
    return result, angle