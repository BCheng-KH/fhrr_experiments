import numpy as np
import tqdm
import torch
torch.set_default_dtype(torch.double)

SPT = np.sqrt(np.pi)/2

def general_normalize(mat):
    a_mat = torch.angle(mat)
    return torch.exp(a_mat*1j)

def init_random_mat(shape):
    return torch.exp(((torch.rand(shape) * 2) - 1) * torch.pi*1j)

class fhrr_space:
    def __init__(self, dims):
        self.dims = dims
        self.ones = torch.exp(torch.zeros(size=(self.dims, 1))*1j)
        self.zeros = torch.zeros_like(self.ones)
    def init_ones_vec(self):
        return self.ones.clone()
    def init_zeros_vec(self):
        return self.zeros.clone()
    def init_random_vec(self):
        a_vec = ((torch.rand(self.dims, 1) * 2) - 1) * torch.pi
        c_vec = torch.exp(a_vec*1j)
        return c_vec
    def init_normal_avec(self, std=torch.pi):
        return torch.randn(self.dims, 1)*std
    def a_to_c(self, vec):
        c_vec = torch.exp(vec*1j)
        return c_vec
    def c_to_a(self, vec):
        a_vec = torch.angle(vec)
        return a_vec
    def normalize(self, vec):
        a_vec = torch.angle(vec)
        return torch.exp(a_vec*1j)
    def similarity_R(self, vec1, vec2):
        return torch.real((vec1.conj().T @ vec2)/self.dims)
    def similarity_C(self, vec1, vec2):
        return ((vec1.conj().T @ vec2)/self.dims)
    def norm_similarity_R(self, vec1, vec2):
        return torch.real((self.normalize(vec1).conj().T @ self.normalize(vec2))/self.dims)
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
    def conj(self, vec):
        return torch.conj(vec)
    def inverse(self, vec):
        return self.normalize(torch.conj(vec)) / torch.absolute(vec)
    def norm_inverse(self, vec):
        return self.normalize(torch.conj(vec))
    def unbind(self, vec1, vec2):
        return self.bind(vec1, self.inverse(vec2))
    def norm_unbind(self, vec1, vec2):
        return self.bind(vec1, self.norm_inverse(vec2))
    def frac_power(self, vec, x):
        a_vec = torch.angle(vec)
        a_vec = a_vec * x
        c_vec = torch.exp(a_vec*1j)
        return c_vec
    def ssp_phase(self, a_vec, x):
        return torch.exp((a_vec*x)*1j)
    def get_pdf(self, d_vec, e_vecs):
        p_unnormalized = torch.abs(self.similarity_R(d_vec, e_vecs))
        return p_unnormalized/torch.sum(p_unnormalized, axis=1, keepdims=True)
    def get_pdf2(self, d_vec, e_vecs):
        p_unnormalized = self.similarity_C(d_vec, e_vecs)
        p_unnormalized = p_unnormalized * torch.conj(p_unnormalized)
        return p_unnormalized/torch.sum(p_unnormalized)


def make_orthogonal_map(space1, space2, passes = 3, verbose=False):
    M = torch.exp((((torch.rand(space2.dims, space1.dims) * 2) - 1) * torch.pi)*1j)
    if verbose:
        total_steps = ((space1.dims-1)*passes*(space1.dims-2))//2
        pbar = tqdm.tqdm(total = total_steps)
    for i in range(1, space1.dims):
        i_vec = M[:, i:i+1].clone()
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
    MTM_inverse = torch.linalg.inv(MTM)
    M_inverse = MTM_inverse @ M.conj().T
    return general_normalize(M_inverse)

class fhrr_map:
    def __init__(self, space1, space2, maps=None, orthogonal = False):
        self.spaces = [space1, space2]
        self.dims = (space1.dims, space2.dims)
        if maps == None:
            if orthogonal == False:
                self.F_map = torch.exp((((torch.rand(self.dims[1], self.dims[0]) * 2) - 1) * torch.pi)*1j)
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
        return fhrr_map(self.spaces[0], self.spaces[1], [self.F_map.clone(), self.B_map.clone()])
    


class learning_map:
    def __init__(self, space1, space2, learn_rate = 0.001, decay_rate = 0.001, maps=None):
        self.space1, self.space2 = space1, space2
        self.dims = (space1.dims, space2.dims)
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        if maps == None:
            self.F_map = torch.exp((((torch.rand(self.dims[1], self.dims[0]) * 2) - 1) * torch.pi)*1j)*learn_rate/10
            self.B_map = torch.exp((((torch.rand(self.dims[0], self.dims[1]) * 2) - 1) * torch.pi)*1j)*learn_rate/10
        else:
            self.F_map = maps[0]
            self.B_map = maps[1]
    def learn(self, vec1, vec2):
        f_map = vec2@self.space1.conj(vec1).T
        b_map = vec1@self.space2.conj(vec2).T
        self.F_map = self.F_map*(1-self.decay_rate) + f_map*self.learn_rate
        self.B_map = self.B_map*(1-self.decay_rate) + b_map*self.learn_rate
    def forwards(self, vec, normalize = True):
        if normalize:
            return self.space2.normalize(self.F_map@vec)
        else:
            return self.F_map@vec
    def backwards(self, vec, normalize = True):
        if normalize:
            return self.space1.normalize(self.B_map@vec)
        else:
            return self.B_map@vec
    def directed_pass(self, vec, direction = 0, normalize = True):
        if direction == 0:
            return self.forwards(vec, normalize = normalize)
        else:
            return self.backwards(vec, normalize = normalize)

def test_map(map, space1, space2, noise_level):
    target = space1.init_random_vec()
    encoded = map.forwards(target)
    noise = space2.init_zeros_vec()
    for _ in range(noise_level):
        noise = space2.bundle(noise, space2.bind(map.forwards(space1.init_random_vec()), map.norm_forwards(space1.init_random_vec())))
    bun = space2.bundle(encoded, noise)
    retrieved = map.norm_backwards(bun)
    result = torch.abs(space1.norm_similarity_C(target, retrieved))
    angle = torch.angle(space1.norm_similarity_C(target, retrieved))
    return result, angle


class block_learning_map:
    def __init__(self, space1, space2, num_blocks, block_shape, learn_rate = 0.0001, decay_rate = 0.0001, maps=None):
        assert(num_blocks * block_shape[0] == space1.dims)
        assert(num_blocks * block_shape[1] == space2.dims)
        self.space1, self.space2 = space1, space2
        self.dims = (space1.dims, space2.dims)
        self.num_blocks = num_blocks
        self.block_shape = block_shape
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        if maps == None:
            self.F_maps = torch.exp((((torch.rand(self.num_blocks, self.block_shape[1], self.block_shape[0]) * 2) - 1) * torch.pi)*1j)*learn_rate/10
            self.B_maps = torch.exp((((torch.rand(self.num_blocks, self.block_shape[0], self.block_shape[1]) * 2) - 1) * torch.pi)*1j)*learn_rate/10
        else:
            self.F_maps = maps[0]
            self.B_maps = maps[1]
    def learn(self, vec1, vec2):
        block_vec1 = vec1.reshape((self.num_blocks, self.block_shape[0], -1))
        block_vec2 = vec2.reshape((self.num_blocks, self.block_shape[1], -1))
        f_maps = torch.einsum("ijk,ilk->ijl", block_vec2, self.space1.conj(block_vec1))
        b_maps = torch.einsum("ijk,ilk->ijl", block_vec1, self.space2.conj(block_vec2))
        self.F_maps = self.F_maps*(1-self.decay_rate) + f_maps*self.learn_rate
        self.B_maps = self.B_maps*(1-self.decay_rate) + b_maps*self.learn_rate
        
    def forwards(self, vec, normalize = True):
        block_vec = vec.reshape((self.num_blocks, self.block_shape[0], -1))
        out_vec = torch.einsum("ijl,ilm->ijm", self.F_maps, block_vec).reshape((self.dims[1], -1))
        if normalize:
            return self.space2.normalize(out_vec)
        else:
            return out_vec
    def backwards(self, vec, normalize = True):
        block_vec = vec.reshape((self.num_blocks, self.block_shape[1], -1))
        out_vec = torch.einsum("ijl,ilm->ijm", self.B_maps, block_vec).reshape((self.dims[0], -1))
        if normalize:
            return self.space1.normalize(out_vec)
        else:
            return out_vec
    def directed_pass(self, vec, direction = 0, normalize = True):
        if direction == 0:
            return self.forwards(vec, normalize = normalize)
        else:
            return self.backwards(vec, normalize = normalize)

class diag_learning_map:
    def __init__(self, space1, space2, learn_rate = 0.001, decay_rate = 0.001, maps=None):
        assert(space1.dims == space2.dims)
        self.space1, self.space2 = space1, space2
        self.dims = (space1.dims, space2.dims)
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        if maps == None:
            self.F_map = space1.init_random_vec()*learn_rate/10
            self.B_map = space1.init_random_vec()*learn_rate/10
        else:
            self.F_map = maps[0]
            self.B_map = maps[1]
    def learn(self, vec1, vec2):
        f_map = self.space1.bind(vec2, self.space1.conj(vec1))
        b_map = self.space1.bind(vec1, self.space1.conj(vec2))
        self.F_map = self.F_map*(1-self.decay_rate) + f_map*self.learn_rate
        self.B_map = self.B_map*(1-self.decay_rate) + b_map*self.learn_rate
    def forwards(self, vec, normalize = True):
        if normalize:
            return self.space2.normalize(self.space1.bind(self.F_map, vec))
        else:
            return self.space1.bind(self.F_map, vec)
    def backwards(self, vec, normalize = True):
        if normalize:
            return self.space1.normalize(self.space1.bind(self.B_map, vec))
        else:
            return self.space1.bind(self.B_map, vec)
    def directed_pass(self, vec, direction = 0, normalize = True):
        if direction == 0:
            return self.forwards(vec, normalize = normalize)
        else:
            return self.backwards(vec, normalize = normalize)