import torch

eps = 1e-10
def pairwise_distances(x):
    bn = x.shape[0]
    x = x.view(bn,-1)
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def calculate_gram_mat(x, sigma):
    dist= pairwise_distances(x)
    return torch.exp(-dist /sigma)

def reyi_entropy(x,sigma):
    alpha = 1.01
    k1 = calculate_gram_mat(x,sigma)
    k = k1/(torch.trace(k1)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy

