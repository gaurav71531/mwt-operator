import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

import numpy as np
import math
import os
import h5py

from functools import partial
from models.utils import train, test, LpLoss, get_filter, UnitGaussianNormalizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_initializer(name):
    
    if name == 'xavier_normal':
        init_ = partial(nn.init.xavier_normal_)
    elif name == 'kaiming_uniform':
        init_ = partial(nn.init.kaiming_uniform_)
    elif name == 'kaiming_normal':
        init_ = partial(nn.init.kaiming_normal_)
    return init_

class sparseKernel(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel,self).__init__()
        
        self.k = k
        self.conv = self.convBlock(alpha*k**2, alpha*k**2)
        self.Lo = nn.Conv1d(alpha*k**2, c*k**2, 1)
        
    def forward(self, x):
        B, c, ich, Nx, Ny, T = x.shape # (B, c, ich, Nx, Ny, T)
        x = x.reshape(B, -1, Nx, Ny, T)
        x = self.conv(x)
        x = self.Lo(x.view(B, c*ich, -1)).view(B, c, ich, Nx, Ny, T)
        
        return x
        
        
    def convBlock(self, ich, och):
        net = nn.Sequential(
            nn.Conv3d(och, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net 
    

def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


class sparseKernelFT(nn.Module):
    def __init__(self,
                 k, alpha1, alpha2, alpha3, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernelFT, self).__init__()        
        
        self.modes1 = alpha1
        self.modes2 = alpha2
        self.modes3 = alpha3

        self.weights1 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes1, self.modes2, self.modes3, 2))
        self.weights2 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes1, self.modes2, self.modes3, 2))        
        self.weights3 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes1, self.modes2, self.modes3, 2))        
        self.weights4 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes1, self.modes2, self.modes3, 2))        
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)
        nn.init.xavier_normal_(self.weights3)
        nn.init.xavier_normal_(self.weights4)
        
        self.Lo = nn.Conv1d(c*k**2, c*k**2, 1)
        self.k = k
        
    def forward(self, x):
        B, c, ich, Nx, Ny, T = x.shape # (B, c, ich, N, N, T)
        
        x = x.reshape(B, -1, Nx, Ny, T)
        x_fft = torch.rfft(x, 3, normalized=True, onesided=True)
        
        # Multiply relevant Fourier modes
        l1 = min(self.modes1, Nx//2+1)
        l2 = min(self.modes2, Ny//2+1)
        out_ft = torch.zeros(B, c*ich, Nx, Ny, T//2 +1, 2, device=x.device)
        
        out_ft[:, :, :l1, :l2, :self.modes3] = compl_mul3d(
            x_fft[:, :, :l1, :l2, :self.modes3], self.weights1[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, :l2, :self.modes3] = compl_mul3d(
                x_fft[:, :, -l1:, :l2, :self.modes3], self.weights2[:, :, :l1, :l2, :])
        out_ft[:, :, :l1, -l2:, :self.modes3] = compl_mul3d(
                x_fft[:, :, :l1, -l2:, :self.modes3], self.weights3[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, -l2:, :self.modes3] = compl_mul3d(
                x_fft[:, :, -l1:, -l2:, :self.modes3], self.weights4[:, :, :l1, :l2, :])
        
        #Return to physical space
        x = torch.irfft(out_ft, 3, normalized=True, onesided=True, signal_sizes=(Nx, Ny, T))
        
        x = F.relu(x)
        x = self.Lo(x.view(B, c*ich, -1)).view(B, c, ich, Nx, Ny, T)
        return x
        
    
class MWT_CZ(nn.Module):
    def __init__(self,
                 k = 3, alpha = 5, 
                 L = 0, c = 1,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT_CZ, self).__init__()
        
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0@PHI0
        G0r = G0@PHI0
        H1r = H1@PHI1
        G1r = G1@PHI1
        
        H0r[np.abs(H0r)<1e-8]=0
        H1r[np.abs(H1r)<1e-8]=0
        G0r[np.abs(G0r)<1e-8]=0
        G1r[np.abs(G1r)<1e-8]=0
        
        self.A = sparseKernelFT(k, alpha, alpha, 10, c)
        self.B = sparseKernelFT(k, alpha, alpha, 10, c)
        self.C = sparseKernelFT(k, alpha, alpha, 10, c)
        
        self.T0 = nn.Conv1d(c*k**2, c*k**2, 1)

        if initializer is not None:
            self.reset_parameters(initializer)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((np.kron(H0, H0).T, 
                            np.kron(H0, H1).T,
                            np.kron(H1, H0).T,
                            np.kron(H1, H1).T,
                           ), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((np.kron(G0, G0).T,
                            np.kron(G0, G1).T,
                            np.kron(G1, G0).T,
                            np.kron(G1, G1).T,
                           ), axis=0)))
        
        self.register_buffer('rc_ee', torch.Tensor(
            np.concatenate((np.kron(H0r, H0r), 
                            np.kron(G0r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_eo', torch.Tensor(
            np.concatenate((np.kron(H0r, H1r), 
                            np.kron(G0r, G1r),
                           ), axis=0)))
        self.register_buffer('rc_oe', torch.Tensor(
            np.concatenate((np.kron(H1r, H0r), 
                            np.kron(G1r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_oo', torch.Tensor(
            np.concatenate((np.kron(H1r, H1r), 
                            np.kron(G1r, G1r),
                           ), axis=0)))
        
        
    def forward(self, x):
        
        B, c, ich, Nx, Ny, T = x.shape # (B, c, k^2, Nx, Ny, T)
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

#         decompose
        for i in range(ns-self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x.reshape(B, c*ich, -1)).view(
            B, c, ich, 2**self.L, 2**self.L, T) # coarsest scale transform

#        reconstruct            
        for i in range(ns-1-self.L,-1,-1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), 2)
            x = self.evenOdd(x)

        return x

    
    def wavelet_transform(self, x):
        xa = torch.cat([x[:, :, :, ::2 , ::2 , :], 
                        x[:, :, :, ::2 , 1::2, :], 
                        x[:, :, :, 1::2, ::2 , :], 
                        x[:, :, :, 1::2, 1::2, :]
                       ], 2)
        waveFil = partial(torch.einsum, 'bcixyt,io->bcoxyt') 
        d = waveFil(xa, self.ec_d)
        s = waveFil(xa, self.ec_s)
        return d, s
        
        
    def evenOdd(self, x):
        
        B, c, ich, Nx, Ny, T = x.shape # (B, c, 2*k^2, Nx, Ny)
        assert ich == 2*self.k**2
        evOd = partial(torch.einsum, 'bcixyt,io->bcoxyt')
        x_ee = evOd(x, self.rc_ee)
        x_eo = evOd(x, self.rc_eo)
        x_oe = evOd(x, self.rc_oe)
        x_oo = evOd(x, self.rc_oo)
        
        x = torch.zeros(B, c, self.k**2, Nx*2, Ny*2, T,
            device = x.device)
        x[:, :, :, ::2 , ::2 , :] = x_ee
        x[:, :, :, ::2 , 1::2, :] = x_eo
        x[:, :, :, 1::2, ::2 , :] = x_oe
        x[:, :, :, 1::2, 1::2, :] = x_oo
        return x
    
    def reset_parameters(self, initializer):
        initializer(self.T0.weight)
    
    
class MWT(nn.Module):
    def __init__(self,
                 ich = 1, k = 3, alpha = 2, c = 1,
                 nCZ = 3,
                 L = 0,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT,self).__init__()
        
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk = nn.Linear(ich, c*k**2)
        
        self.MWT_CZ = nn.ModuleList(
            [MWT_CZ(k, alpha, L, c, base, 
            initializer) for _ in range(nCZ)]
        )
        self.BN = nn.ModuleList(
            [nn.BatchNorm3d(c*k**2) for _ in range(nCZ)]
        )
        self.Lc0 = nn.Linear(c*k**2, 128)
        self.Lc1 = nn.Linear(128, 1)
        
        if initializer is not None:
            self.reset_parameters(initializer)
        
    def forward(self, x):
        
        B, Nx, Ny, T, ich = x.shape # (B, Nx, Ny, T, d)
        ns = math.floor(np.log2(Nx))
        x = self.Lk(x)
        x = x.view(B, Nx, Ny, T, self.c, self.k**2)
        x = x.permute(0, 4, 5, 1, 2, 3)
    
        for i in range(self.nCZ):
            x = self.MWT_CZ[i](x)
            x = self.BN[i](x.view(B, -1, Nx, Ny, T)).view(
                B, self.c, self.k**2, Nx, Ny, T)
            if i < self.nCZ-1:
                x = F.relu(x)

        x = x.view(B, -1, Nx, Ny, T) # collapse c and k**2
        x = x.permute(0, 2, 3, 4, 1)
        x = self.Lc0(x)
        x = F.relu(x)
        x = self.Lc1(x)
        return x.squeeze()
    
    def reset_parameters(self, initializer):
        initializer(self.Lc0.weight)
        initializer(self.Lc1.weight)
        
        
def load_data():
    
    data_path = 'Data/NS/ns_V1e-4_N10000_T30.mat'

    ntest = 200
    ntrain = 10000-ntest
    
    sub = 1
    S = 64 // sub
    T_in = 10
    T = 20

    dataloader = h5py.File(data_path)
    u_data = dataloader['u']
    t_data = dataloader['u']

    train_a = torch.from_numpy(u_data[:T_in, ::sub,::sub,:ntrain]
                ).permute(3, 1, 2, 0)
    train_u = torch.from_numpy(u_data[T_in:T_in+T, ::sub,::sub,:ntrain]
                ).permute(3, 1, 2, 0)

    test_a = torch.from_numpy(u_data[:T_in, ::sub,::sub,-ntest:]
                ).permute(3, 1, 2, 0)
    test_u = torch.from_numpy(u_data[T_in:T_in+T, ::sub,::sub,-ntest:]
                ).permute(3, 1, 2, 0)

    print('data loading complete')
    assert (S == train_u.shape[-2])
    assert (T == train_u.shape[-1])
    
    a_normalizer = UnitGaussianNormalizer(train_a)
#     x_train = a_normalizer.encode(train_a)
    x_test = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
#     y_train = y_normalizer.encode(train_u)

#     x_train = x_train.reshape(ntrain,S,S,1,T_in).repeat([1,1,1,T,1])
    x_test = x_test.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])
    
    # pad locations (x,y,t)
    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

#     x_train = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
#                             gridt.repeat([ntrain,1,1,1,1]), x_train), dim=-1)
    x_test = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                           gridt.repeat([ntest,1,1,1,1]), x_test), dim=-1)
    
    return x_test, test_u, y_normalizer
 

def main():
    x_test, y_test, y_normalizer = load_data()
    
    batch_size = 10
#     train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    
#     Legendre
    model = torch.load('ptmodels/NS_v_1e-4_N9800_T30_alpha_12_c_4_k_3_nCZ_4_L_0_3CNN_BN_epoch_200.pt')
    model.to(device)

    l2_test = test(model, test_loader, device, lossFn=myloss, post_proc=y_normalizer.decode)
    print(f'test relative L2 error for N=10000, T=30, nu=1e-4 with Legendre = {l2_test}')
    
#     Chebyshev
    model = torch.load('ptmodels/NS_v_1e-4_N9800_T20_alpha_12_c_4_k_3_nCZ_4_L_0_3CNN_BN_Chb_epoch_200.pt')
    model.to(device)
    
    l2_test = test(model, test_loader, device, lossFn=myloss, post_proc=y_normalizer.decode)
    print(f'test relative L2 error for N=10000, T=30, nu=1e-4 with Chebyshev = {l2_test}')
    
        
if __name__ == '__main__':
    main()

