import torch

num_dim_x = 4
num_dim_control = 2

def f_func(x_local):
    # x_local: bs x n x 1 (local coordinates)
    # f: bs x n x 1
    bs = x_local.shape[0]

    dx, dy, theta, v = [x_local[:,i,0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x_local.type())
    f[:, 0, 0] = v * torch.cos(theta)  # dx/dt
    f[:, 1, 0] = v * torch.sin(theta)  # dy/dt
    f[:, 2, 0] = 0  # dtheta/dt (will be controlled)
    f[:, 3, 0] = 0  # dv/dt (will be controlled)
    return f

def DfDx_func(x_local):
    raise NotImplementedError('NotImplemented')

def B_func(x_local):
    # x_local: bs x n x 1 (local coordinates)
    # B: bs x n x m
    bs = x_local.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x_local.type())

    # The control inputs affect angular velocity and linear acceleration
    B[:, 2, 0] = 1  # Angular velocity control
    B[:, 3, 1] = 1  # Linear acceleration control
    return B

def DBDx_func(x_local):
    raise NotImplementedError('NotImplemented')
