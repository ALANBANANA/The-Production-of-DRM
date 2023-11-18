import torch
import matplotlib.pyplot as plt
import numpy as np
import time



class RitzNet(torch.nn.Module):
    def __init__(self):
        super(RitzNet, self).__init__()
        self.linear_input = torch.nn.Linear(10, 10)
        self.linear = torch.nn.ModuleList()
        for block in range(4):
            self.linear.append(torch.nn.Linear(10, 10))
            self.linear.append(torch.nn.Linear(10, 10))
        self.linear_ouput = torch.nn.Linear(10, 1)
        """激活函数"""
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear_input(x))
        for block2 in range(4):
            y = self.activation(self.linear[block2*2](x))
            x = x + self.activation(self.linear[block2*2+1](y))
        return self.linear_ouput(x)


ritz_net = RitzNet()
if torch.cuda.is_available():
    ritz_net = ritz_net.cuda()


def get_interior_points(num=1000, dim=10):
    X_interior = torch.rand(num, dim) * 2 - 1
    if torch.cuda.is_available():
        X_interior = X_interior.cuda()
    return X_interior.requires_grad_(True)


X_interior = get_interior_points(num=1000, dim=10)


def get_boundary_points(N=100):
    X_boundary = torch.rand(2 * 10 * N, 10)
    for i in range(10):
        X_boundary[2 * i * N: (2 * i + 1) * N, i] = 0
        X_boundary[(2 * i + 1) * N: (2 * i + 2) * N, i] = 1
        if torch.cuda.is_available():
            X_boundary = X_boundary.cuda()
    return X_boundary.requires_grad_(True)


X_boundary = get_boundary_points()


def grad(u, x):
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               retain_graph=True, create_graph=True, only_inputs=True)


x1 = X_interior[:, 0].cuda()
x2 = X_interior[:, 1].cuda()
x3 = X_interior[:, 2].cuda()
x4 = X_interior[:, 3].cuda()
x5 = X_interior[:, 4].cuda()
x6 = X_interior[:, 5].cuda()
x7 = X_interior[:, 6].cuda()
x8 = X_interior[:, 7].cuda()
x9 = X_interior[:, 8].cuda()
x10 = X_interior[:, 9].cuda()
def right_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    func = -160*torch.pi**2*(torch.sin(4*torch.pi*x1)*torch.sin(4*torch.pi*x2)*
                             torch.sin(4*torch.pi*x3)*torch.sin(4*torch.pi*x4)*
                             torch.sin(4*torch.pi*x5)*torch.sin(4*torch.pi*x6)*
                             torch.sin(4*torch.pi*x7)*torch.sin(4*torch.pi*x8)*
                             torch.sin(4*torch.pi*x9)*torch.sin(4*torch.pi*x10))
    if torch.cuda.is_available():
        func = func.cuda()
    return func


def loss(beta=500):
    u_pred = ritz_net(X_interior)
    grad_u_pred = grad(u_pred, X_interior)
    grad_u_pred = torch.tensor([item.cpu().detach().numpy() for item in grad_u_pred]).cuda()
    loss_i = torch.mean(0.5*torch.sum(grad_u_pred*grad_u_pred, axis=-1) -
                        (right_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)*u_pred[:, 0]))
    loss_b = torch.mean(ritz_net(X_boundary)[:, 0]**2)
    total_loss = loss_b*beta + loss_i
    return loss_i, loss_b, total_loss


print(torch.cuda.is_available())
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
model = ritz_net
model = model.to(device)
torch.load('test_parameters_10dim.pth')
with torch.no_grad():
    Z = torch.rand(100000, 10)
    Z = Z.to(device)
    pred = model(Z)

print(Z)