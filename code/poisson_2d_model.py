import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from  mpl_toolkits.mplot3d import Axes3D
import matplotlib


class RitzNet(torch.nn.Module):
    def __init__(self):
        super(RitzNet, self).__init__()
        self.linear_input = torch.nn.Linear(2, 10)
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
num = 1000
num_boundary = 500
x1 = torch.rand(num, 1)*2-1  # 内点布置
x2 = torch.rand(num, 1)*2-1
"""边界点x2为1"""
x_b_1_positive = torch.rand(num_boundary, 1)*2-1
x_b_2_positive_ub = torch.ones_like(x_b_1_positive)
"""边界点x2为-1"""
x_b_1_negative = torch.rand(num_boundary, 1)*2-1
x_b_2_negative_lb = torch.ones_like(x_b_1_negative)*-1
"""边界x1为1"""
x_b_2_positive1 = torch.rand(num_boundary, 1)*2-1
x_b_1_positive1_ub = torch.ones_like(x_b_2_positive1)
"""边界x1为-1"""
x_b_2_negative1 = torch.rand(num_boundary, 1)*2-1
x_b_1_positive1_lb = torch.ones_like(x_b_2_negative1)*-1
def generate_boundary_x2_positive(x_b_1_positive, x_b_2_positive_ub):
    X2_boundary_positive = torch.cat([x_b_1_positive, x_b_2_positive_ub], dim=1)
    if torch.cuda.is_available():
        x_b_1_positive = x_b_1_positive.cuda()
        x_b_2_positive_ub = x_b_2_positive_ub.cuda()
        X2_boundary_positive = X2_boundary_positive.cuda()
    return X2_boundary_positive.requires_grad_(True), x_b_1_positive.requires_grad_(True),\
        x_b_2_positive_ub.requires_grad_(True)


def generate_boundary_x2_negative(x_b_1_negative, x_b_2_negative_lb):
    X2_boundary_negative = torch.cat([x_b_1_negative, x_b_2_negative_lb], dim=1)
    if torch.cuda.is_available():
        x_b_1_negative = x_b_1_negative.cuda()
        x_b_2_negative_lb = x_b_2_negative_lb.cuda()
        X2_boundary_negative = X2_boundary_negative.cuda()
    return X2_boundary_negative.requires_grad_(True), x_b_1_negative.requires_grad_(True),\
        x_b_2_negative_lb.requires_grad_(True)


def generate_boundary_x1_positive(x_b_1_positive1_ub, x_b_2_positive1):
    X1_boundary_positive = torch.cat([x_b_1_positive1_ub, x_b_2_positive1], dim=1)
    if torch.cuda.is_available():
        x_b_1_positive1_ub = x_b_1_positive1_ub.cuda()
        x_b_2_positive1 = x_b_2_positive1.cuda()
        X1_boundary_positive = X1_boundary_positive.cuda()
    return X1_boundary_positive.requires_grad_(True), x_b_1_positive1_ub.requires_grad_(True),\
        x_b_2_positive1.requires_grad_(True)


def generate_boundary_x1_negative(x_b_1_negative1_lb, x_b_2_negative1):
    X1_boundary_negative = torch.cat([x_b_1_negative1_lb, x_b_2_negative1], dim=1)
    if torch.cuda.is_available():
        x_b_1_negative1_lb = x_b_1_negative1_lb.cuda()
        x_b_2_negative1 = x_b_2_negative1.cuda()
        X1_boundary_negative = X1_boundary_negative.cuda()
    return X1_boundary_negative.requires_grad_(True), x_b_1_negative1_lb.requires_grad_(True),\
        x_b_2_negative1.requires_grad_(True)


def generate_data_interior(x1, x2):
    X = torch.cat([x1, x2], dim=1)
    if torch.cuda.is_available():
        x1 = x1.cuda()
        x2 = x2.cuda()
        X = X.cuda()
    return X.requires_grad_(True), x1.requires_grad_(True), x2.requires_grad_(True)


def right_function(x1, x2):
    X = torch.cat([x1, x2], dim=1)
    func = -32*torch.pi**2*torch.sin(4*torch.pi*X[:, 0])*torch.sin(4*torch.pi*X[:, 1])
    if torch.cuda.is_available():
        x1 = x1.cuda()
        x2 = x2.cuda()
        func = func.cuda()
    return func


def grad(u, x):
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               retain_graph=True, create_graph=True, only_inputs=True)


X, x1, x2 = generate_data_interior(x1, x2)
X2_boundary_positive, x_b_1_positive, x_b_2_positive_ub = generate_boundary_x2_positive(x_b_1_positive,
                                                                                        x_b_2_positive_ub)
X2_boundary_negative, x_b_1_negative, x_b_2_negative_lb = generate_boundary_x2_negative(x_b_1_negative,
                                                                                        x_b_2_negative_lb)
X1_boundary_positive, x_b_1_positive1_ub, x_b_2_positive1 = generate_boundary_x1_positive(x_b_1_positive1_ub,
                                                                                          x_b_2_positive1)
X1_boundary_negative, x_b_1_positive1_lb, x_b_2_negative1 = generate_boundary_x1_negative(x_b_1_positive1_lb,

                                                                                          x_b_2_negative1)
def loss(beta=500):
    u_pred = ritz_net(X)
    grad_u_pred = grad(u_pred, X)
    grad_u_pred = torch.tensor([item.cpu().detach().numpy() for item in grad_u_pred]).cuda()
    loss_i = torch.mean(0.5*torch.sum(grad_u_pred*grad_u_pred, axis=-1) - (right_function(x1, x2)*u_pred[:, 0]))
    loss_b = torch.mean(ritz_net(X2_boundary_positive)[:, 0]**2 + ritz_net(X2_boundary_negative)[:, 0]**2 +
                        ritz_net(X1_boundary_positive)[:, 0]**2 + ritz_net(X1_boundary_negative)[:, 0]**2)
    total_loss = loss_b*beta + loss_i
    return loss_i, loss_b, total_loss

print(torch.cuda.is_available())
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
model = ritz_net
model = model.to(device)
torch.load('test_parameters_2dim.pth')
with torch.no_grad():
    x1 = torch.linspace(-1, 1, 1001)
    x2 = torch.linspace(-1, 1, 1001)
    X, Y = torch.meshgrid(x1, x2)
    Z = torch.cat((X.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
    Z = Z.to(device)
    pred = model(Z)

plt.figure()
pred = pred.cpu().numpy()
pred = pred.reshape(1001, 1001)
ax = plt.subplot(1, 1, 1)
h = plt.imshow(pred, interpolation='nearest', cmap='rainbow', origin='lower',
               extent=[-1, 1, -1, 1],aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(h, cax=cax)
plt.show()

