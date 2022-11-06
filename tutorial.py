from turtle import forward
from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn import *
import torch
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
import pytorch_lightning as pl

device = torch.device("cuda:0") # all of this works in GPU as well :)

d = ToyDataset()
X, yn = d.generate(n_samples=512, noise=1e-1, dataset_type='moons')
print(X.shape)


#Plotting data
colors = ['orange', 'blue']
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
for i in range(len(X)):
    ax.scatter(X[i,0], X[i,1], s=1, color=colors[yn[i].int()])
plt.show()


X_train = nn.Softplus()(torch.Tensor(X).to(device))
print(X_train.shape)
exit()
y_train = torch.LongTensor(yn.long()).to(device)
train = data.TensorDataset(X_train, y_train)
trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)

class Learner(pl.LightningModule):
    def __init__(self, t_span:torch.Tensor, model:nn.Module):
        super().__init__()
        self.model, self.t_span = model, t_span

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x, t_span)
        y_hat = y_hat[-1] # select last point of solution trajectory
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return trainloader

f = nn.Sequential(
        nn.Linear(2, 4),
        nn.Softplus(),
        nn.Linear(4, 2)
    )





class CTRNN_DT(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms. 
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()
        
    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, hidden_size, tau=100, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.tau =  tau

        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_size, 1))
        self.taus = torch.Tensor(hidden_size, 1).to(device)
        self.non_linearity = torch.nn.Sigmoid()

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def forward(self, state):
        """Run network for one time step.
        
        Inputs:
            State: tensor of shape (batch, input_size)
        
        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        s = state.add(self.bias.T)
        output = self.non_linearity(s)
        incoming = self.h2h(output)
        new = incoming - state
        return new#torch.div(new, self.taus.T)

    def out_size(self):
        return self.hidden_size

f = CTRNN_DT(5)

model = NeuralODE(f, sensitivity='adjoint', solver='dopri5').to(device)

def plot_traj(trajectory,t_span):
    color=['orange', 'blue']

    fig = plt.figure(figsize=(10,2))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    for i in range(trajectory.shape[1]):
        ax0.plot(t_span, trajectory[:,i,0], color=color[int(yn[i])], alpha=.1);
        ax1.plot(t_span, trajectory[:,i,1], color=color[int(yn[i])], alpha=.1);
    ax0.set_xlabel(r"$t$ [Depth]") ; ax0.set_ylabel(r"$h_0(t)$")
    ax1.set_xlabel(r"$t$ [Depth]") ; ax1.set_ylabel(r"$z_1(t)$")
    ax0.set_title("Dimension 0") ; ax1.set_title("Dimension 1")

t_span = torch.linspace(0,5,300)
print(X_train.shape)
t_eval, trajectory = model(X_train, t_span)
trajectory = trajectory.detach().cpu()
plot_traj(trajectory, t_span)
plt.show()

learn = Learner(t_span, model)
trainer = pl.Trainer(min_epochs=200, max_epochs=300,devices=1, accelerator="gpu")
trainer.fit(learn)

t_eval, trajectory = model(X_train, t_span)
trajectory = trajectory.detach().cpu()
plot_traj(trajectory, t_span)
plt.show()

n_pts = 50
x = torch.linspace(trajectory[:,:,0].min(), trajectory[:,:,0].max(), n_pts)
y = torch.linspace(trajectory[:,:,1].min(), trajectory[:,:,1].max(), n_pts)
X, Y = torch.meshgrid(x, y) ; z = torch.cat([X.reshape(-1,1), Y.reshape(-1,1)], 1)
f = model.vf(0,z.to(device)).cpu().detach()
fx, fy = f[:,0], f[:,1] ; fx, fy = fx.reshape(n_pts , n_pts), fy.reshape(n_pts, n_pts)
# plot vector field and its intensity
fig = plt.figure(figsize=(4, 4)) ; ax = fig.add_subplot(111)
ax.streamplot(X.numpy().T, Y.numpy().T, fx.numpy().T, fy.numpy().T, color='black')
ax.contourf(X.T, Y.T, torch.sqrt(fx.T**2+fy.T**2), cmap='RdYlBu')
plt.show()
