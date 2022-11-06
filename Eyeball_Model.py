
from turtle import forward
from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn import *
import torch
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

device = torch.device("cuda:0") # all of this works in GPU as well :)
def plot_traj(trajectory,t_span):
    color=['cyan','orange', 'blue']

    fig = plt.figure(figsize=(10,2))
    ax0 = fig.add_subplot(121)
    #ax1 = fig.add_subplot(122)
    for i in range(trajectory.shape[0]):
        ax0.plot(t_span, trajectory[i,0,:], color=color[0], alpha=.1);
        #ax1.plot(t_span, trajectory[:,i,1], color=color[int(yn[i])], alpha=.1);
    ax0.set_xlabel(r"$t$ [Depth]") ; ax0.set_ylabel(r"$h_0(t)$")
    #ax1.set_xlabel(r"$t$ [Depth]") ; ax1.set_ylabel(r"$z_1(t)$")
    ax0.set_title("Dimension 0") #; ax1.set_title("Dimension 1")
class Plant_DT(nn.Module):
    def __init__(self, t_span, initial=0, gamma=0.1, omega=0.11, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.omega = omega
        self.O = (omega**2 - gamma**2)
        self.t_span = t_span
        self.restoration = torch.cos(self.O * self.t_span).to(device)
        self.damping = gamma/self.O * torch.sin(self.O * self.t_span).to(device)
        self.decay = torch.exp(-gamma * self.t_span).to(device)
        self.x0 = initial

    def forward(self, commands):
        xf = commands/(self.omega**2)
        rv = xf + (self.x0 - xf) * self.decay * (self.restoration + self.damping)
        return rv

#X_train = nn.Softplus()(torch.Tensor(X).to(device))
#y_train = torch.LongTensor(yn.long()).to(device)
neuron_count = 4
X_train = torch.rand((1028,1)).to(device)
yn = (X_train - 0.5) * 2

t_span = torch.linspace(0,90,100).to(device)
eye = Plant_DT(t_span).to(device)
y_train = []
for y in yn:
    commands = torch.ones((1,t_span.shape[0])).to(device) * y * eye.omega**2
    #positions = eye(commands)
    #print(positions)
    y_train.append(commands)

y_train = torch.stack(y_train).to(device)
plot_traj(y_train.detach().cpu(),t_span.detach().cpu())
plt.show()
print(y_train.shape)
train = data.TensorDataset(yn, y_train)
trainloader = data.DataLoader(train, batch_size=len(yn), shuffle=True)


class Learner(pl.LightningModule):
    def __init__(self, t_span:torch.Tensor, brain:nn.Module, body:nn.Module):
        super().__init__()
        self.brain, self.body, self.t_span = brain.to(device), body.to(device), t_span
        #self.neural_readout = nn.Linear(neuron_count,1).to(device)
        self.params = list(self.brain.parameters()) #+ list(self.neural_readout.parameters()) + list(self.input_readin.parameters())

    def forward(self, x):
        #torch.cat((x,torch.zeros()))
        t_eval, commands = self.brain(x, t_span)
        commands = commands.reshape((commands.shape[1],commands.shape[2],commands.shape[0]))
        #commands = self.neural_readout(commands)
        #commands = commands.reshape((commands.shape[1],commands.shape[2],commands.shape[0]))
        #print(commands)
        #position = self.body(commands) # select last point of solution trajectory
        return commands

    def training_step(self, batch, batch_idx):
        x, y = batch
        #x.pad()
        t_eval, commands = self.brain(x, t_span)
        #commands = self.neural_readout(commands)
        commands = commands.reshape((commands.shape[1],commands.shape[2],commands.shape[0]))
        #print(commands.shape)
        #position = self.body(commands)[:,:,:] # select last point of solution trajectory
        #print(position.shape)
        #print(y.shape)
        loss = nn.MSELoss()(commands, y)
        
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.params, lr=0.01)

    def train_dataloader(self):
        return trainloader



#brain = NeuralODE(f, sensitivity='adjoint', solver='dopri5').to(device)
#t_span = torch.linspace(0,70,100).to(device)
#eye = Plant_DT(t_span).to(device)
"""for y in yn:
    commands = torch.ones((1,t_span.shape[0])).to(device) * y * eye.omega**2
    res = eye(commands)
    #print(y)
    #print(res[:,-1])
    trajectory = res[0,:].detach().cpu()
    plt.plot(trajectory)
plt.show()
#exit()"""
    

class CTRNN_DT(nn.Module):
    """Continuous-time RNN.

    Parameters:
        hidden_size: Number of hidden neurons

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()
        
    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_size, 1).to(device))
        self.taus = torch.nn.Parameter(torch.Tensor(hidden_size, 1).to(device))
        self.non_linearity = torch.nn.Sigmoid()

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

    def size(self):
        return self.hidden_size


f = CTRNN_DT(neuron_count)
f = nn.Sequential(
        nn.Linear(1, 20),
        nn.Softplus(),
        nn.Linear(20, 20),
        nn.Softplus(),
        nn.Linear(20,1)
    ).to(device)

brain = NeuralODE(f, sensitivity='adjoint', solver='dopri5').to(device)
eye = Plant_DT(t_span).to(device)
learn = Learner(t_span, body=eye, brain=brain)
trainer = pl.Trainer(min_epochs=200, max_epochs=300,devices=1, accelerator="gpu")
trainer.fit(learn)

#chk_path = "/home/denpak/Research/NeuralODEModeling/lightning_logs/version_22/checkpoints/epoch=299-step=3300.ckpt"
#model2 = brain.load_from_checkpoint(chk_path)
#results = trainer.test(model=model2, datamodule=my_datamodule, verbose=True)

trajectory = learn.to(device)(yn)
trajectory = trajectory.detach().cpu()
print(trajectory)
plot_traj(trajectory, t_span.detach().cpu())
plt.show()

exit()


print(X_train.shape)
t_eval, trajectory = brain(X_train, t_span)
trajectory = trajectory.detach().cpu()
print(trajectory.shape)
trajectory = eye(trajectory.T)
print(trajectory.squeeze().shape)
plot_traj(trajectory.squeeze(), t_span)
plt.show()
exit()
