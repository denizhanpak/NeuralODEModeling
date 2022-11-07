import torch 
import matplotlib.pyplot as plt
from torch import nn

device = torch.device("cpu") # all of this works in GPU as well :)

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

vals = torch.linspace(-5,5,10)
t_span = torch.linspace(0,100,1000)
eye = Plant_DT(t_span,omega=0.21)
for val in vals:
    commands = torch.ones(len(t_span)) * (eye.omega**2) * val
    results = eye(commands)
    plt.plot(results[::10])
plt.ylabel("Position of Eye")
plt.xlabel("Time (ms)")
plt.show()
