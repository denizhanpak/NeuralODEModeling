import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

t = np.linspace(0,15,1000)
omega_sq = 0.11 ** 2
gamma = 0.1

y = [0,2] #y[0]=x and y[1]=v

def harmonic(t,y):
    solution = [y[1],-omega_sq*y[0]]
    return solution
    
sho = solve_ivp(harmonic, [0,1000], y0 = y, t_eval = t)

plt.plot(t,sho.y[0])
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('SHO', fontsize = 20)


t = np.linspace(0,200,1000)
y = [0,1]
gamma = 0.1
omega_sqr = 0.11**2

def sho(t,y):
    solution = (y[1],(-gamma*y[1]-omega_sqr*y[0]))
    return solution

solution = solve_ivp(sho, [0,1000], y0 = y, t_eval = t)
plt.plot(t,solution.y[0])
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('Damped Oscillator', fontsize = 20)
plt.show()

def vf(x0, v0, gamma=0.1, omega=0.11,t=t):
    O = np.sqrt(omega**2-gamma**2)
    term1 = x0 * np.cos(O*t)
    term2 = v0/omega * np.sin(O*t)
    return np.exp(-gamma*t)*(term1 + term2)

y = vf(0.1,0.1)
plt.plot(t,y)
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('Damped Oscillator', fontsize = 20)
plt.show()

def vf_step(x0,xf,ti=10,gamma=0.1,omega=0.21):
    O = np.sqrt(omega**2-gamma**2)
    t = np.linspace(0,100,10000)
    dt = (t-ti).clip(0,None)
    term1 = np.cos(O*(dt))
    term2 = gamma/O * np.sin(O*(dt))
    rv = xf + (x0-xf) * np.exp(-gamma * (dt)) * (term1+term2)
    return rv, t


x0 = -0.9
xf = -2.3
y,t = vf_step(x0,xf) + vf(x0,0)
plt.plot(t,y)
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('Forced Oscillator', fontsize = 20)
plt.show()