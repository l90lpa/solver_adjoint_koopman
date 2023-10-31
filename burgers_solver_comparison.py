
from functools import partial
import numpy as np
from pykoopman.common import vbe
from burgers_solver import burgers_solver
from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def vbe_simulate(u0, n_int, n_sample, nu, dt, nx, L):
    # n_traj = x0.shape[1]
    dx = L / nx
    u = u0
    U = np.zeros((n_int // n_sample, nx), dtype=np.float64)
    t = 0
    j = 0
    t_list = []
    for step in range(n_int):
        t += dt
        u = burgers_solver(u, nu, dx, nx, dt, 1)
        if (step + 1) % n_sample == 0:
            U[j, :] = u
            j += 1
            t_list.append(t)
    return U, np.array(t_list)

nu = 0.1
nx = 256
a = -15
b = 15
L = b - a
x = np.linspace(a, b, nx, endpoint=False, dtype=np.float64)
u0 = np.array(np.exp(-(x+2)**2), dtype=np.float64)
# u0 = 2.0 / np.cosh(x)
# u0 = u0.reshape(-1,1)
n_int = 3000
n_snapshot = 30
dt = 30. / n_int
n_sample = n_int // n_snapshot


U, T = vbe_simulate(u0, n_int, n_sample, nu, dt, nx, L)


model_vbe = vbe(nx, x, nu=nu, dt=dt, L=L)
X, t = model_vbe.simulate(u0, n_int, n_sample)

assert np.shape(U) == np.shape(X)

diff = np.abs(U - X)

print(np.max(diff, axis=0))

plt.plot(x, U[-1,:], label="My Solver")
plt.plot(x, X[-1,:], label="Their Solver")
plt.xlabel("Position (x)")
plt.ylabel("Velocity (u)")
plt.legend()
plt.show()