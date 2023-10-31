import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt5Agg')

import pykoopman as pk

from pykoopman.common import vbe
from burgers_solver import burgers_solver

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

def visualize_data(dt, x, t, X):
    plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=Axes3D.name)
    for i in range(X.shape[0]):
        ax.plot(x, X[i], zs=t[i], zdir="t", label="time = " + str(i * dt))
    # plt.legend(loc='best')
    ax.view_init(elev=35.0, azim=-65, vertical_axis="y")
    ax.set(ylabel=r"$u(x,t)$", xlabel=r"$x$", zlabel=r"time $t$")
    plt.title("1D Viscous Burgers equation (Kutz et al., Complexity, 2018)")
    plt.show()

def visualize_state_space(X):
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    plt.figure(figsize=(6, 6))
    plt.semilogy(s)
    plt.xlabel("number of SVD terms")
    plt.ylabel("singular values")
    plt.title("PCA singular value decays")
    plt.show()

    # this is a pde problem so the number of snapshots are smaller than dof
    pca_1, pca_2, pca_3 = u[:, 0], u[:, 1], u[:, 2]
    plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=Axes3D.name)
    ax.plot3D(pca_1, pca_2, pca_3, "k-o")
    ax.set(xlabel="pc1", ylabel="pc2", zlabel="pc3")
    plt.title("PCA visualization")
    plt.show()

nx = 256
a = -15
b = 15
L = b-a
x = np.linspace(a, b, nx, endpoint=False)
u0 = np.exp(-(x+2)**2)
# u0 = 2.0 / np.cosh(x)
# u0 = u0.reshape(-1,1)
n_int = 3000
n_snapshot = 30
dt = 30. / n_int
n_sample = n_int // n_snapshot
nu = 0.1


X, t = vbe_simulate(u0, n_int, n_sample, nu, dt, nx, L)
delta_t = t[1]-t[0]

visualize_data(dt, x, t, X)
visualize_state_space(X)

from pydmd import DMD

dmd = DMD(svd_rank=5)
model = pk.Koopman(regressor=dmd)
model.fit(X, dt=delta_t)

K = model.A
U = model.ur

# Let's have a look at the eigenvalues of the Koopman matrix
evals, evecs = np.linalg.eig(K)
evals_cont = np.log(evals)/delta_t

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.plot(evals_cont.real, evals_cont.imag, 'bo', label='estimated',markersize=5)
plt.show()


# ax.set_xlim([-0.1,1])
# ax.set_ylim([2,3])
plt.legend()
plt.xlabel(r'$Re(\lambda)$')
plt.ylabel(r'$Im(\lambda)$')
# print(omega1,omega2)

def plot_pde_dynamics(x, t, X, X_pred, title_list, ymin=0, ymax=1):

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(131, projection='3d')
    for i in range(X.shape[0]):
        if X.dtype != 'complex':
            ax.plot(x, X[i], zs=t[i], zdir='t')
        else:
            ax.plot(x, abs(X[i]), zs=t[i], zdir='t')
    ax.set_ylim([ymin, ymax])
    ax.view_init(elev=35., azim=-65, vertical_axis='y')
    if X.dtype != 'complex':
        ax.set(ylabel=r'$u(x,t)$', xlabel=r'$x$', zlabel=r'time $t$')
    else:
        ax.set(ylabel=r'mag. of $u(x,t)$', xlabel=r'$x$', zlabel=r'time $t$')

    plt.title(title_list[0])

    ax = fig.add_subplot(132, projection='3d')
    for i in range(X.shape[0]):
        if X.dtype != 'complex':
            ax.plot(x, X_pred[i], zs=t[i], zdir='t')
        else:
            ax.plot(x, abs(X_pred[i]), zs=t[i], zdir='t')
    ax.set_ylim([ymin, ymax])
    ax.view_init(elev=35., azim=-65, vertical_axis='y')
    if X.dtype != 'complex':
        ax.set(ylabel=r'$u(x,t)$', xlabel=r'$x$', zlabel=r'time $t$')
    else:
        ax.set(ylabel=r'mag. of $u(x,t)$', xlabel=r'$x$', zlabel=r'time $t$')
    plt.title(title_list[1])

    ax = fig.add_subplot(133, projection='3d')
    for i in range(X.shape[0]):
        if X.dtype != 'complex':
            ax.plot(x, X_pred[i]-X[i], zs=t[i], zdir='t')
        else:
            ax.plot(x, abs(X_pred[i]-X[i]), zs=t[i], zdir='t')
    ax.set_ylim([ymin, ymax])
    ax.view_init(elev=35., azim=-65, vertical_axis='y')
    if X.dtype != 'complex':
        ax.set(ylabel=r'$u(x,t)$', xlabel=r'$x$', zlabel=r'time $t$')
    else:
        ax.set(ylabel=r'mag. of $u(x,t)$', xlabel=r'$x$', zlabel=r'time $t$')
    plt.title(title_list[2])

    plt.show()

def model_simulate(x, K, U, n_steps):
    def adjoint(M):
        return np.transpose(np.conjugate(M))
    
    U_adj = adjoint(U)

    observable_transition_matrix = U @ K @ U_adj
    
    y = np.empty((n_steps, np.size(U, axis=0)), dtype=K.dtype)
    y[0,:] = observable_transition_matrix @ x
    for i in range(1, n_steps):
        y[i,:] = observable_transition_matrix @ y[i-1,:]

    return y

X_predicted = np.vstack((X[0], model_simulate(X[0], K, U, n_steps=X.shape[0] - 1)))

plot_pde_dynamics(x,t,X, X_predicted, ['Truth','DMD-rank:'+str(model.A.shape[0]),'Residual'])