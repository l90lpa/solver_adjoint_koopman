
from jax import lax, jit
from jax import numpy as jnp
from functools import partial


def jax_scan(kernel, 
             initial_condition, 
             num_steps):
    
    final_condition, _ = lax.scan(kernel, initial_condition, None, num_steps)

    return final_condition


def integrate(dynamics, 
              ic, 
              dt, 
              nt, 
              solver_step, 
              *, 
              scan_function=jax_scan):
    
    def solver_step_wrapped(y, _):
        return y + dt * solver_step(dynamics, y, dt), None
    
    return scan_function(solver_step_wrapped, ic, nt)


def RK4_step(f, y, dt):

    k1 = f(y)
    k2 = f(y + dt * 0.5 * k1)
    k3 = f(y + dt * 0.5 * k2)
    k4 = f(y + dt * k3)

    return (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def RK4(f, y0, dt, num_steps):

    y = integrate(f, y0, dt, num_steps, RK4_step)
    
    return y


def create_diffusion_opeartor(dx):

    def diffusion_op(u):
        dxdx = dx ** 2
        u_new = (jnp.roll(u, -1) - 2.0 * u + jnp.roll(u, 1)) / dxdx
        return u_new
    
    return diffusion_op


def create_advection_opeartor(dx):

    def advection_op(u):
        dx2 = dx * 2
        u_new = (jnp.roll(u, -1) - jnp.roll(u, 1)) / dx2
        return u_new
    
    return advection_op


@partial(jit, static_argnames=['nx', 'nt'])
def burgers_solver(u0, nu, dx, nx, dt, nt):

    A = create_advection_opeartor(dx)
    D = create_diffusion_opeartor(dx)

    def burgers_dynamics(u):
        return - u * A(u) + nu * D(u)

    return RK4(burgers_dynamics, u0, dt, nt)
    





