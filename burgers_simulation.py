
from burgers_solver import burgers_solver

from jax import numpy as jnp
from math import pi
import matplotlib.pyplot as plt

# Example usage
if __name__ == "__main__":
    # Define problem parameters
    num_points = 100
    domain_length = 2 * pi
    dx = domain_length / (num_points - 1)
    nu = 0.1
    dt = 0.001  # Time step size
    num_steps = 3141

    # Create the initial velocity profile (e.g., a sinusoidal profile)
    x = jnp.linspace(0, domain_length - dx, num_points)
    u_initial = jnp.sin(x)

    solver_partial = lambda u0 : burgers_solver(u0, nu, dx, num_points, dt, num_steps)
    
    # Solve the Burgers' equation
    u_final = solver_partial(u_initial)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Velocity Profile")
    plt.plot(x, u_final, label=f"Velocity Profile after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (u)")
    plt.legend()
    plt.savefig("burgers-simulation.png")