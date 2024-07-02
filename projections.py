import jax.numpy as jnp
import numpy as np


def threshold_filter(rho: jnp.ndarray, 
                     beta: float = 8., 
                     c0: float = 0.5):
  """
  Apply a threshold filter to the input array using a hyperbolic tangent function.
  
  Args:
    rho: Array of densities.
    beta: Sharpness parameter of the threshold filter.
    c0: Threshold center.

  Returns:
    jnp.ndarray: The filtered array.
  """
  v1 = np.tanh(c0*beta)
  nm = v1 + jnp.tanh(beta*(rho- c0))
  dnm = v1 + jnp.tanh(beta*(1.- c0))
  return nm/dnm