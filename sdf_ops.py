"""Boolean and projection operations on signed distance fields."""

import jax
import jax.numpy as jnp

import mesher

_Mesher = mesher.Mesher


def project_sdf_to_density(sdf: jnp.ndarray,
                           mesh: _Mesher,
                           order = 50.)-> jnp.ndarray:
  """Projects primitive onto a mesh, given primitive parameters and mesh coords.

  The resulting density field has a value of one when an element intersects
  with a primitive and zero when it lies outside the mesh.

  Args:
    sdf: Array that is the signed distance value for each object as for each 
      element on a mesh. 
    sharpness: The sharpness value controls the slope of the sigmoid function.
      While a larger value makes the transition more sharper, it makes it more
      non-linear.
    order: The sigmoid entries are scaled to roughly [-order, order]. This
      is done to prevent the gradients from dying for large magnitudes of
      the entries.
  Returns:
    density: Array where the values are in range [0, 1] where 0 means the mesh 
      element did not intersect with the primitive and 1 means it intersected.
  """
  # the sigmoid function has dying gradients for large values of argument.
  # to avoid this we scale it to the order of `order`. Note that simply scaling
  # doesn't shift the 0 isosurface and hence doesn't mess up or calculations.

  scale = order/mesh.bounding_box.lx  # assume lx, ly are in same order
  return jax.nn.sigmoid(-sdf*scale)


def compute_union_density_fields(density: jnp.ndarray,
                                 penal: float = 6.,
                                 x_min: float = 1e-3) -> jnp.ndarray:
  """Differentiable max function to compute union of densities.

  Computes the maximum value of array along specified axis.
  The smooth max scheme is set in the constructor of the class

  Args:
    density: Array of size (num_objects, num_elems) which contain the density of
      each object on the mesh.
    penal: Used in the computation of a penalty based smooth max function,
      the value indicates the p-th norm to take. A larger value while making
      the value closer to the true max value also makes the problem more
      nonlinear.
    x_min: To avoid numerical issues in stiffness matrices used in simulation,
      a small lower bound value > 0 is added to the density.

  Returns: Array of size (num_elems,) which contain the density of the object
  """
  dx = x_min**penal
  return jnp.clip(
                  (jnp.sum(density**penal, axis=0))**(1./penal),
         a_min=dx, a_max=1.)


def compute_circ_sdf(xy: jnp.ndarray,
                     center_x: float = 0.,
                     center_y: float = 0.,
                     radius: float = 0.,
                     n: float = 2.):
  """Unit circle is centered at origin and unit radius.
     By default, computes the SDF of a point centered at origin.

  Args:
    xy: Array of (num_pts, 2) on which we want to compute the SDF.
    center_x:
    center_y:
    radius:
    n: Degree of the circle. By default it is 2. A higher value indicates a
      squircle.
  Returns: Array of (num_pts,) of the SDF values
  """
  return jnp.power((xy[:,0] - center_x)**n +
                   (xy[:,1] - center_y)**n,
                   1./n) - radius
