"""Constraints for the optimization."""

import numpy as np
import jax
import jax.numpy as jnp

import mesher
import sdf_ops
import projections

_mesh = mesher.Mesher


def _compute_overlapping_volume_frac(densities: jnp.ndarray,
                                     mesh: _mesh, 
                                     scale = 10.)->float:
  """
  Args:
    densities: Array of (num_objects, num_elems) that is the density in [0,1]
      for each object as for each element on a mesh.
    mesh: A dataclass of `Mesher` which has the `elem_area` and other info.
    scale: A scaling factor to weight the overlapping constraint.
  Returns: A float indicating the total volume of overlap between the primitives.
  """
  dom_ratio = mesh.elem_area/mesh.domain_volume
  return scale*dom_ratio*jnp.sum(jax.nn.relu(jnp.sum(densities, axis=0) - 1.))


def compute_min_separation_constraint(sdfs: jnp.ndarray,
                                      mesh: _mesh,
                                      min_separation: float,
                                      threshold = 1e-2)->float:
  """
  Args:
    sdf: Array of (num_objects, num_elems) that have the SDF value for each of
      the objects. We assume the SDFs have been scaled by half the min sepearation
      value
    min_separation: A value that indicates how much each object must be
      separated from one another.
    mesh: A dataclass of `Mesher` that contains information about the underlying
      domain onto which the `sdf` is projected.
    threshold: A small positive value added to let the constraints go
      inactive (ie < 0). 
  """
  inflated_sdf = sdfs - 0.5*min_separation
  densities = sdf_ops.project_sdf_to_density(inflated_sdf, mesh)
  densities = projections.threshold_filter(densities)
  return _compute_overlapping_volume_frac(densities, mesh) - threshold


def latent_distance_constraint(predicted_latent_coordn: jnp.ndarray,
                               part_latent_coordn: jnp.ndarray,
                               max_allowed_latent_dist: float,
                               order: float = 200.
                               )->jnp.ndarray:
  """Ensures that the latent coordinate predicted is close to a
      part in the latent space.

      min(dist_predicted_actual_latent_coordns) < max_allowed_dist
    
    Where we use a smooth maximum defined by the logsumexp form.
  Args:
    predicted_latent_coordn: Tensor of size (num_des_parts, num_latent_dim) that is the
      latent coordinate of the predicted material.
    part_latent_coordn: Tensor of size (num_avail_parts, num_latent_dim) that are
      the latent coordinates of the available parts.
  """

  # d -> num_des_parts, a -> num_avail_parts, l -> num_latent_dim
  delta_des_avail_latent = (predicted_latent_coordn[:, np.newaxis, :] - 
                            part_latent_coordn[np.newaxis, :, :])#{dal}
  
  dist_des_avail_latent = jnp.linalg.norm(delta_des_avail_latent, axis=-1) #{da}

  dist_nograd = jax.lax.stop_gradient(dist_des_avail_latent)
  scale = np.amax(np.abs(dist_nograd))/order


  min_dist_des_parts_from_avail = -scale*jax.scipy.special.logsumexp(
                                        -dist_des_avail_latent/scale, axis=-1)

  max_min_dist = scale*jax.scipy.special.logsumexp(
                                          min_dist_des_parts_from_avail/scale)

  return max_min_dist - max_allowed_latent_dist