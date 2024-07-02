"""Optimization functions."""

import jax
import jax.numpy as jnp
import numpy as np

import mesher
import transforms
import sdf_ops
import projections
import constraints



def unnormalize_latent_coordns(x: jnp.ndarray, 
                               low: jnp.ndarray,
                               high: jnp.ndarray)-> jnp.ndarray:
  """
  Unnormalize latent coordinates from [0, 1] to the specified range.

  Args:
    x: Array of (num_shapes, latent_dim) with values in [0,1]
    low: Array of (latent_dim,)
    high: Array of (latent_dim,)
  Returns: Array of (num_shape, latent_dim)
  """
  nwax = np.newaxis
  range = high - low
  return low[nwax, :] + range[nwax, :]*x


def normalize_latent_coordns(x: jnp.ndarray, 
                            low: jnp.ndarray,
                            high: jnp.ndarray)-> jnp.ndarray:
  """
  Normalize latent coordinates to the range [0, 1].

  Args:
    x: Array of (num_shapes, latent_dim) with values in [0,1]
    low: Array of (latent_dim,)
    high: Array of (latent_dim,)
  Returns: Array of (num_shape, latent_dim)
  """
  nwax = np.newaxis
  range = high - low
  return (x - low[nwax, :])/range[nwax, :]


def compute_transforms_and_latent_coordn_from_opt_params(
                                  opt_params: jnp.ndarray,
                                  num_stamps: int,
                                  transform_extent: transforms.TransformExtent,
                                  num_latent_params: int,
                                  latent_dim: int,
                                  min_encoded_coordn: float,
                                  max_encoded_coordn: float
                                  )->transforms.Transform:
  
  """
  Compute transforms and latent coordinates from optimization parameters.
  
  Args:
    opt_params: Optimization parameters array of shape (num_elems, 6).
    num_stamps: Number of stamps.
    transform_extent: Transformation extent object.
    num_latent_params: Number of latent parameters.
    latent_dim: Latent dimension.
    min_encoded_coordn: Minimum encoded coordinate.
    max_encoded_coordn: Maximum encoded coordinate.

  Returns:
    Tuple of predicted stamp latent coordinates and shape transforms.
  """

  norm_latent_params = opt_params[:num_latent_params].reshape((num_stamps, latent_dim))
  norm_shape_params = opt_params[num_latent_params:]
  
  pred_stamp_latent_coordns = unnormalize_latent_coordns(norm_latent_params,
                                                         min_encoded_coordn,
                                                         max_encoded_coordn)
  
  shape_transforms =  transforms.Transform.from_normalized_array(
                                                    norm_shape_params,
                                                    num_stamps,
                                                    transform_extent)

  return pred_stamp_latent_coordns, shape_transforms


def compute_shape_sdfs(sdf_net,
                       sdf_net_params,
                       dom_mesh: mesher.Mesher,
                       shape_transforms: transforms.Transform,
                       pred_stamp_latent_coordns:jnp.ndarray,
                       stamp_bbox: mesher.BoundingBox):
  """
  Compute signed distance fields (SDFs) for shape instances.
  
  Args:
    sdf_net: SDF network.
    sdf_net_params: SDF network parameters.
    dom_mesh: Mesh object containing mesh information.
    shape_transforms: Shape transforms.
    pred_stamp_latent_coordns: Predicted stamp latent coordinates.
    stamp_bbox: Bounding box for stamping.

  Returns:
    Array of shape SDFs.
  """
  
  num_objects = shape_transforms.num_objects
  trans_mesh_xy = transforms.transform_coordinates(dom_mesh.elem_centers,
                                                   shape_transforms)
  sdfs = jnp.zeros((num_objects, dom_mesh.num_elems))
  trans_xy_nograd = jax.lax.stop_gradient(trans_mesh_xy)

  # compute the stamped indices
  stamped_indices = mesher.compute_point_indices_in_box(
                                        trans_xy_nograd,
                                        stamp_bbox)
  unstamped_indices = jnp.logical_not(stamped_indices)

  # compute the SDF and density of each object
  for obj in range(num_objects):
    stamp_xy = trans_mesh_xy[obj, stamped_indices[obj,:],:]
    unstamp_xy = trans_mesh_xy[obj, unstamped_indices[obj,:],:]

    scale = shape_transforms.scale[obj]

    # compute the stamped sdf
    stamp_sdf = sdf_net.apply({'params': sdf_net_params},
                              pred_stamp_latent_coordns[obj,:].reshape((1, -1)),
                              stamp_xy,
                              method='call_decoder'
                              )*scale
    sdfs = sdfs.at[obj, stamped_indices[obj,:]].set(stamp_sdf.reshape(-1))

    # compute the unstamped approx sdf
    sdfs = sdfs.at[obj, unstamped_indices[obj,:]].set(
                sdf_ops.compute_circ_sdf(unstamp_xy)*scale)

  return sdfs



def compute_objective(opt_params: jnp.ndarray,
                      transform_extent: transforms.TransformExtent,
                      challenge,
                      dom_mesh: mesher.Mesher,
                      sdf_net,
                      sdf_net_params,
                      stamp_bbox,
                      dens_array,
                      num_stamps: int,
                      num_latent_params: int,
                      latent_dim: int,
                      min_encoded_coordn: float,
                      max_encoded_coordn: float
                      ):
  
  """
  Compute the constraints for optimization.
  
  Args:
    opt_params: Optimization parameters of shape (num_elems, 6).
    min_separation: Minimum separation constraint.
    transform_extent: Transformation extent object.
    dom_mesh: Mesh object containing mesh information.
    sdf_net: SDF network.
    sdf_net_params: SDF network parameters.
    stamp_bbox: Bounding box for stamping.
    num_latent_params: Number of latent parameters.
    latent_dim: Latent dimension.
    min_encoded_coordn: Minimum encoded coordinate.
    max_encoded_coordn: Maximum encoded coordinate.
    encoded_z: Encoded latent coordinates.
    num_stamps: Number of stamps.
    epoch: Current epoch of the optimization.
  
  Returns:
    Tuple of constraints and their gradients.
  """

  def objective_wrapper(opt_params)->float:
    (pred_stamp_latent_coordns, 
     shape_transforms) = compute_transforms_and_latent_coordn_from_opt_params(
                                                          opt_params,
                                                          num_stamps,
                                                          transform_extent,
                                                          num_latent_params,
                                                          latent_dim,
                                                          min_encoded_coordn,
                                                          max_encoded_coordn)
   
    shape_sdfs = compute_shape_sdfs(sdf_net,
                                  sdf_net_params,
                                  dom_mesh,
                                  shape_transforms,
                                  pred_stamp_latent_coordns,
                                  stamp_bbox)
    shape_densities = sdf_ops.project_sdf_to_density(shape_sdfs, dom_mesh)
    density = sdf_ops.compute_union_density_fields(shape_densities).reshape((
                                                                  dom_mesh.nelx,
                                                                  dom_mesh.nely))
    density = projections.threshold_filter(density)
    dens_array.array = density

    response, aux = challenge.component.response(dens_array)
    performance_loss = jnp.log(challenge.loss(response))
    distance = challenge.distance_to_target(response)
    metrics = challenge.metrics(response, dens_array, aux)

    return performance_loss, (shape_transforms, pred_stamp_latent_coordns, density,
                              response, distance, metrics, aux)
  
  (obj, auxs), grad_obj = jax.value_and_grad(objective_wrapper,
                                             has_aux=True)(opt_params.reshape(-1))

  return jnp.array([obj]), grad_obj.reshape((-1, 1)), auxs


def compute_constraint(opt_params: jnp.ndarray,
                       min_separation: float,
                       transform_extent: transforms.TransformExtent,
                       dom_mesh: mesher.Mesher,
                       sdf_net,
                       sdf_net_params,
                       stamp_bbox,
                       num_latent_params,
                       latent_dim,
                       min_encoded_coordn,
                       max_encoded_coordn,
                       encoded_z,
                       num_stamps: int,
                       epoch: int):
  
  """
  Compute signed distance fields (SDFs) for shapes.
  
  Args:
    sdf_net: SDF network.
    sdf_net_params: SDF network parameters.
    dom_mesh: Mesh object containing mesh information.
    shape_transforms: Shape transforms.
    pred_stamp_latent_coordns: Predicted stamp latent coordinates.
    stamp_bbox: Bounding box for stamping.
  
  Returns:
    Array of shape SDFs.
  """
  def sep_constraint_wrapper(opt_params: jnp.ndarray)->float:
    (pred_stamp_latent_coordns, 
     shape_transforms) = compute_transforms_and_latent_coordn_from_opt_params(
                                                          opt_params,
                                                          num_stamps,
                                                          transform_extent,
                                                          num_latent_params,
                                                          latent_dim,
                                                          min_encoded_coordn,
                                                          max_encoded_coordn)

    shape_sdfs = compute_shape_sdfs(sdf_net,
                                   sdf_net_params,
                                   dom_mesh,
                                   shape_transforms,
                                   pred_stamp_latent_coordns,
                                   stamp_bbox)
    min_sep_cons = constraints.compute_min_separation_constraint(shape_sdfs,
                                                                dom_mesh,
                                                              min_separation)

    return min_sep_cons

  sep_cons, grad_sep_cons = jax.value_and_grad(sep_constraint_wrapper)(opt_params.reshape(-1))

  def latent_cons_wrapper(opt_params):
    (pred_stamp_latent_coordns, 
     shape_transforms) = compute_transforms_and_latent_coordn_from_opt_params(
                                                          opt_params,
                                                          num_stamps,
                                                          transform_extent,
                                                          num_latent_params,
                                                          latent_dim,
                                                          min_encoded_coordn,
                                                          max_encoded_coordn)
    d = max(0.02, 4. - 0.08*epoch) # HARDCODING
    latent_space_cons = constraints.latent_distance_constraint(
                                                pred_stamp_latent_coordns,
                                                encoded_z,
                                                max_allowed_latent_dist=d) 
    return latent_space_cons
  lat_cons, grad_lat_cons = jax.value_and_grad(latent_cons_wrapper)(opt_params.reshape(-1))

  cons = jnp.array([sep_cons, lat_cons]).reshape((-1, 1))
  grad_cons = jnp.stack((grad_sep_cons, grad_lat_cons), axis=-1)
  return cons, grad_cons.T