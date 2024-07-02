"""2D transformation of the objects.

author: Aaditya Chandrasekhar (cs.aaditya@gmail.com)
Rahul K Padhy rkpadhy@wisc.edu
"""

import dataclasses
import numpy as np
import jax.numpy as jnp
import utils

_Ext = utils.Extent


@dataclasses.dataclass
class TransformExtent:
  """Define extents of the transforms.
  """
  trans_x: _Ext
  trans_y: _Ext
  rot_rad: _Ext
  scale: _Ext


@dataclasses.dataclass
class Transform:
  """
  Attributes:
    trans_x: Tensor of (num_objects,)
  """
  trans_x: jnp.ndarray
  trans_y: jnp.ndarray
  rot_rad: jnp.ndarray
  scale: jnp.ndarray


  @property
  def num_objects(self)->int:
    return self.trans_x.shape[0]


  @property
  def num_transform_params_per_obj(self)->int:
    return 4


  @property
  def num_transform_parameters(self)->int:
    return self.num_transform_params_per_obj * self.num_objects


  @property
  def rot_matrix(self)->jnp.ndarray:
    """Array of (num_objects, 2, 2) that contains the rotation matrix for
      all the objects."""
    s = jnp.sin(self.rot_rad)
    c = jnp.cos(self.rot_rad)

    rot_mtrx = jnp.zeros((self.num_objects, 2, 2))  #{o,2,2}

    rot_mtrx = rot_mtrx.at[:, 0, 0].set(c)
    rot_mtrx = rot_mtrx.at[:, 0, 1].set(-s)
    rot_mtrx = rot_mtrx.at[:, 1, 0].set(s)
    rot_mtrx = rot_mtrx.at[:, 1, 1].set(c)

    return rot_mtrx


  @classmethod
  def from_array(
      cls,
      state_array: jnp.ndarray,
      num_objects: int,
  ) -> 'Transform':
    """Converts a rank-1 array into `Transform`."""
    num_t = num_objects
    tx = state_array[0:num_t]
    ty = state_array[num_t:2*num_t]
  
    r = state_array[2*num_t:3*num_t]
    s = state_array[3*num_t:4*num_t]

    return Transform(tx, ty, r, s)


  def to_array(self) -> jnp.ndarray:
    """Converts the `Transform3D` into a rank-1 array."""
    return jnp.concatenate([f.reshape((-1)) for f in dataclasses.astuple(self)])


  def to_normalized_array(self, trans_extents: TransformExtent) -> jnp.ndarray:
    """Converts the `Transform3D` into a rank-1 array with values normalized."""

    tx = utils.normalize(self.trans_x, trans_extents.trans_x)
    ty = utils.normalize(self.trans_y, trans_extents.trans_y)

    r = utils.normalize(self.rot_rad, trans_extents.rot_rad)
    s = utils.normalize(self.scale, trans_extents.scale)
 
    return  jnp.concatenate(( tx.reshape((-1)),
                              ty.reshape((-1)),
                              r.reshape((-1)),
                              s.reshape((-1)),
                            ))


  @classmethod
  def from_normalized_array(cls, state_array: jnp.ndarray,
                                 num_objects: int,
                                 extents: TransformExtent)->'Transform':
    """Converts a normalized rank-1 array into `Transform3D`."""
    nt = num_objects

    tx = utils.unnormalize(state_array[0*nt:1*nt], extents.trans_x)
    ty = utils.unnormalize(state_array[1*nt:2*nt], extents.trans_y)

    r = utils.unnormalize(state_array[2*nt:3*nt], extents.rot_rad)
    s = utils.unnormalize(state_array[3*nt:4*nt], extents.scale)

    return Transform(tx, ty, r, s)


def transform_coordinates(xy: jnp.ndarray,
                          transform: Transform,
                          )->jnp.ndarray:
  """Translate coordinates `xy`, followed by rotation and scale as defined by 
    `transform`.
  Args:
    xy: Tensor of (num_pts, 2) containing the coordinates of the points to
      transform.
    transform: A dataclass of `Transform` that defines (num_objects,)
      transforms.
  Returns: A tensor of (num_objects, num_pts, 2) that are the translated
    followed by rotated coordinates followed by scaled.
  """
  num_pts = xy.shape[0]
  xy_t = jnp.zeros((transform.num_objects, num_pts, 2))  #{tpi}

  # o -> objects,  p -> points, i -> index(dim)

  # translate
  xy_t = xy_t.at[:, :, 0].set(xy[:, 0][None, :] - transform.trans_x[:, None]) #{opi}
  xy_t = xy_t.at[:, :, 1].set(xy[:, 1][None, :] - transform.trans_y[:, None])

  # rotate
  xy_t_r = jnp.einsum('opi, oij -> opj', xy_t, transform.rot_matrix)

  # scale
  xy_t_r_s = jnp.einsum('opi, o -> opi', xy_t_r, 1./transform.scale)
  return xy_t_r_s


def init_random_transforms(num_objects: int,
                           extent: TransformExtent,
                           seed: int=27)->Transform:

  # TODO: Use JAX random generator
  rng = np.random.default_rng(seed)
  tx = rng.uniform(extent.trans_x.min, extent.trans_x.max, (num_objects,))
  ty = rng.uniform(extent.trans_y.min, extent.trans_y.max, (num_objects,))
  r = rng.uniform(extent.rot_rad.min, extent.rot_rad.max, (num_objects,))
  s = rng.uniform(extent.scale.min, extent.scale.max, (num_objects,))

  return Transform(trans_x=tx,
                   trans_y=ty,
                   rot_rad=r,
                   scale=s)


def init_grid_transforms(num_objects_x: int,
                         num_objects_y: int,
                         extent: TransformExtent
                         )->Transform:

  num_objects = num_objects_x*num_objects_y
  len_x = np.abs(extent.trans_x.range)
  len_y = np.abs(extent.trans_y.range)
  del_x = len_x/(4*num_objects_x)
  del_y = len_y/(4*num_objects_y)

  cx = extent.trans_x.min + np.linspace(2*del_x, len_x - 2*del_x, num_objects_x)
  cy = extent.trans_y.min + np.linspace(2*del_y, len_y - 2*del_y, num_objects_y)
  [cx_grid, cy_grid] = np.meshgrid(cx, cy)

  mean_rot = 0.5*(extent.rot_rad.min + extent.rot_rad.max)
  rot = mean_rot*np.ones((num_objects,))

  # scale = extent.scale.min*np.ones((num_objects,))
  scale = 0.25*(extent.scale.min + extent.scale.max)*np.ones((num_objects,))

  return Transform(trans_x=cx_grid.reshape(-1),
                   trans_y=cy_grid.reshape(-1),
                   rot_rad=rot.reshape(-1),
                   scale=scale.reshape(-1))