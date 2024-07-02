"""2D structured domain mesher."""

import dataclasses
import numpy as np
import jax.numpy as jnp
import utils

_Ext = utils.Extent


@dataclasses.dataclass
class BoundingBox:
  x: _Ext
  y: _Ext

  @property
  def lx(self)->float:
    """
    Returns the range in the x-dimension.
    
    Returns:
      Range in the x-dimension.
    """
    return self.x.range

  @property
  def ly(self)->float:
    """
    Returns the range in the y-dimension.
    
    Returns:
      Range in the y-dimension.
    """
    return self.y.range
  
  @property
  def diag_length(self)->float:
    """
    Returns the diagonal length of the bounding box.
    
    Returns:
      Diagonal length of the bounding box.
    """
    return jnp.sqrt(self.lx**2 + self.ly**2)


class Mesher:
  def __init__(self, nelx:int, nely:int, bounding_box: BoundingBox):
    """
    Initialize the Mesher object.

    Args:
      nelx (int): Number of elements along the x-dimension.
      nely (int): Number of elements along the y-dimension.
      bounding_box (BoundingBox): Bounding box defining the domain.
    """
    self.num_dim = 2
    self.nelx, self.nely = nelx, nely
    self.num_elems = nelx*nely
    self.bounding_box = bounding_box

    dx, dy = self.bounding_box.lx/nelx, self.bounding_box.ly/nely
    self.elem_size = np.array([dx, dy])
    self.elem_area = dx*dy
    self.domain_volume = self.elem_area*self.num_elems
    self.num_nodes = (nelx+1)*(nely+1)

    [x_grid, y_grid] = np.meshgrid(
      np.linspace(bounding_box.x.min + dx/2., bounding_box.x.max - dx/2., nelx),
      np.linspace(bounding_box.y.min + dy/2., bounding_box.y.max - dy/2., nely))
    self.elem_centers = jnp.stack((x_grid, y_grid)).T.reshape(-1, self.num_dim)

    [x_grid, y_grid] = np.meshgrid(
               np.linspace(self.bounding_box.x.min,
                           self.bounding_box.x.max,
                           nelx+1),
               np.linspace(self.bounding_box.y.min,
                           self.bounding_box.y.max,
                           nely+1),)
    self.node_xy = jnp.stack((x_grid, y_grid)).T.reshape(-1, self.num_dim)


def compute_point_indices_in_box(xy: np.ndarray, bbox: BoundingBox):
  """
  Filters the coordinates in `xy` that are within the bounding box.

  Args:
    xy: An array of shape (num_obj, num_elems, 2) containing the coordinates.
    bbox: Defines the coordinates of the bounding box.

  Returns: A Boolean array of shape (num_obj, num_elems,) with True values for indices
    of `xy` whose coordinates are within the bounding box.
  """

  x_in_box = jnp.logical_and(xy[:, :, 0] >= bbox.x.min,
                             xy[:, :, 0] <= bbox.x.max)
  y_in_box = jnp.logical_and(xy[:, :, 1] >= bbox.y.min,
                             xy[:, :, 1] <= bbox.y.max)
  filtered_indices = jnp.logical_and(x_in_box, y_in_box)
  return filtered_indices