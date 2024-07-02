"""Helper functions.

author: Aaditya Chandrasekhar (cs.aaditya@gmail.com)
"""

import dataclasses
import numpy as np
import jax.numpy as jnp


@dataclasses.dataclass
class Extent:
  """
  A class representing a numerical range with minimum and maximum values.
  """
  min: float
  max: float


  @property
  def range(self)->float:
    """
    Calculate the range of the extent.
    
    Returns:
      The range (max - min).
    """
    return self.max - self.min


  @property
  def center(self)->float:
    """
    Calculate the center of the extent.
    
    Returns:
      The center ((min + max) / 2).
    """
    return 0.5*(self.min + self.max)


  def scale(
      self,
      scale_val:float
  ) -> 'Extent':
    """Scale the `Extent`."""
    return Extent(self.min*scale_val, self.max*scale_val)


  def translate(
      self,
      dx:float,
  ) -> 'Extent':
    """Translate the `Extent` by `dx`."""
    return Extent(self.min + dx, self.max + dx)
  
 
  def pad(
      self,
      pad_amount: float,
  ) -> 'Extent':
    """Pad the `Extent` by a specified amount on both sides.

    Args:
      pad_amount: The amount of padding to add on each side.

    Returns:
      A new `Extent` object with the padded boundaries.


    Example:
      Before padding:
          |----Extent----|
          |  min    max  |
          
      After padding by pad_amount:
         |------Extent------|
       |-|--min        max--|-|

      Negative padding example:
        |----Extent-----|
          |--min max--|

      
    """
    return Extent(self.min - pad_amount, self.max + pad_amount)


  


def normalize(x: jnp.ndarray, extent: Extent)->jnp.ndarray:
  """Linearly normalize `x` using `extent` ranges."""
  return (x - extent.min)/extent.range


def unnormalize(x: jnp.ndarray, extent: Extent)->jnp.ndarray:
  """Recover array from linearly normalized `x` using `extent` ranges."""
  return x*extent.range + extent.min


def inverse_sigmoid(y: jnp.ndarray)->jnp.ndarray:
  """The inverse of the sigmoid function.

  The sigmoid function f:x->y is defined as:

           f(x) = 1 / (1 + exp(-x))
  
  The inverse sigmoid function g: y->x is defined as:

           g(y) = ln(y / (1 - y))
  
  For details see https://tinyurl.com/y7mr76hm
  """
  return jnp.log(y / (1. - y))