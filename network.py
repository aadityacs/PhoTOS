"""Convo-implicit neural network"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Sequence
import flax.linen as nn
import numpy as np


Array = Any


def siren_init(weight_std: float, dtype: Any) -> Callable:
  """
  Initialize the weights for SIREN network.

  Args:
    weight_std: Standard deviation for the weight initialization.
    dtype: Data type of the weights.

  Returns:
    Function to initialize weights.
  """
  def init_fun(key: jax.random.PRNGKey, shape: Sequence[int], dtype=dtype) -> Array:
    if dtype == jnp.dtype(jnp.array([1j])):
      key1, key2 = jax.random.split(key)
      dtype = jnp.dtype(jnp.array([1j]).real)
      a = jax.random.uniform(key1, shape, dtype) * 2 * weight_std - weight_std
      b = jax.random.uniform(key2, shape, dtype) * 2 * weight_std - weight_std
      return a + 1j*b
    else:
      return jax.random.uniform(key, shape, dtype) * 2 * weight_std - weight_std

  return init_fun


class Sine(nn.Module):
  """
  Sine activation function module.
  """
  w0: float = 1.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    inputs = jnp.asarray(inputs, self.dtype)
    return jnp.sin(self.w0 * inputs)


class SirenLayer(nn.Module):
  """
  SIREN layer with sine activation function.

  This layer is part of the SIREN network as described in the paper
  "Implicit Neural Representations with Periodic Activation Functions" 
  by Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein.
  (https://arxiv.org/abs/2006.09661)

  """
  features: int = 32
  w0: float = 1.0
  c: float = 6.0
  is_first: bool = False
  use_bias: bool = True
  act: Callable = jnp.sin
  precision: Any = None
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    inputs = jnp.asarray(inputs, self.dtype)
    input_dim = inputs.shape[-1]

    # Linear projection with init proposed in SIREN paper
    weight_std = (
        (1/input_dim) if self.is_first else jnp.sqrt(self.c/input_dim)/self.w0
    )

    kernel = self.param(
        "kernel", siren_init(weight_std, self.dtype), (input_dim, self.features)
    )
    kernel = jnp.asarray(kernel, self.dtype)

    y = jax.lax.dot_general(
        inputs,
        kernel,
        (((inputs.ndim - 1,), (0,)), ((), ())),
        precision=self.precision,
    )

    if self.use_bias:
      bias = self.param("bias", jax.random.uniform, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias

    return self.act(self.w0 * y)


class Siren(nn.Module):
  """
  SIREN network composed of multiple SIREN layers.

  """
  hidden_dim: int = 30
  output_dim: int = 5
  num_layers: int = 4
  w0: float = 1e1
  w0_first_layer: float = 1e1
  use_bias: bool = True
  final_activation: Callable = lambda x: x  # Identity
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    x = jnp.asarray(inputs, self.dtype)
    
    for layernum in range(self.num_layers - 1):
      is_first = layernum == 0

      x = SirenLayer(
          features=self.hidden_dim,
          w0=self.w0_first_layer if is_first else self.w0,
          is_first=is_first,
          use_bias=self.use_bias,
      )(x)

    # Last layer, with different activation function
    x = SirenLayer(
        features=self.output_dim,
        w0=self.w0,
        is_first=False,
        use_bias=self.use_bias,
        act=self.final_activation,
    )(x)

    return x

class MLP(nn.Module):
  """
  Multi-Layer Perceptron (MLP) network.
  """
  num_layers: int = 3
  hidden_dim: int = 40
  output_dim: int = 1
  activation: Callable = nn.relu
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    x = jnp.asarray(inputs, self.dtype)

    for _ in range(self.num_layers):
      x = nn.Dense(features=self.hidden_dim, dtype=self.dtype)(x)
      x = self.activation(x)

    x = nn.Dense(features=self.output_dim, dtype=self.dtype)(x)
    return x
    

class ConvEncoder(nn.Module):
  """
  Convolutional Encoder to encode SDF images.
  """
  latent_dim: int

  @nn.compact
  def __call__(self, x, key: jax.random.PRNGKey, is_training: bool=True):

   

    #cnn_1
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))


    #cnn_2
    x = nn.Conv(features=8, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    #flatten
    x = x.reshape((x.shape[0], -1))
    
    #mu
    mu = nn.Dense(features=self.latent_dim)(x)

    #sigma
    sigma = jnp.exp(nn.Dense(features=self.latent_dim)(x))

    if is_training:
      eps = jax.random.normal(key, shape=mu.shape)
      z = mu + sigma*eps
    else:
      z = mu

    return z, mu, sigma
  

class ConvoImplicitAutoEncoder(nn.Module):
  """
  Convolutional Implicit Autoencoder.
  """

  latent_dim: int
  implicit_hidden_dim: int
  implicit_num_layers: int
  implicit_siren_freq: float


  def setup(self):
    self.encoder = ConvEncoder(self.latent_dim)
    self.siren_net = Siren(hidden_dim=self.implicit_hidden_dim,
                        num_layers=self.implicit_num_layers,
                        w0=self.implicit_siren_freq,
                        w0_first_layer=self.implicit_siren_freq)
    self.mlp_net = MLP()


  def call_decoder(self,
                   latent_coordns: jnp.ndarray,
                   mesh_xy:jnp.ndarray)->jnp.ndarray:
    """
    Decode the latent coordinates using the decoder network.

    Args:
      latent_coordns: Array of (num_shapes, latent_dim)
      mesh_xy: Array of (num_pixels, num_dim)
  
    Returns: Array of (num_shapes, num_pixels, num_dim)
    """
    num_pix, _ = mesh_xy.shape
    num_shapes, _ = latent_coordns.shape
    siren_coordinates = self.siren_net(mesh_xy)

    concat_dec_input = jnp.concatenate((
      jnp.tile(latent_coordns[:, np.newaxis, :], (1, num_pix, 1)),
      jnp.tile(siren_coordinates[np.newaxis, :, :], (num_shapes, 1, 1))), 
      axis=-1)
    

    return self.mlp_net(concat_dec_input)


  def __call__(self,
              input_imgs: jnp.ndarray,
              mesh_xy: jnp.ndarray,
              key: jnp.array,
              is_training: bool = False):
    """
    Forward pass of the Convolutional Implicit Autoencoder.
    
    Args:
      input_imgs: Array of (num_imgs, num_pix_x, num_pix_y, 1)
      key: PRNGKey for random number generation
      mesh_xy: Array of (num_pix, 2)
      is_training: Whether in training mode or not
    
    Returns: Tuple of (predicted_sdf, enc_mu, enc_sigma, encoded_z)
    
    """

    encoded_z, enc_mu, enc_sigma = self.encoder(input_imgs, key, is_training)
    pred_sdf = self.call_decoder(encoded_z, mesh_xy)
    return pred_sdf, enc_mu, enc_sigma, encoded_z





