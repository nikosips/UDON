from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn


from scenic.model_lib.base_models import model_utils


#taken from optax (updated for our needs)
def kl_divergence(
    log_predictions,
    targets,
):
  """Computes the Kullback-Leibler divergence (relative entropy) loss.

  Measures the information gain achieved if target probability distribution
  would be used instead of predicted probability distribution.

  References:
    [Kullback, Leibler, 1951](https://www.jstor.org/stable/2236703)

  Args:
    log_predictions: Probabilities of predicted distribution with shape [...,
      dim]. Expected to be in the log-space to avoid underflow.
    targets: Probabilities of target distribution with shape [..., dim].
      Expected to be strictly positive.

  Returns:
    Kullback-Leibler divergence of predicted distribution from target
    distribution with shape [...].
  """
  loss = targets * (
    jnp.where(targets == 0, 0, jnp.log(targets)) - log_predictions
  )
  
  return jnp.sum(loss, axis=-1)



def _transform_logits(
    logits,
    one_hot,
    loss_config,
):
  '''
  Transformation of the logits for different softmax margin losses.
  Transform type can be one of : [arcface, normface, cosface].
  '''

  if loss_config.transform_logits_type == "arcface":

    theta_yi = jax.lax.acos(logits * one_hot)
    
    transformed_logits = jax.lax.cos(
        theta_yi + loss_config.m
    ) * one_hot + logits * (1 - one_hot)

  elif loss_config.transform_logits_type == "cosface":  
    transformed_logits = (logits - loss_config.m) * one_hot + logits * (1 - one_hot)

  elif loss_config.transform_logits_type == "normface":  
    transformed_logits = logits
  
  transformed_logits *= loss_config.scale
  
  return transformed_logits



def udon_logits_distillation_loss(
  teacher_logits,
  student_logits,
  config,
):

  #no gradient through teacher
  teacher_logits = jax.lax.stop_gradient(teacher_logits)

  teacher_logits /= config.loss.distill_udon_logits_temperature["teacher"]
  student_logits /= config.loss.distill_udon_logits_temperature["universal_student"]

  teacher_logits = nn.softmax(teacher_logits)
  student_logits = nn.softmax(student_logits)

  loss = jnp.mean(
    kl_divergence(
      jnp.log(student_logits),
      teacher_logits,
    )
  )
  
  return loss



def udon_embedding_distillation_loss(
  teacher_embeddings,
  universal_student_embeddings,
):

  batch_simils_student = jnp.dot(universal_student_embeddings,universal_student_embeddings.T) #student
  batch_simils_teacher = jnp.dot(teacher_embeddings,teacher_embeddings.T) #teacher
  
  batch_simils_teacher = jax.lax.stop_gradient(batch_simils_teacher)

  #make sure that it matches original implem.
  #that was using 4 devices.
  #the mismatch comes from using below loss function.
  coeff = len(jax.devices())/4.0

  return coeff*model_utils.weighted_mean_squared_error(
    batch_simils_student,
    batch_simils_teacher,
  )