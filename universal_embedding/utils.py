import os
import json
import numpy as np

from tensorflow.io import gfile
import ml_collections

import jax
from flax.training import checkpoints

from universal_embedding import info_utils




def save_descriptors(descr_save_path,all_descriptors_dict):

  with gfile.GFile(descr_save_path, mode='wb') as data:
    data.write(json.dumps(all_descriptors_dict,cls = NumpyEncoder))
    print(f"descriptors file complete: {descr_save_path}")



def calc_train_dependent_config_values(config):

  #model
  if 'clip' in config.model_class:  
    model_configs = info_utils.CLIP_ViT_configs

  else:
    model_configs = info_utils.ViT_configs

  config.model.hidden_size = model_configs[config.model_type]["hidden_size"]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = model_configs[config.model_type]["patches_size"]
  config.model.num_heads = model_configs[config.model_type]["num_heads"]
  config.model.mlp_dim = model_configs[config.model_type]["mlp_dim"]
  config.model.num_layers = model_configs[config.model_type]["num_layers"]


  #checkpoint
  config.pretrained_ckpt = os.path.join(config.pretrained_ckpt_dir, model_configs[config.model_type]["checkpoint"])


  #frequent ops
  config.steps_per_epoch = (info_utils.get_aggregated_size(config.dataset_name) // config.batch_size)

  #number of steps to log knn validation metrics
  config.log_eval_steps = config.steps_per_epoch //config.log_eval_steps_frequency
  
  #number of steps to log train metrics like loss etc.
  config.log_summary_steps = config.steps_per_epoch // config.log_summary_steps_frequency

  config.checkpoint_steps = config.steps_per_epoch // config.checkpoint_steps_frequency


  #optimizer parameters
  if config.frozen_epochs == -1: #case where you want the backbone parameters to stay frozen for the entire training
      
    config.lr_configs.backbone.frozen_steps = (
        config.num_training_epochs * config.steps_per_epoch
    )

  else:

    config.lr_configs.backbone.frozen_steps = (
        config.frozen_epochs * config.steps_per_epoch
    )

  config.lr_configs.backbone.base_learning_rate = config.lr_configs.base_learning_rate * config.backbone_learning_rate_multiplier



def save_best_checkpoint(
    workdir,
    train_state,
):
  """Saves a checkpoint.

  Args:
    workdir: Experiment directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    max_to_keep: The number of checkpoints to keep.
    overwrite: Overwrite existing checkpoint  if a checkpoint at the current or
      a later step already exits (default: False).
    **kwargs: Passed on to flax.training.checkpoints.save_checkpoint.
  """
  if jax.process_index() == 0:
    # Get train state from the first replica.
    checkpoint_state = jax.device_get(train_state)
    checkpoints.save_checkpoint(
        workdir,
        checkpoint_state,
        -1,
        overwrite=True,
    )


def read_config(path):

  with gfile.GFile(path) as f:
    x = json.load(f)
    x = ml_collections.ConfigDict(x)

  return x


def normalize(a,axis=-1,order=2):
    '''Normalize descriptors (l2 normalization by default)
    '''
    l2 = np.linalg.norm(a, order, axis)
    l2[l2==0] = 1

    return a / np.expand_dims(l2, axis)



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)