import jax
import jax.numpy as jnp
import numpy as np

from scenic.train_lib import train_utils

#TODO: add comments to explain the code



class Sampler():

  def __init__(
    self,
    config,
    dataset_dict,
    total_steps,
  ):

    self.ds_indices_per_step,self.sampling_weights = self.give_ds_indices_per_step(config,dataset_dict,total_steps)
    self.dataset_dict = dataset_dict
    self.total_steps = total_steps
    self.config = config



  def get_next_train_batch(
      self,
      step,
    ):

    #subtract 1 because the first step is 1
    dataset_idx = self.ds_indices_per_step[step-1]
    dataset_name = self.dataset_dict.meta_data["dataset_name"].split(",")[dataset_idx]

    return next(self.dataset_dict.train_iter[dataset_name]),dataset_idx,dataset_name


  def update_ds_indices(
    self,
    train_domain_metrics,
    current_step,
  ):

    #We call it epoch here but its not really epoch, epoch in this context is how many batches until we update the sampling weight
    epoch_domain_loss = {}

    sampler_update_type = self.config.update_sampler_logit_type

    for dataset_name in train_domain_metrics:

      dataset_train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get, train_domain_metrics[dataset_name])

      dataset_train_metrics = train_utils.stack_forest(dataset_train_metrics)

      train_metrics_summary = jax.tree_util.tree_map(lambda x: x.sum(), dataset_train_metrics)

      train_metrics_summary = train_utils.normalize_metrics_summary(train_metrics_summary, 'train')

      dataset_loss = train_metrics_summary[f"{sampler_update_type}_classifier_loss"]

      epoch_domain_loss[dataset_name] = dataset_loss

    unnorm_weights = {}

    for dataset_name in epoch_domain_loss:        
      unnorm_weights[dataset_name] = epoch_domain_loss[dataset_name]

    normalizer = sum(list(unnorm_weights.values()))

    for dataset_name in unnorm_weights:
      self.sampling_weights[dataset_name] = unnorm_weights[dataset_name]/normalizer #add small constant in the denominator for instability issues?

    #in the case that the dataset was not seen in the last 
    #update_sampler steps, because it was not sampled at all because
    #of very small sampling weight,
    #do label smoothing (a small discounting) here by giving them a very small value here

    #renormalize again here in case not all datasets exist in train domain metrics
    if len(train_domain_metrics) != len(self.dataset_dict.meta_data["dataset_samples"]):
      
      normalizer = sum(list(self.sampling_weights.values()))

      for dataset_name in self.sampling_weights:
        self.sampling_weights[dataset_name] = self.sampling_weights[dataset_name]/normalizer #add safety constant in the denominator?


    #after that, create new ds indices
    step_counter = 0
    self.ds_indices_per_step = []

    #put below into a separate method that all use
    for i,(dataset_name,dataset_samples) in enumerate(self.dataset_dict.meta_data["dataset_samples"].items()):
      
      if i == len(self.dataset_dict.meta_data["dataset_samples"]) - 1: 

        #should we do it like that or increase the number of total steps?
        #because the last dataset misses some steps (just a few)
        total_steps_of_ds = self.total_steps - step_counter

        if total_steps_of_ds < 0:
          sys.exit(f"Negative number found here, total steps of ds: {total_steps_of_ds}")
          

      else:
        
        try:
          total_steps_of_ds = int(self.total_steps*self.sampling_weights[dataset_name])
        except:
          import ipdb; ipdb.set_trace()

      #for some reason, total_steps_of_ds became less than 0 

      self.ds_indices_per_step.append(jnp.full((total_steps_of_ds,), i))
      step_counter += total_steps_of_ds

    #check if i have completed the total number of steps or some are missing or if i am more

    self.ds_indices_per_step = jnp.concatenate(self.ds_indices_per_step)
    self.ds_indices_per_step = jax.random.permutation(jax.random.PRNGKey(current_step), self.ds_indices_per_step)




  @staticmethod
  def give_ds_indices_per_step(config,dataset_dict,total_steps):

    print(f"creating the sampling indices")
    print(f"sampling strategy: {config.sampling_strategy}")

    step_counter = 0
    ds_indices_per_step = []

    if config.sampling_strategy == "dataset_size":
      
      sampling_weights = {}
      total_samples = dataset_dict.meta_data["num_train_examples"]
      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        sampling_weights[dataset_name] = dataset_samples/total_samples 


      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        if i == len(dataset_dict.meta_data["dataset_samples"]) - 1:

          total_steps_of_ds = total_steps - step_counter
        else:
          total_steps_of_ds = int(total_steps*sampling_weights[dataset_name])
    
        ds_indices_per_step.append(jnp.full((total_steps_of_ds,), i))
        step_counter += total_steps_of_ds

      ds_indices_per_step = jnp.concatenate(ds_indices_per_step)
      ds_indices_per_step = jax.random.permutation(jax.random.PRNGKey(0), ds_indices_per_step)


    elif config.sampling_strategy == "balanced":
      
      sampling_weights = {}
      total_samples = dataset_dict.meta_data["num_train_examples"]
      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        sampling_weights[dataset_name] = 1/len(dataset_dict.meta_data["dataset_samples"])

      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        if i == len(dataset_dict.meta_data["dataset_samples"]) - 1:

          total_steps_of_ds = total_steps - step_counter
        else:
          total_steps_of_ds = int(total_steps*sampling_weights[dataset_name])
    
        ds_indices_per_step.append(jnp.full((total_steps_of_ds,), i))
        step_counter += total_steps_of_ds

      ds_indices_per_step = jnp.concatenate(ds_indices_per_step)
      ds_indices_per_step = jax.random.permutation(jax.random.PRNGKey(0), ds_indices_per_step)


    elif config.sampling_strategy == "round_robin":
      #by definition sampling weights here are equal

      sampling_weights = {}
      total_samples = dataset_dict.meta_data["num_train_examples"]
      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        sampling_weights[dataset_name] = 1/len(dataset_dict.meta_data["dataset_samples"])


      one_round = jnp.arange(len(dataset_dict.meta_data["dataset_samples"])) 

      times_to_repeat = int(total_steps/len(dataset_dict.meta_data["dataset_samples"]))
      ds_indices_per_step = jnp.tile(one_round,times_to_repeat)

      steps_left = total_steps - len(ds_indices_per_step)

      ds_indices_per_step = jnp.concatenate([ds_indices_per_step,one_round[:steps_left]])


    elif config.sampling_strategy == "specialist_top_steps":

      specialist_top_steps = config.specialist_top_steps
      
      sampling_weights = {}
      total_samples = dataset_dict.meta_data["num_train_examples"]
      
      total_specialist_steps = sum(specialist_top_steps)

      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        sampling_weights[dataset_name] = specialist_top_steps[i]/total_specialist_steps

      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        if i == len(dataset_dict.meta_data["dataset_samples"]) - 1:

          total_steps_of_ds = total_steps - step_counter
        else:
          total_steps_of_ds = int(total_steps*sampling_weights[dataset_name])
    
        ds_indices_per_step.append(jnp.full((total_steps_of_ds,), i))
        step_counter += total_steps_of_ds

      ds_indices_per_step = jnp.concatenate(ds_indices_per_step)
      ds_indices_per_step = jax.random.permutation(jax.random.PRNGKey(0), ds_indices_per_step)


    return ds_indices_per_step,sampling_weights