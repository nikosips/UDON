"""Utils for K-nearest neighbor evaluation."""

import collections
import copy
from absl import logging
from clu import metric_writers
from flax import jax_utils
import jax
import numpy as np
import jax.numpy as jnp

import functools
import json
from tensorflow.io import gfile
import os

from universal_embedding import datasets
from universal_embedding import utils
from universal_embedding import metrics




class KNNEvaluator:
  """Class for knn evaluation."""

  def __init__(
    self,
    config,
    representation_fn,
    knn_query_batch_size,
    extract_only_descriptors = False,
  ):

    self.config = config
    self.extract_only_descriptors = extract_only_descriptors

    if representation_fn is not None:
      
      self.repr_fn = jax.pmap(
        functools.partial(
          representation_fn, 
        ),
        donate_argnums=(1,), 
        axis_name='batch',
        static_broadcasted_argnums=(2),
      )

    else: #TODO: do I even need this case?

      self.repr_fn = None

    self.knn_query_batch_size = knn_query_batch_size
    self.compute_knn_metrics_fun = self.compute_knn_metrics



  @staticmethod
  def _run_knn(
    k,
    index_descrs, 
    query_descrs
  ):

    all_similarities = jnp.matmul(query_descrs, jnp.transpose(index_descrs))
    similarities_k_sorted, indices_k_sorted = jax.lax.top_k(all_similarities, k)

    return similarities_k_sorted,indices_k_sorted



  def _get_repr(
    self, 
    train_state, 
    data,
    domain_idx,
  ):

    """Compute representation for a dataset split."""
  
    embedding_dict_final = {}

    print("extracting representations")

    for i,batch in enumerate(data):

      embeddings_dict, mask = self.repr_fn(
        train_state, 
        batch,
        domain_idx,
      )

      # We need to unreplicate the output of `lax.all_gather`.
      # Shapes at this point are:
      #   embedding: `[hosts, devices, global_batch, features]`.

      mask = np.array(jax_utils.unreplicate(mask)).astype(bool)
            
      for embed_type in embeddings_dict.keys():

        if embed_type not in embedding_dict_final:
          embedding_dict_final[embed_type] = []
        embedding_dict_final[embed_type].append(np.array(jax_utils.unreplicate(embeddings_dict[embed_type]))[mask])

    for embed_type in embedding_dict_final:
      embedding_dict_final[embed_type] = np.concatenate(embedding_dict_final[embed_type], axis=0)

    print("extracted representations")
  
    return embedding_dict_final



  def compute_knn_metrics(
    self,
    lookup_key,
    query_results, 
    index_results, 
    query_paths,
    index_paths,
    throw_first, 
    top_k, 
    config=None,
    query_labels=None,
    index_labels=None,
    query_domains=None,
    index_domains=None,
    embed_types = None,
  ):
    """Compute knn metrics on the query and index."""


    actual_top_k = top_k
    knn_top_k = 1
    
    if throw_first:
      knn_top_k += 1
      top_k += 1

    results_dict = {}

    for embed_type in embed_types:

      results_dict[embed_type] = {}

      query_emb = query_results[embed_type]
      index_emb = index_results[embed_type]

      num_query = len(query_emb)

      query_emb = np.array(query_emb)
      index_emb, index_labels = np.array(index_emb), np.array(index_labels)

      logging.info(f'num query embedding: {num_query}')
      logging.info(f'num index embedding: {len(index_emb)}')
      logging.info(f'embedding dimension: {query_emb.shape[-1]}')

      #we are performing evaluation with separate query splits per dataset
      #the index can either be separate or merged though
      assert len(np.unique(query_domains)) == 1

      classes_in_index = 0

      for domain in np.unique(index_domains):
        domain_classes = index_labels[np.where(index_domains == domain)[0]]
        classes_in_index += len(np.unique(domain_classes))

      logging.info(f'classes in index: {classes_in_index}')

      index_label_counter = collections.Counter(index_labels[np.where(index_domains == np.unique(query_domains)[0])[0]])

      num_batch = num_query // self.knn_query_batch_size

      if num_query % self.knn_query_batch_size != 0:
        num_batch += 1

      logging.info('num_eval_batch: %d', num_batch)
      num_knn_correct = 0
      mp = 0.0

      pmapped_clf_predict = jax.pmap(
        functools.partial(
          self._run_knn,
          k = top_k,
          index_descrs = index_emb,
        )
      )

      for i in range(num_batch):
        
        batch_queries = query_emb[i* self.knn_query_batch_size : min(
                              (i + 1) * self.knn_query_batch_size, num_query)]

        array_batches,masks = self.split_and_pad(batch_queries)
        masks = masks.astype(bool)

        similarities_k_sorted, indices_k_sorted = pmapped_clf_predict(query_descrs = array_batches)      

        similarities_k_sorted = np.array(similarities_k_sorted[masks])
        indices_k_sorted = np.array(indices_k_sorted[masks])
        
        predicted_positions = indices_k_sorted

        for k in range(
            i * self.knn_query_batch_size,
            min((i + 1) * self.knn_query_batch_size, num_query),
        ): #for every query in the batch of queries
          
          m = k - (i * self.knn_query_batch_size)

          nearest = [
              (index_labels[j], index_domains[j], similarities_k_sorted[m,l]) #was j instead of l
              for l,j in enumerate(predicted_positions[m])
          ]
          
          #R@1 calc
          pred_label, pred_domain = (
              nearest[knn_top_k - 1][0],
              nearest[knn_top_k - 1][1],
          )

          query_label, query_domain = query_labels[k], query_domains[k]

          num_knn_correct += metrics.universal_classif_accuracy(
            pred_label,
            pred_domain,
            query_label,
            query_domain,
          )

          if isinstance(query_label,int):          

            num_index_label = index_label_counter[query_label]

          else:

            #will not work for the case that query_label is not a list
            num_index_label = np.sum(
                [index_label_counter[label] for label in query_label]
            )
          
          # num_index_label == how many from the same class are in the index
          # if the query set is the index set this value is +1 of the correct
          # that must be used as n_q in the definition

          if throw_first:
            num_true_index_label = num_index_label - 1
          else:
            num_true_index_label = num_index_label



          mp_sample,relevances = metrics.universal_mmp_at_k(
            top_k,
            actual_top_k,
            num_index_label,
            num_true_index_label,
            nearest,
            query_label,
            query_domain,
            throw_first,
          )

          mp+= mp_sample


      results_dict[embed_type]['dimensionality'] = query_emb.shape[-1]
      results_dict[embed_type]['mean_acc'] = np.round(np.array(num_knn_correct * 1.0 / num_query),3)
      results_dict[embed_type]['mean_mmp_at_k'] = np.round(np.array(mp / num_query),3)

    return results_dict




  def run_separate_knn(
    self,
    train_state,
    base_dir,
    dataset_names,
    batch_size,
    disabled_knns='',
    all_descriptors_dict = None,
    config = None,
  ):

    formated_total_results = {}

    """Runs seperate knn evals defined in the dataset."""
    dataset = datasets.get_knn_eval_datasets(
      self.config,
      base_dir, 
      dataset_names.split(','), 
      batch_size,
      disabled_knns = disabled_knns,
    )

    knn_info = dataset.knn_info
    
    if all_descriptors_dict is None:
      all_descriptors_dict = {}

    for dataset_name, info in knn_info['knn_setup'].items(): #for every dataset  
     
      if dataset_name in all_descriptors_dict:
        inference_lookup = all_descriptors_dict[dataset_name]
      else:
        inference_lookup = {}


      for knn_name, val in info.items(): #for every split, dont like the name val here
        #either train, test or validation split

        if knn_name in disabled_knns:
          logging.info(
              'Skipping disabled knn %s in separate knn eval.', knn_name
          )
          continue

        for split in [val['query'], val['index']]:
          #we do not extract twice for the same split
          lookup_key = datasets.dataset_lookup_key(dataset_name, split)
          
          if split not in inference_lookup and train_state is not None:
            logging.info('Getting inference for %s.', lookup_key)

            inference_lookup[split] = self._get_repr(
                train_state, 
                knn_info[lookup_key],
                domain_idx = (config.knn_eval_names.split(",")).index(dataset_name),
                
            )

          else:
            print(f"descriptors already extracted for {lookup_key}")
            

      if dataset_name in all_descriptors_dict:
        all_descriptors_dict[dataset_name].update(inference_lookup)
      else:
        all_descriptors_dict[dataset_name] = inference_lookup

      if self.extract_only_descriptors:
        continue

      for knn_name, val in info.items():

        if knn_name in disabled_knns:
          logging.info(
              'Skipping disabled knn %s in separate knn eval', knn_name
          )
          continue
        
        logging.info(
            'Running knn on dataset %s with split %s in separate knn eval.',
            dataset_name,
            knn_name,
        )
        
        query_split = val['query']
        index_split = val['index']

        query_labels = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, query_split)]["labels"]
        index_labels = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, index_split)]["labels"]

        query_domains = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, query_split)]["domains"]
        index_domains = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, index_split)]["domains"]

        query_paths = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, query_split)]["paths"]
        index_paths = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, index_split)]["paths"]

        logging.info(
            'query_split: %s, index_split: %s.', query_split, index_split
        )
        
        throw_1st = True if query_split == index_split else False
        
        logging.info('throw_1st: %s.', throw_1st)


        results_dict = self.compute_knn_metrics_fun(
          datasets.dataset_lookup_key(dataset_name, query_split),
          inference_lookup[query_split],
          inference_lookup[index_split],
          query_paths,
          index_paths,
          throw_1st,
          dataset.meta_data['top_k'],
          config,
          query_labels,
          index_labels,
          query_domains,
          index_domains,
          embed_types = config.embedd_to_eval.split(","),
        )

        formated_total_results_temp = (
          self.format_results(          
            results_dict,
            config,
            dataset_name,
            knn_name,
            dataset,
            separate = True,
          )
        )

        formated_total_results = merge(dict(formated_total_results),formated_total_results_temp)

    #average the dicts here
    average_results = self.average_datasets(formated_total_results)
    formated_total_results = merge(dict(formated_total_results),average_results)

    return formated_total_results, all_descriptors_dict




  def run_merged_knn(
    self,
    train_state,
    base_dir,
    query_dataset_names,
    index_dataset_names,
    batch_size,
    disabled_knns='',
    all_descriptors_dict = None,
    config = None,
  ):

    formated_total_results = {}

    """Runs  knn evals using a common database."""
    query_dataset_names = set(query_dataset_names.split(','))
    index_dataset_names = set(index_dataset_names.split(','))

    
    assert query_dataset_names.issubset(
        index_dataset_names
    ), 'Please make sure query set names are a subset of index set names.'

    dataset_names = query_dataset_names.union(index_dataset_names)

    dataset = datasets.get_knn_eval_datasets(
      self.config,
      base_dir,
      list(dataset_names),
      batch_size,
      disabled_knns = disabled_knns,
    )
    
    #i need to get the dataset names here according to their domain label

    knn_info = dataset.knn_info

    lookup_keys = set()

    for dataset_name, info in knn_info['knn_setup'].items(): #for each dataset
    
      for knn_name, val in info.items(): #for each split
        
        if knn_name in disabled_knns:
          logging.info('Skipping disabled knn %s in merged knn eval', knn_name)
          continue

        if dataset_name in index_dataset_names:
          lookup_key = datasets.dataset_lookup_key(dataset_name, val['index'])
          lookup_keys.add(lookup_key)
        if dataset_name in query_dataset_names:
          lookup_key = datasets.dataset_lookup_key(dataset_name, val['query'])
          lookup_keys.add(lookup_key)


    inference_lookup = {}

    if all_descriptors_dict is not None:
      for dataset_name in all_descriptors_dict.keys():
        for split in all_descriptors_dict[dataset_name].keys():
          inference_lookup[f"{dataset_name}:{split}"] = all_descriptors_dict[dataset_name][split]


    for lookup_key in lookup_keys:
      
      if lookup_key not in inference_lookup and train_state is not None:

        logging.info('Getting inference for %s in merged knn eval.', lookup_key)

        inference_lookup[lookup_key] = self._get_repr(
            train_state, 
            knn_info[lookup_key],
            domain_idx = (config.knn_eval_names.split(",")).index(lookup_key.split(":")[0]),
        )

      else:

        print(f"representations already extracted for {lookup_key}")
        

    if not self.extract_only_descriptors:

      # Build up index (merge all indexes)
      index_lookup = {}
      index_paths = {}
      
      index_labels = {}
      index_domains = {}

      for dataset_name in index_dataset_names: #for each dataset in the index ones
        
        knn_setup = knn_info['knn_setup'][dataset_name]
        
        for knn_name, val in knn_setup.items(): #for each split in this dataset
          
          if knn_name in disabled_knns:
            continue

          #get the index part only
          lookup_key = datasets.dataset_lookup_key(dataset_name, val['index'])

          #get the labels here from knn info
          if knn_name not in index_lookup:

            index_lookup[knn_name] = copy.deepcopy(inference_lookup[lookup_key])
                    
            index_labels[knn_name] = knn_info['json_data'][lookup_key]["labels"]
            index_domains[knn_name] = knn_info['json_data'][lookup_key]["domains"]

            index_paths[knn_name] = knn_info['json_data'][lookup_key]["paths"]
          
          else:
          
            #concat embeds, domains, labels
            lookup_result = inference_lookup[lookup_key]

            for descr_type in index_lookup[knn_name]:
              
              index_lookup[knn_name][descr_type] = np.concatenate(
                  (index_lookup[knn_name][descr_type], lookup_result[descr_type]), axis=0
              )
            
            index_labels[knn_name] = np.concatenate(
                (index_labels[knn_name], knn_info['json_data'][lookup_key]["labels"]), axis=0
            )

            index_domains[knn_name] = np.concatenate(
                (index_domains[knn_name], knn_info['json_data'][lookup_key]["domains"]), axis=0
            )

            index_paths[knn_name] = np.concatenate(
                (index_paths[knn_name], knn_info['json_data'][lookup_key]["paths"]), axis=0
            )

      # Running knn.
      for dataset_name in query_dataset_names:
        #for each query split
        
        knn_setup = knn_info['knn_setup'][dataset_name]
        
        for knn_name, val in knn_setup.items(): #either train,val or test

          if knn_name in disabled_knns:
            continue
          
          logging.info(
              'Running knn on dataset %s with split %s in merged knn eval.',
              dataset_name,
              knn_name,
          )

          query_split = val['query']
          index_split = val['index']


          logging.info(
              'query_split: %s, index_split: %s.', query_split, index_split
          )
          
          throw_1st = True if query_split == index_split else False

          lookup_key = datasets.dataset_lookup_key(dataset_name, val['query'])

          query_labels = knn_info['json_data'][lookup_key]["labels"]
          index_labels_copy = index_labels[knn_name]

          query_paths = knn_info['json_data'][lookup_key]["paths"]
          index_paths_copy = index_paths[knn_name]

          query_domains = knn_info['json_data'][lookup_key]["domains"]
          index_domains_copy = index_domains[knn_name]

          results_dict = self.compute_knn_metrics_fun(
            lookup_key,
            inference_lookup[lookup_key],
            index_lookup[knn_name],
            query_paths,
            index_paths_copy,
            throw_1st,
            dataset.meta_data['top_k'],
            config,
            query_labels,
            index_labels_copy,
            query_domains,
            index_domains_copy,
            embed_types = config.embedd_to_eval.split(","),
          )

          formated_total_results_temp = (
            self.format_results(          
              results_dict,
              config,
              dataset_name,
              knn_name,
              dataset,
              separate = False,
            )
          )

          formated_total_results = merge(dict(formated_total_results),formated_total_results_temp)

    #average the dicts here
    average_results = self.average_datasets(formated_total_results)
    formated_total_results = merge(dict(formated_total_results),average_results)

    #turn inference_lookup to all descriptors_dict format
    all_descriptors_dict = {}

    for lookup_key in inference_lookup:
      dataset_name,split = lookup_key.split(":")
      if dataset_name not in all_descriptors_dict:
        all_descriptors_dict[dataset_name] = {}
      all_descriptors_dict[dataset_name][split] = inference_lookup[lookup_key]

    return formated_total_results, all_descriptors_dict



  def format_results(
    self,
    results_dict,
    config,
    dataset_name,
    knn_name,
    dataset,
    separate,
    ):

    if separate:
      keyword = ':separate:'
    
    else:
      keyword = ':common:'
    
    knn_results, mp_results = {}, {}

    dimensionality = {}

    for embed_type in config.embedd_to_eval.split(","):
      
      if embed_type not in knn_results:
        knn_results[embed_type] = {}

      if embed_type not in dimensionality:
        dimensionality[embed_type] = {}
      
      if embed_type not in mp_results:
        mp_results[embed_type] = {}

      knn_results[embed_type][dataset_name + keyword + knn_name + ':top_1'] = results_dict[embed_type]['mean_acc']
      mp_results[embed_type][dataset_name + keyword + knn_name + f':mp_{dataset.meta_data["top_k"]}'] = results_dict[embed_type]['mean_mmp_at_k']
      dimensionality[embed_type][dataset_name + keyword + knn_name] = results_dict[embed_type]['dimensionality']

    formated_total_results = {}
    formated_total_results['knn_results'] = knn_results
    formated_total_results['mp_results'] = mp_results 
    formated_total_results['dimensionality'] = dimensionality

    return formated_total_results



  def average_datasets(
    self,
    results,
  ):

    new_results = {}

    for metric in results:

      if metric not in new_results:
        new_results[metric] = {}

      for embed_type in results[metric]:

        if embed_type not in new_results[metric]:
          new_results[metric][embed_type] = {}

        for knn_split in ['train_knn','val_knn','test_knn']:

          results_list = []
          for dataset_key,dataset_value in results[metric][embed_type].items():
            
            if knn_split in dataset_key:
              split_dataset_key = dataset_key #small hack to do it
              results_list.append(dataset_value)

          if len(results_list) == 0:
            continue

          new_results[metric][embed_type][split_dataset_key.replace(split_dataset_key.split(":")[0],"average")] = np.round(np.mean(results_list),3)

    return new_results



  def log_knn_summary(
    self, 
    writer: metric_writers.MetricWriter, 
    step, 
    results,
  ):
  
    """
    Call `writer` with a descriptive string and the results.
    """
  
    scalars = {}

    for embed_type, embed_type_result in results['knn_results'].items():
      for knn_name,result in embed_type_result.items():
        scalars[f'knn/{embed_type}/{knn_name}'] = result

    for embed_type, embed_type_result in results['mp_results'].items():
      for mp_name,result in embed_type_result.items():
        scalars[f'mp/{embed_type}/{mp_name}'] = result

    for embed_type, embed_type_result in results['dimensionality'].items():
      for pR_name,result in embed_type_result.items():
        scalars[f'dimensionality/{embed_type}/{pR_name}'] = result

    writer.write_scalars(step, scalars)



  @staticmethod
  def split_and_pad(array):

    #first split almost uniformly
    list_of_arrays = np.array_split(array,jax.local_device_count())
    max_len = max([np.shape(subarray)[0] for subarray in list_of_arrays])

    #then pad everything to the max size
    new_list = []
    masks_list = []
    for subarray in list_of_arrays:

      mask = np.ones(subarray.shape[0])
      pad_len = max_len-subarray.shape[0]

      if pad_len == 0:
        new_list.append(subarray)
        masks_list.append(mask)

      elif pad_len<0:
        raise Exception("error")

      else:
        padded_subarray = np.pad(subarray,((0,pad_len),(0,0)),"constant")
        new_list.append(padded_subarray)
        padded_mask = np.pad(mask,(0,pad_len),"constant")
        masks_list.append(padded_mask)

    #then stack them to np array
    stacked_arrays = np.stack(new_list)
    masks = np.stack(masks_list)

    return stacked_arrays,masks



def knn_step(
  knn_evaluator,
  train_state,
  config,
  train_dir,
  step,
  writer,
  load_descrs = True,
):

  knn_dataset_names = config.knn_eval_names.split(',')

  if config.descr_save_path is not None:
    descr_base_dir = config.descr_save_path
  else:
    descr_base_dir = train_dir

  descr_base_dir = os.path.join(descr_base_dir,"descriptors")

  os.makedirs(descr_base_dir,exist_ok = True)

  descr_save_path = os.path.join(descr_base_dir,f"descriptors_step_{step}.pkl")
  neigh_save_path = os.path.join(descr_base_dir,f"neighbors_step_{step}.pkl")

  if gfile.exists(descr_save_path) and load_descrs:

    with gfile.GFile(descr_save_path, 'rb') as f:
      all_descriptors_dict = json.load(f)
      print(f"some descriptors loaded")

  else:
    all_descriptors_dict = None

  knn_datasets_dir = config.eval_dataset_dir

  knn_dataset_names = ','.join(knn_dataset_names)
  logging.info('Running knn evals using separate database.')

  knn_datasets = datasets
    
  results, all_descriptors_dict = knn_evaluator.run_separate_knn(
    train_state,
    knn_datasets_dir, 
    knn_dataset_names,
    config.get('eval_batch_size', config.batch_size),
    config.get('disabled_separate_knns', ''),
    all_descriptors_dict,
    config,
  )

  logging.info(
      'Running knn evals using common database made of %s.',
      knn_dataset_names,
  )

  merged_results,all_descriptors_dict = knn_evaluator.run_merged_knn(
    train_state,
    knn_datasets_dir,
    knn_dataset_names,
    knn_dataset_names,
    config.get('eval_batch_size', config.batch_size),
    config.get('disabled_merged_knns', ''),
    all_descriptors_dict,
    config,
  )

  results = merge(dict(results),merged_results)
  
  if not config.extract_only_descrs:

    print(f"step: {step}, results : {results}")

    if config.write_summary:

      if config.universal_embedding_is is not None:
        #duplicate the knn entries of the universal embedding
        results = create_universal_embedding_entry(
          results,
          config,
        )

      knn_evaluator.log_knn_summary(
        writer=writer, 
        step=step, 
        results=results,
      )


    if config.save_neighbors:

      #numpy encoder might not be needed here.
      with gfile.GFile(neigh_save_path, mode='wb') as data:
        data.write(json.dumps(results_visuals,cls = utils.NumpyEncoder))
        print(f"neighbors file complete: {neigh_save_path}")


  if config.save_descriptors:
    
    descr_to_save_dict = {}

    for dataset in all_descriptors_dict.keys():
      descr_to_save_dict[dataset] = {}
      for split in all_descriptors_dict[dataset].keys():
        descr_to_save_dict[dataset][split] = {}
        for embed_type in all_descriptors_dict[dataset][split]:
          if embed_type in config.embedd_to_eval.split(","):
            descr_to_save_dict[dataset][split][embed_type] = all_descriptors_dict[dataset][split][embed_type]
    
    with gfile.GFile(descr_save_path, mode='wb') as data:
      data.write(json.dumps(descr_to_save_dict,cls = utils.NumpyEncoder))
      print(f"descriptors file complete: {descr_save_path}")  


  return results



def create_universal_embedding_entry(
  results_dict,
  config,
):
  
  results_dict['knn_results']['universal'] = results_dict['knn_results'][config.universal_embedding_is]
  results_dict['mp_results']['universal'] = results_dict['mp_results'][config.universal_embedding_is]
  results_dict['dimensionality']['universal'] = results_dict['dimensionality'][config.universal_embedding_is]
  
  return results_dict



def log_epoch_dict(values_dict, step, csv_path):
  
  #traverse the dict until you see scalars as values of the last nested dict

  flattened_dict = flatten(values_dict)
  file_exists = os.path.isfile(csv_path)

  with gfile.GFile(csv_path, mode='w') as csvfile:
    writer = csv.writer(csvfile)  
    if not file_exists:
      writer.writerow(["step"] + list(flattened_dict.keys()))
    writer.writerow([step] + list(flattened_dict.values()))



def flatten(d, parent_key='', sep='/'):
  items = []
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, MutableMapping):
      items.extend(flatten(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)



def merge(a: dict, b: dict, path=[]):
  for key in b:
    if key in a:
      if isinstance(a[key], dict) and isinstance(b[key], dict):
        merge(a[key], b[key], path + [str(key)])
      elif a[key] != b[key]:
        import ipdb; ipdb.set_trace()
        raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
    else:
      a[key] = b[key]
  return a