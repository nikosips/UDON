"""Class for all Universal Embedding project classification models."""

import functools
from typing import Dict, Optional, Tuple, Union

from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections

from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models import multilabel_classification_model

from universal_embedding import loss_utils



def classification_metrics_function(
		outputs: Dict,
		batch: base_model.Batch,
		loss_config: ml_collections.ConfigDict,
		config: ml_collections.ConfigDict,
		target_is_multihot: bool = False,
		axis_name: Union[str, Tuple[str, ...]] = 'batch',
) -> Dict[str, Tuple[float, int]]:
	"""Calculates metrics for the multi-label classification task.

	Currently we assume each metric_fn has the API:
		```metric_fn(logits, targets, weights)```
	and returns an array of shape [batch_size]. We also assume that to compute
	the aggregate metric, one should sum across all batches, then divide by the
	total samples seen. In this way we currently only support metrics of the 1/N
	sum f(inputs, targets). Note, the caller is responsible for dividing by
	the normalizer when computing the mean of each metric.

	Args:
	 logits: Output of model in shape [batch, length, num_classes].
	 batch: Batch of data that has 'label' and optionally 'batch_mask'.
	 loss_config: the configuration for the loss.
	 target_is_multihot: If the target is a multi-hot vector.
	 axis_name: List of axes on which we run the pmsum.

	Returns:
		A dict of metrics, in which keys are metrics name and values are tuples of
		(metric, normalizer).
	"""
	evaluated_metrics = {}

	universal_student_logits = outputs['classifier']['universal_student_logits']
	teacher_logits = outputs['classifier']['teacher_logits']

	if target_is_multihot:
		multihot_target = batch['label']
	else:
		# This is to support running a multi-label classification model on
		# single-label classification tasks:    
		multihot_target = common_utils.onehot(batch['label'], teacher_logits.shape[-1])    


	#classification losses  
	classif_losses_dict = {
		'teacher':{
			'logits' : teacher_logits,
			'weight' : config.loss.classif_udon_teacher_weight,
		},

		'universal_student':{
			'logits' : universal_student_logits,
			'weight' : config.loss.classif_udon_universal_student_weight,
		}
	}

	for classif_loss_type in config.loss.classif_losses_on.split(","):

		logits = classif_losses_dict[classif_loss_type]['logits']

		#transform based on type of softmax margin loss
		transformed_logits = loss_utils._transform_logits(logits, multihot_target, config.loss)

		weights = batch.get('batch_mask') #are these needed at all? Can we delete them?

		#logits for prec@1, transformed logits for loss
		evaluated_metrics.update(
			{
			
				f'{classif_loss_type}_classifier_prec@1': model_utils.psum_metric_normalizer(
					(
						model_utils.weighted_top_one_correctly_classified(
								logits, multihot_target, weights
						),
						model_utils.num_examples(logits, multihot_target, weights
						),
					),
					axis_name=axis_name,
				),

				#the loss is masked if we use domain masks
				f'{classif_loss_type}_classifier_loss': model_utils.psum_metric_normalizer(
					(
						model_utils.weighted_unnormalized_softmax_cross_entropy(
								transformed_logits, multihot_target, weights
						),
						model_utils.num_examples(
								transformed_logits, multihot_target, weights
						),
					),
					axis_name=axis_name,
				),
			}
		)


	#distillation losses

	if config.loss.distill_udon_logits:
		
		#distill logits
		evaluated_metrics.update(
			{
				f'batch_logits_distill_loss': model_utils.psum_metric_normalizer(
						(
							loss_utils.udon_logits_distillation_loss(
								teacher_logits,
								universal_student_logits,
								config,
							),
							1.0,
						),
						axis_name=axis_name,
					),
			}
		)


	if config.loss.distill_udon_batch_simils:
		
		#distill batch similarities

		teacher_embeddings=outputs['embeddings']['teacher_embedd']
		universal_student_embeddings=outputs['embeddings']['universal_student_embedd']

		evaluated_metrics.update(
			{
				f'batch_simils_distill_loss': model_utils.psum_metric_normalizer(
						(
							loss_utils.udon_embedding_distillation_loss(
								teacher_embeddings,
								universal_student_embeddings,
							),
							1.0,
						),
						axis_name=axis_name,
					),
			}
		)

	return evaluated_metrics



class UniversalEmbeddingUdonModel(
		multilabel_classification_model.MultiLabelClassificationModel
):
	"""Defines commonalities between all classification models.

	A model is class with three members: get_metrics_fn, loss_fn, & a flax_model.
	get_metrics_fn returns a callable function, metric_fn, that calculates the
	metrics and returns a dictionary. The metric function computes f(x_i, y_i) on
	a minibatch, it has API: ```metric_fn(logits, label, weights).``` The trainer
	will then aggregate and compute the mean across all samples evaluated. loss_fn
	is a function of API loss = loss_fn(logits, batch, model_params=None). This
	model class defines a softmax_cross_entropy_loss with weight decay, where the
	weight decay factor is determined by config.l2_decay_factor. flax_model is
	returned from the build_flax_model function. A typical usage pattern will be:
	``` model_cls =
	model_lib.models.get_model_cls('fully_connected_classification') model =
	model_cls(config, dataset.meta_data) flax_model = model.build_flax_model
	dummy_input = jnp.zeros(input_shape, model_input_dtype) model_state, params =
	flax_model.init(

			rng, dummy_input, train=False).pop('params')
	```
	And this is how to call the model:
	variables = {'params': params, **model_state}
	logits, new_model_state = flax_model.apply(variables, inputs, ...)
	```
	"""


	
	def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
		"""Returns a callable metric function for the model.

		Args:
			split: The split for which we calculate the metrics. It should be one of
				the ['train',  'validation', 'test'].
		Returns: A metric function with the following API: ```metrics_fn(logits,
			batch)```
		"""
		del split  # For all splits, we return the same metric functions.
		return functools.partial(
			classification_metrics_function,
			target_is_multihot=self.dataset_meta_data.get(
				'target_is_onehot', False
			),
			loss_config=self.config.loss, #comment this out?
			config=self.config,
		)



	def loss_function(
			self,
			outputs: Dict,
			batch: base_model.Batch,
			model_params: Optional[jnp.array] = None, #optionally for regularization purposes
	) -> float:
		"""Returns the softmax loss.

		Args:
			logits: Output of model in shape [batch, length, num_classes].
			batch: Batch of data that has 'label' and optionally 'batch_mask'.
			model_params: Parameters of the model, for optionally applying
				regularization.

		Returns:
			Total loss.
		"""

		total_loss=0.0
		total_weights=0.0

		universal_student_logits = outputs['classifier']['universal_student_logits']
		teacher_logits = outputs['classifier']['teacher_logits']

		# logits are cosine similarities at this point
		one_hot_targets = common_utils.onehot(batch['label'], teacher_logits.shape[-1])


		#classification losses  
		classif_losses_dict = {
			'teacher':{
				'logits' : teacher_logits,
				'weight' : self.config.loss.classif_udon_teacher_weight,
			},

			'universal_student':{
				'logits' : universal_student_logits,
				'weight' : self.config.loss.classif_udon_universal_student_weight,
			}
		}


		for classif_loss_type in self.config.loss.classif_losses_on.split(","):

			logit = classif_losses_dict[classif_loss_type]['logits']
			logit_type_weight = classif_losses_dict[classif_loss_type]['weight']

			#transform based on type of softmax margin loss
			transformed_logits = loss_utils._transform_logits(logit, one_hot_targets, self.config.loss)

			sof_ce_loss = model_utils.weighted_softmax_cross_entropy(
				transformed_logits,
				one_hot_targets,
				label_smoothing=self.config.get('label_smoothing'),
			)

			total_loss+=logit_type_weight*sof_ce_loss
			total_weights+=logit_type_weight



		#distillation losses
		
		if self.config.loss.distill_udon_logits:

			#distill logits
			udon_logits_distillation_loss = loss_utils.udon_logits_distillation_loss(
				teacher_logits,
				universal_student_logits,
				self.config,
			)


			total_loss+=self.config.loss.distill_udon_logits_weight*udon_logits_distillation_loss
			total_weights+=self.config.loss.distill_udon_logits_weight


		if self.config.loss.distill_udon_batch_simils:

			#distill batch similarities
			teacher_embeddings=outputs['embeddings']['teacher_embedd']
			universal_student_embeddings=outputs['embeddings']['universal_student_embedd']

			udon_embedding_distillation_loss = loss_utils.udon_embedding_distillation_loss(
				teacher_embeddings,
				universal_student_embeddings,
			)

			total_loss+=self.config.loss.distill_udon_simils_weight*udon_embedding_distillation_loss
			total_weights+=self.config.loss.distill_udon_simils_weight

		
		return total_loss/total_weights