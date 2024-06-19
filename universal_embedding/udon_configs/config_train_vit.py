import ml_collections


#these default values can be overwritten by the cmd line args



def get_config():
	"""Returns the ViT experiment configuration."""

	config = ml_collections.ConfigDict()

	config.experiment_name = 'universal-embedding-vit'

	# Dataset.

	config.dataset_name = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"
	config.knn_eval_names = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"

	config.data_dtype_str = 'float32'

	config.dataset_configs = ml_collections.ConfigDict()

	#Sampler configs

	#config.sampling_strategy = "dataset_size" #strategy that samples according to the length of each dataset
	#config.sampling_strategy = "balanced"
	config.sampling_strategy = "round_robin"

	#config.update_sampler = False
	config.update_sampler = True

	config.update_sampler_every_steps = 1000

	config.update_sampler_logit_type="teacher"
	#config.update_sampler_logit_type="universal_student"


	config.classifier = "separate"

	config.count_flops = False #bugged?

	# Model.
	config.model_class = 'udon_vit_with_embedding'
	#config.model_class = 'udon_clip_vit_with_embedding'

	#config.model_type = "S/16"
	config.model_type = "B/16"

	config.model = ml_collections.ConfigDict()

	config.model.representation_size = None #we will always use that as None

	config.model.output_dim = 64 #our chosen embedding dimension

	config.model.classifier = 'token'
	config.model.attention_dropout_rate = 0.0
	config.model.dropout_rate = 0.0

	config.model_dtype_str = 'float32'
	#config.model_dtype_str = 'bfloat16'

	#config.model.positional_embedding = 'none'
	config.model.positional_embedding = 'learned_1d'




	#checkpoints
	config.pretrained_ckpt_dir = 'data/models/'
	
	config.init_ckpt = ''

	# Training.
	config.optimizer = 'adam' #its actually adamw that scenic uses if you use weight decay
	config.optimizer_configs = ml_collections.ConfigDict()
	config.optimizer_configs.beta1 = 0.9
	config.optimizer_configs.beta2 = 0.999
	config.optimizer_configs.weight_decay = 1e-6
	config.explicit_weight_decay = None  # No explicit weight decay
	config.l2_decay_factor = None
	# config.max_grad_norm = 1.0
	config.label_smoothing = None

	config.num_training_epochs = 10 #total number of training epochs

	config.batch_size = 128

	config.eval_batch_size = 1024
	config.knn_eval_batch_size = 2048

	#splits to not do knn during training
	config.disabled_separate_knns = 'train_knn,test_knn'
	config.disabled_merged_knns = 'train_knn,test_knn'

	config.rng_seed = 0

	#config.init_head_bias = -10.0 #not used anywhere by us, can we remove it from here?

	config.loss = ml_collections.ConfigDict()
	config.loss.m = 0.0
	config.loss.scale = 16

	config.loss.transform_logits_type = 'normface'
	#config.loss.transform_logits_type = 'arcface'
	#config.loss.transform_logits_type = 'cosface'

	#UDON configs
	config.model.universal_student_dim=(64,)
	config.model.teacher_dim=(256,)

	config.loss.distill_udon_logits=True
	config.loss.distill_udon_logits_temperature=ml_collections.ConfigDict()
	config.loss.distill_udon_logits_temperature.teacher=0.1
	config.loss.distill_udon_logits_temperature.universal_student=0.1

	config.loss.distill_udon_batch_simils=True

	config.loss.classif_udon_universal_student_weight=1.0
	config.loss.classif_udon_teacher_weight=1.0
	config.loss.distill_udon_logits_weight=1.0
	config.loss.distill_udon_simils_weight=1.0

	config.loss.classif_losses_on="teacher,universal_student"

	config.max_to_keep = 1000

	config.log_eval_steps_frequency = 1 #for knn eval
	config.log_summary_steps_frequency = 10 #for logging training metrics
	config.checkpoint_steps_frequency = 1 #for doing checkpoinint

	# Learning rate.
	config.lr_configs = ml_collections.ConfigDict()
	config.lr_configs.learning_rate_schedule = 'compound'
	config.lr_configs.factors = 'constant'
	config.lr_configs.base_learning_rate = 1e-3 #lr of params_early_train

	config.lr_configs.backbone = ml_collections.ConfigDict()
	
	config.frozen_epochs = 2 #that means for 2 epochs we train only the params_early_train

	config.backbone_learning_rate_multiplier = 1e-2 #multiplies base_learning rate to get lr of the backbone

	config.params_early_train = ['output_projection','teacher_projection_domain','universal_student_projection_domain']


	# kNN
	config.do_knn = True

	config.embedd_to_eval = "universal_student_embedd,teacher_embedd"
	config.universal_embedding_is="universal_student_embedd"

	config.do_final_testing = True

	config.save_descriptors = False

	config.extract_only_descrs = False



	# Logging.
	config.write_summary = True

	config.checkpoint = True
	#config.checkpoint = False  # Do checkpointing.

	config.only_best_checkpoint = True
	#config.only_best_checkpoint = False

	config.debug_train = False  # Debug mode during training.
	config.debug_eval = False  # Debug mode during eval.

	config.eval_dataset_dir = ''
	config.train_dataset_dir = ''

	config.project_feats_knn = True

	#config.descr_save_path = "."
	config.descr_save_path = None

	config.save_neighbors = False

	config.top_k = 5

	config.log_domain_acc = True

	config.log_csv = False

	config.info_files_dir = ''

	return config