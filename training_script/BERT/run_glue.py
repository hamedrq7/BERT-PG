"""
options:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH, --model-name-or-path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from
                        huggingface.co/models (default: None)
  --config_name CONFIG_NAME, --config-name CONFIG_NAME
                        Pretrained config name or path if not the same as
                        model_name (default: None)
  --tokenizer_name TOKENIZER_NAME, --tokenizer-name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as
                        model_name (default: None)
  --cache_dir CACHE_DIR, --cache-dir CACHE_DIR
                        Where do you want to store the pretrained models
                        downloaded from huggingface.co (default: None)
  --use_fast_tokenizer [USE_FAST_TOKENIZER], --use-fast-tokenizer [USE_FAST_TOKENIZER]
                        Whether to use one of the fast tokenizer (backed by
                        the tokenizers library) or not. (default: True)
  --no_use_fast_tokenizer, --no-use-fast-tokenizer
                        Whether to use one of the fast tokenizer (backed by
                        the tokenizers library) or not. (default: False)
  --model_revision MODEL_REVISION, --model-revision MODEL_REVISION
                        The specific model version to use (can be a branch
                        name, tag name or commit id). (default: main)
  --use_auth_token [USE_AUTH_TOKEN], --use-auth-token [USE_AUTH_TOKEN]
                        Will use the token generated when running
                        `transformers-cli login` (necessary to use this script
                        with private models). (default: False)
  --hidden_dropout_prob HIDDEN_DROPOUT_PROB, --hidden-dropout-prob HIDDEN_DROPOUT_PROB
                        hidden_dropout_prob in ALBERT. (default: 0.1)
  --attention_probs_dropout_prob ATTENTION_PROBS_DROPOUT_PROB, --attention-probs-dropout-prob ATTENTION_PROBS_DROPOUT_PROB
                        attention_probs_dropout_prob in ALBERT. (default: 0.0)
  --task_name TASK_NAME, --task-name TASK_NAME
                        The name of the task to train on: cola, mnli, mrpc,
                        qnli, qqp, rte, sst2, stsb, wnli (default: None)
  --max_seq_length MAX_SEQ_LENGTH, --max-seq-length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        tokenization. Sequences longer than this will be
                        truncated, sequences shorter will be padded. (default:
                        128)
  --overwrite_cache [OVERWRITE_CACHE], --overwrite-cache [OVERWRITE_CACHE]
                        Overwrite the cached preprocessed datasets or not.
                        (default: False)
  --pad_to_max_length [PAD_TO_MAX_LENGTH], --pad-to-max-length [PAD_TO_MAX_LENGTH]
                        Whether to pad all samples to `max_seq_length`. If
                        False, will pad the samples dynamically when batching
                        to the maximum length in the batch. (default: True)
  --no_pad_to_max_length, --no-pad-to-max-length
                        Whether to pad all samples to `max_seq_length`. If
                        False, will pad the samples dynamically when batching
                        to the maximum length in the batch. (default: False)
  --max_train_samples MAX_TRAIN_SAMPLES, --max-train-samples MAX_TRAIN_SAMPLES
                        For debugging purposes or quicker training, truncate
                        the number of training examples to this value if set.
                        (default: None)
  --max_val_samples MAX_VAL_SAMPLES, --max-val-samples MAX_VAL_SAMPLES
                        For debugging purposes or quicker training, truncate
                        the number of validation examples to this value if
                        set. (default: None)
  --max_test_samples MAX_TEST_SAMPLES, --max-test-samples MAX_TEST_SAMPLES
                        For debugging purposes or quicker training, truncate
                        the number of test examples to this value if set.
                        (default: None)
  --train_file TRAIN_FILE, --train-file TRAIN_FILE
                        A csv or a json file containing the training data.
                        (default: None)
  --validation_file VALIDATION_FILE, --validation-file VALIDATION_FILE
                        A csv or a json file containing the validation data.
                        (default: None)
  --test_file TEST_FILE, --test-file TEST_FILE
                        A csv or a json file containing the test data.
                        (default: None)
  --output_dir OUTPUT_DIR, --output-dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written. Defaults to
                        'trainer_output' if not provided. (default: None)
  --overwrite_output_dir [OVERWRITE_OUTPUT_DIR], --overwrite-output-dir [OVERWRITE_OUTPUT_DIR]
                        Overwrite the content of the output directory. Use
                        this to continue training if output_dir points to a
                        checkpoint directory. (default: False)
  --do_train [DO_TRAIN], --do-train [DO_TRAIN]
                        Whether to run training. (default: False)
  --do_eval [DO_EVAL], --do-eval [DO_EVAL]
                        Whether to run eval on the dev set. (default: False)
  --do_predict [DO_PREDICT], --do-predict [DO_PREDICT]
                        Whether to run predictions on the test set. (default:
                        False)
  --eval_strategy {no,steps,epoch}, --eval-strategy {no,steps,epoch}
                        The evaluation strategy to use. (default: no)
  --prediction_loss_only [PREDICTION_LOSS_ONLY], --prediction-loss-only [PREDICTION_LOSS_ONLY]
                        When performing evaluation and predictions, only
                        returns the loss. (default: False)
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE, --per-device-train-batch-size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per device accelerator core/CPU for
                        training. (default: 8)
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE, --per-device-eval-batch-size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size per device accelerator core/CPU for
                        evaluation. (default: 8)
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE, --per-gpu-train-batch-size PER_GPU_TRAIN_BATCH_SIZE
                        Deprecated, the use of `--per_device_train_batch_size`
                        is preferred. Batch size per GPU/TPU core/CPU for
                        training. (default: None)
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE, --per-gpu-eval-batch-size PER_GPU_EVAL_BATCH_SIZE
                        Deprecated, the use of `--per_device_eval_batch_size`
                        is preferred. Batch size per GPU/TPU core/CPU for
                        evaluation. (default: None)
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS, --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass. (default: 1)
  --eval_accumulation_steps EVAL_ACCUMULATION_STEPS, --eval-accumulation-steps EVAL_ACCUMULATION_STEPS
                        Number of predictions steps to accumulate before
                        moving the tensors to the CPU. (default: None)
  --eval_delay EVAL_DELAY, --eval-delay EVAL_DELAY
                        Number of epochs or steps to wait for before the first
                        evaluation can be performed, depending on the
                        eval_strategy. (default: 0)
  --torch_empty_cache_steps TORCH_EMPTY_CACHE_STEPS, --torch-empty-cache-steps TORCH_EMPTY_CACHE_STEPS
                        Number of steps to wait before calling
                        `torch.<device>.empty_cache()`.This can help avoid
                        CUDA out-of-memory errors by lowering peak VRAM usage
                        at a cost of about [10{'option_strings': ['--
                        torch_empty_cache_steps', '--torch-empty-cache-
                        steps'], 'dest': 'torch_empty_cache_steps', 'nargs':
                        None, 'const': None, 'default': None, 'type': 'int',
                        'choices': None, 'required': False, 'help': 'Number of
                        steps to wait before calling
                        `torch.<device>.empty_cache()`.This can help avoid
                        CUDA out-of-memory errors by lowering peak VRAM usage
                        at a cost of about [10% slower performance](https://gi
                        thub.com/huggingface/transformers/issues/31372).If
                        left unset or set to None, cache will not be
                        emptied.', 'metavar': None, 'container':
                        <argparse._ArgumentGroup object at 0x7c9544f6ff20>,
                        'prog': 'run_glue.py'}lower performance](https://githu
                        b.com/huggingface/transformers/issues/31372).If left
                        unset or set to None, cache will not be emptied.
                        (default: None)
  --learning_rate LEARNING_RATE, --learning-rate LEARNING_RATE
                        The initial learning rate for AdamW. (default: 5e-05)
  --weight_decay WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        Weight decay for AdamW if we apply some. (default:
                        0.0)
  --adam_beta1 ADAM_BETA1, --adam-beta1 ADAM_BETA1
                        Beta1 for AdamW optimizer (default: 0.9)
  --adam_beta2 ADAM_BETA2, --adam-beta2 ADAM_BETA2
                        Beta2 for AdamW optimizer (default: 0.999)
  --adam_epsilon ADAM_EPSILON, --adam-epsilon ADAM_EPSILON
                        Epsilon for AdamW optimizer. (default: 1e-08)
  --max_grad_norm MAX_GRAD_NORM, --max-grad-norm MAX_GRAD_NORM
                        Max gradient norm. (default: 1.0)
  --num_train_epochs NUM_TRAIN_EPOCHS, --num-train-epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform. (default:
                        3.0)
  --max_steps MAX_STEPS, --max-steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs. (default: -1)
  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,cosine_warmup_with_min_lr,warmup_stable_decay}, --lr-scheduler-type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,cosine_warmup_with_min_lr,warmup_stable_decay}
                        The scheduler type to use. (default: linear)
  --lr_scheduler_kwargs LR_SCHEDULER_KWARGS, --lr-scheduler-kwargs LR_SCHEDULER_KWARGS
                        Extra parameters for the lr_scheduler such as
                        {'num_cycles': 1} for the cosine with hard restarts.
                        (default: {})
  --warmup_ratio WARMUP_RATIO, --warmup-ratio WARMUP_RATIO
                        Linear warmup over warmup_ratio fraction of total
                        steps. (default: 0.0)
  --warmup_steps WARMUP_STEPS, --warmup-steps WARMUP_STEPS
                        Linear warmup over warmup_steps. (default: 0)
  --log_level {detail,debug,info,warning,error,critical,passive}, --log-level {detail,debug,info,warning,error,critical,passive}
                        Logger log level to use on the main node. Possible
                        choices are the log levels as strings: 'debug',
                        'info', 'warning', 'error' and 'critical', plus a
                        'passive' level which doesn't set anything and lets
                        the application set the level. Defaults to 'passive'.
                        (default: passive)
  --log_level_replica {detail,debug,info,warning,error,critical,passive}, --log-level-replica {detail,debug,info,warning,error,critical,passive}
                        Logger log level to use on replica nodes. Same choices
                        and defaults as ``log_level`` (default: warning)
  --log_on_each_node [LOG_ON_EACH_NODE], --log-on-each-node [LOG_ON_EACH_NODE]
                        When doing a multinode distributed training, whether
                        to log once per node or just once on the main node.
                        (default: True)
  --no_log_on_each_node, --no-log-on-each-node
                        When doing a multinode distributed training, whether
                        to log once per node or just once on the main node.
                        (default: False)
  --logging_dir LOGGING_DIR, --logging-dir LOGGING_DIR
                        Tensorboard log dir. (default: None)
  --logging_strategy {no,steps,epoch}, --logging-strategy {no,steps,epoch}
                        The logging strategy to use. (default: steps)
  --logging_first_step [LOGGING_FIRST_STEP], --logging-first-step [LOGGING_FIRST_STEP]
                        Log the first global_step (default: False)
  --logging_steps LOGGING_STEPS, --logging-steps LOGGING_STEPS
                        Log every X updates steps. Should be an integer or a
                        float in range `[0,1)`. If smaller than 1, will be
                        interpreted as ratio of total training steps.
                        (default: 500)
  --logging_nan_inf_filter [LOGGING_NAN_INF_FILTER], --logging-nan-inf-filter [LOGGING_NAN_INF_FILTER]
                        Filter nan and inf losses for logging. (default: True)
  --no_logging_nan_inf_filter, --no-logging-nan-inf-filter
                        Filter nan and inf losses for logging. (default:
                        False)
  --save_strategy {no,steps,epoch,best}, --save-strategy {no,steps,epoch,best}
                        The checkpoint save strategy to use. (default: steps)
  --save_steps SAVE_STEPS, --save-steps SAVE_STEPS
                        Save checkpoint every X updates steps. Should be an
                        integer or a float in range `[0,1)`. If smaller than
                        1, will be interpreted as ratio of total training
                        steps. (default: 500)
  --save_total_limit SAVE_TOTAL_LIMIT, --save-total-limit SAVE_TOTAL_LIMIT
                        If a value is passed, will limit the total amount of
                        checkpoints. Deletes the older checkpoints in
                        `output_dir`. When `load_best_model_at_end` is
                        enabled, the 'best' checkpoint according to
                        `metric_for_best_model` will always be retained in
                        addition to the most recent ones. For example, for
                        `save_total_limit=5` and
                        `load_best_model_at_end=True`, the four last
                        checkpoints will always be retained alongside the best
                        model. When `save_total_limit=1` and
                        `load_best_model_at_end=True`, it is possible that two
                        checkpoints are saved: the last one and the best one
                        (if they are different). Default is unlimited
                        checkpoints (default: None)
  --save_safetensors [SAVE_SAFETENSORS], --save-safetensors [SAVE_SAFETENSORS]
                        Use safetensors saving and loading for state dicts
                        instead of default torch.load and torch.save.
                        (default: True)
  --no_save_safetensors, --no-save-safetensors
                        Use safetensors saving and loading for state dicts
                        instead of default torch.load and torch.save.
                        (default: False)
  --save_on_each_node [SAVE_ON_EACH_NODE], --save-on-each-node [SAVE_ON_EACH_NODE]
                        When doing multi-node distributed training, whether to
                        save models and checkpoints on each node, or only on
                        the main one (default: False)
  --save_only_model [SAVE_ONLY_MODEL], --save-only-model [SAVE_ONLY_MODEL]
                        When checkpointing, whether to only save the model, or
                        also the optimizer, scheduler & rng state.Note that
                        when this is true, you won't be able to resume
                        training from checkpoint.This enables you to save
                        storage by not storing the optimizer, scheduler & rng
                        state.You can only load the model using
                        from_pretrained with this option set to True.
                        (default: False)
  --restore_callback_states_from_checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT], --restore-callback-states-from-checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT]
                        Whether to restore the callback states from the
                        checkpoint. If `True`, will override callbacks passed
                        to the `Trainer` if they exist in the checkpoint.
                        (default: False)
  --no_cuda [NO_CUDA], --no-cuda [NO_CUDA]
                        This argument is deprecated. It will be removed in
                        version 5.0 of 🤗 Transformers. (default: False)
  --use_cpu [USE_CPU], --use-cpu [USE_CPU]
                        Whether or not to use cpu. If left to False, we will
                        use the available torch device/backend
                        (cuda/mps/xpu/hpu etc.) (default: False)
  --use_mps_device [USE_MPS_DEVICE], --use-mps-device [USE_MPS_DEVICE]
                        This argument is deprecated. `mps` device will be used
                        if available similar to `cuda` device. It will be
                        removed in version 5.0 of 🤗 Transformers (default:
                        False)
  --seed SEED           Random seed that will be set at the beginning of
                        training. (default: 42)
  --data_seed DATA_SEED, --data-seed DATA_SEED
                        Random seed to be used with data samplers. (default:
                        None)
  --jit_mode_eval [JIT_MODE_EVAL], --jit-mode-eval [JIT_MODE_EVAL]
                        Whether or not to use PyTorch jit trace for inference
                        (default: False)
  --bf16 [BF16]         Whether to use bf16 (mixed) precision instead of
                        32-bit. Requires Ampere or higher NVIDIA architecture
                        or using CPU (use_cpu) or Ascend NPU. This is an
                        experimental API and it may change. (default: False)
  --fp16 [FP16]         Whether to use fp16 (mixed) precision instead of
                        32-bit (default: False)
  --fp16_opt_level FP16_OPT_LEVEL, --fp16-opt-level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3']. See details at
                        https://nvidia.github.io/apex/amp.html (default: O1)
  --half_precision_backend {auto,apex,cpu_amp}, --half-precision-backend {auto,apex,cpu_amp}
                        The backend to be used for half precision. (default:
                        auto)
  --bf16_full_eval [BF16_FULL_EVAL], --bf16-full-eval [BF16_FULL_EVAL]
                        Whether to use full bfloat16 evaluation instead of
                        32-bit. This is an experimental API and it may change.
                        (default: False)
  --fp16_full_eval [FP16_FULL_EVAL], --fp16-full-eval [FP16_FULL_EVAL]
                        Whether to use full float16 evaluation instead of
                        32-bit (default: False)
  --tf32 TF32           Whether to enable tf32 mode, available in Ampere and
                        newer GPU architectures. This is an experimental API
                        and it may change. (default: None)
  --local_rank LOCAL_RANK, --local-rank LOCAL_RANK
                        For distributed training: local_rank (default: -1)
  --ddp_backend {nccl,gloo,mpi,ccl,hccl,cncl,mccl}, --ddp-backend {nccl,gloo,mpi,ccl,hccl,cncl,mccl}
                        The backend to be used for distributed training
                        (default: None)
  --tpu_num_cores TPU_NUM_CORES, --tpu-num-cores TPU_NUM_CORES
                        TPU: Number of TPU cores (automatically passed by
                        launcher script) (default: None)
  --tpu_metrics_debug [TPU_METRICS_DEBUG], --tpu-metrics-debug [TPU_METRICS_DEBUG]
                        Deprecated, the use of `--debug tpu_metrics_debug` is
                        preferred. TPU: Whether to print debug metrics
                        (default: False)
  --debug DEBUG [DEBUG ...]
                        Whether or not to enable debug mode. Current options:
                        `underflow_overflow` (Detect underflow and overflow in
                        activations and weights), `tpu_metrics_debug` (print
                        debug metrics on TPU). (default: None)
  --dataloader_drop_last [DATALOADER_DROP_LAST], --dataloader-drop-last [DATALOADER_DROP_LAST]
                        Drop the last incomplete batch if it is not divisible
                        by the batch size. (default: False)
  --eval_steps EVAL_STEPS, --eval-steps EVAL_STEPS
                        Run an evaluation every X steps. Should be an integer
                        or a float in range `[0,1)`. If smaller than 1, will
                        be interpreted as ratio of total training steps.
                        (default: None)
  --dataloader_num_workers DATALOADER_NUM_WORKERS, --dataloader-num-workers DATALOADER_NUM_WORKERS
                        Number of subprocesses to use for data loading
                        (PyTorch only). 0 means that the data will be loaded
                        in the main process. (default: 0)
  --dataloader_prefetch_factor DATALOADER_PREFETCH_FACTOR, --dataloader-prefetch-factor DATALOADER_PREFETCH_FACTOR
                        Number of batches loaded in advance by each worker. 2
                        means there will be a total of 2 * num_workers batches
                        prefetched across all workers. (default: None)
  --past_index PAST_INDEX, --past-index PAST_INDEX
                        If >=0, uses the corresponding part of the output as
                        the past state for next step. (default: -1)
  --run_name RUN_NAME, --run-name RUN_NAME
                        An optional descriptor for the run. Notably used for
                        trackio, wandb, mlflow comet and swanlab logging.
                        (default: None)
  --disable_tqdm DISABLE_TQDM, --disable-tqdm DISABLE_TQDM
                        Whether or not to disable the tqdm progress bars.
                        (default: None)
  --remove_unused_columns [REMOVE_UNUSED_COLUMNS], --remove-unused-columns [REMOVE_UNUSED_COLUMNS]
                        Remove columns not required by the model when using an
                        nlp.Dataset. (default: True)
  --no_remove_unused_columns, --no-remove-unused-columns
                        Remove columns not required by the model when using an
                        nlp.Dataset. (default: False)
  --label_names LABEL_NAMES [LABEL_NAMES ...], --label-names LABEL_NAMES [LABEL_NAMES ...]
                        The list of keys in your dictionary of inputs that
                        correspond to the labels. (default: None)
  --load_best_model_at_end [LOAD_BEST_MODEL_AT_END], --load-best-model-at-end [LOAD_BEST_MODEL_AT_END]
                        Whether or not to load the best model found during
                        training at the end of training. When this option is
                        enabled, the best checkpoint will always be saved. See
                        `save_total_limit` for more. (default: False)
  --metric_for_best_model METRIC_FOR_BEST_MODEL, --metric-for-best-model METRIC_FOR_BEST_MODEL
                        The metric to use to compare two different models.
                        (default: None)
  --greater_is_better GREATER_IS_BETTER, --greater-is-better GREATER_IS_BETTER
                        Whether the `metric_for_best_model` should be
                        maximized or not. (default: None)
  --ignore_data_skip [IGNORE_DATA_SKIP], --ignore-data-skip [IGNORE_DATA_SKIP]
                        When resuming training, whether or not to skip the
                        first epochs and batches to get to the same training
                        data. (default: False)
  --fsdp FSDP           Whether or not to use PyTorch Fully Sharded Data
                        Parallel (FSDP) training (in distributed training
                        only). The base option should be `full_shard`,
                        `shard_grad_op` or `no_shard` and you can add CPU-
                        offload to `full_shard` or `shard_grad_op` like this:
                        full_shard offload` or `shard_grad_op offload`. You
                        can add auto-wrap to `full_shard` or `shard_grad_op`
                        with the same syntax: full_shard auto_wrap` or
                        `shard_grad_op auto_wrap`. (default: None)
  --fsdp_min_num_params FSDP_MIN_NUM_PARAMS, --fsdp-min-num-params FSDP_MIN_NUM_PARAMS
                        This parameter is deprecated. FSDP's minimum number of
                        parameters for Default Auto Wrapping. (useful only
                        when `fsdp` field is passed). (default: 0)
  --fsdp_config FSDP_CONFIG, --fsdp-config FSDP_CONFIG
                        Config to be used with FSDP (Pytorch Fully Sharded
                        Data Parallel). The value is either a fsdp json config
                        file (e.g., `fsdp_config.json`) or an already loaded
                        json file as `dict`. (default: None)
  --fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP, --fsdp-transformer-layer-cls-to-wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP
                        This parameter is deprecated. Transformer layer class
                        name (case-sensitive) to wrap, e.g, `BertLayer`,
                        `GPTJBlock`, `T5Block` .... (useful only when `fsdp`
                        flag is passed). (default: None)
  --accelerator_config ACCELERATOR_CONFIG, --accelerator-config ACCELERATOR_CONFIG
                        Config to be used with the internal Accelerator object
                        initialization. The value is either a accelerator json
                        config file (e.g., `accelerator_config.json`) or an
                        already loaded json file as `dict`. (default: None)
  --parallelism_config PARALLELISM_CONFIG, --parallelism-config PARALLELISM_CONFIG
                        Parallelism configuration for the training run.
                        Requires Accelerate `1.10.1` (default: None)
  --deepspeed DEEPSPEED
                        Enable deepspeed and pass the path to deepspeed json
                        config file (e.g. `ds_config.json`) or an already
                        loaded json file as a dict (default: None)
  --label_smoothing_factor LABEL_SMOOTHING_FACTOR, --label-smoothing-factor LABEL_SMOOTHING_FACTOR
                        The label smoothing epsilon to apply (zero means no
                        label smoothing). (default: 0.0)
  --optim {adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_torch_npu_fused,adamw_apex_fused,adafactor,adamw_anyprecision,adamw_torch_4bit,adamw_torch_8bit,ademamix,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,ademamix_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_ademamix_32bit,paged_ademamix_8bit,paged_lion_32bit,paged_lion_8bit,rmsprop,rmsprop_bnb,rmsprop_bnb_8bit,rmsprop_bnb_32bit,galore_adamw,galore_adamw_8bit,galore_adafactor,galore_adamw_layerwise,galore_adamw_8bit_layerwise,galore_adafactor_layerwise,lomo,adalomo,grokadamw,schedule_free_radam,schedule_free_adamw,schedule_free_sgd,apollo_adamw,apollo_adamw_layerwise,stable_adamw}
                        The optimizer to use. (default: adamw_torch_fused)
  --optim_args OPTIM_ARGS, --optim-args OPTIM_ARGS
                        Optional arguments to supply to optimizer. (default:
                        None)
  --adafactor [ADAFACTOR]
                        Whether or not to replace AdamW by Adafactor.
                        (default: False)
  --group_by_length [GROUP_BY_LENGTH], --group-by-length [GROUP_BY_LENGTH]
                        Whether or not to group samples of roughly the same
                        length together when batching. (default: False)
  --length_column_name LENGTH_COLUMN_NAME, --length-column-name LENGTH_COLUMN_NAME
                        Column name with precomputed lengths to use when
                        grouping by length. (default: length)
  --report_to REPORT_TO, --report-to REPORT_TO
                        The list of integrations to report the results and
                        logs to. (default: None)
  --project PROJECT     The name of the project to use for logging. Currenly,
                        only used by Trackio. (default: huggingface)
  --trackio_space_id TRACKIO_SPACE_ID, --trackio-space-id TRACKIO_SPACE_ID
                        The Hugging Face Space ID to deploy to when using
                        Trackio. Should be a complete Space name like
                        'username/reponame' or 'orgname/reponame', or just
                        'reponame' in which case the Space will be created in
                        the currently-logged-in Hugging Face user's namespace.
                        If `None`, will log to a local directory. Note that
                        this Space will be public unless you set
                        `hub_private_repo=True` or your organization's default
                        is to create private Spaces. (default: trackio)
  --ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS, --ddp-find-unused-parameters DDP_FIND_UNUSED_PARAMETERS
                        When using distributed training, the value of the flag
                        `find_unused_parameters` passed to
                        `DistributedDataParallel`. (default: None)
  --ddp_bucket_cap_mb DDP_BUCKET_CAP_MB, --ddp-bucket-cap-mb DDP_BUCKET_CAP_MB
                        When using distributed training, the value of the flag
                        `bucket_cap_mb` passed to `DistributedDataParallel`.
                        (default: None)
  --ddp_broadcast_buffers DDP_BROADCAST_BUFFERS, --ddp-broadcast-buffers DDP_BROADCAST_BUFFERS
                        When using distributed training, the value of the flag
                        `broadcast_buffers` passed to
                        `DistributedDataParallel`. (default: None)
  --dataloader_pin_memory [DATALOADER_PIN_MEMORY], --dataloader-pin-memory [DATALOADER_PIN_MEMORY]
                        Whether or not to pin memory for DataLoader. (default:
                        True)
  --no_dataloader_pin_memory, --no-dataloader-pin-memory
                        Whether or not to pin memory for DataLoader. (default:
                        False)
  --dataloader_persistent_workers [DATALOADER_PERSISTENT_WORKERS], --dataloader-persistent-workers [DATALOADER_PERSISTENT_WORKERS]
                        If True, the data loader will not shut down the worker
                        processes after a dataset has been consumed once. This
                        allows to maintain the workers Dataset instances
                        alive. Can potentially speed up training, but will
                        increase RAM usage. (default: False)
  --skip_memory_metrics [SKIP_MEMORY_METRICS], --skip-memory-metrics [SKIP_MEMORY_METRICS]
                        Whether or not to skip adding of memory profiler
                        reports to metrics. (default: True)
  --no_skip_memory_metrics, --no-skip-memory-metrics
                        Whether or not to skip adding of memory profiler
                        reports to metrics. (default: False)
  --use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP], --use-legacy-prediction-loop [USE_LEGACY_PREDICTION_LOOP]
                        Whether or not to use the legacy prediction_loop in
                        the Trainer. (default: False)
  --push_to_hub [PUSH_TO_HUB], --push-to-hub [PUSH_TO_HUB]
                        Whether or not to upload the trained model to the
                        model hub after training. (default: False)
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT, --resume-from-checkpoint RESUME_FROM_CHECKPOINT
                        The path to a folder with a valid checkpoint for your
                        model. (default: None)
  --hub_model_id HUB_MODEL_ID, --hub-model-id HUB_MODEL_ID
                        The name of the repository to keep in sync with the
                        local `output_dir`. (default: None)
  --hub_strategy {end,every_save,checkpoint,all_checkpoints}, --hub-strategy {end,every_save,checkpoint,all_checkpoints}
                        The hub strategy to use when `--push_to_hub` is
                        activated. (default: every_save)
  --hub_token HUB_TOKEN, --hub-token HUB_TOKEN
                        The token to use to push to the Model Hub. (default:
                        None)
  --hub_private_repo HUB_PRIVATE_REPO, --hub-private-repo HUB_PRIVATE_REPO
                        Whether to make the repo private. If `None` (default),
                        the repo will be public unless the organization's
                        default is private. This value is ignored if the repo
                        already exists. If reporting to Trackio with
                        deployment to Hugging Face Spaces enabled, the same
                        logic determines whether the Space is private.
                        (default: None)
  --hub_always_push [HUB_ALWAYS_PUSH], --hub-always-push [HUB_ALWAYS_PUSH]
                        Unless `True`, the Trainer will skip pushes if the
                        previous one wasn't finished yet. (default: False)
  --hub_revision HUB_REVISION, --hub-revision HUB_REVISION
                        The revision to use when pushing to the Hub. Can be a
                        branch name, a tag, or a commit hash. (default: None)
  --gradient_checkpointing [GRADIENT_CHECKPOINTING], --gradient-checkpointing [GRADIENT_CHECKPOINTING]
                        If True, use gradient checkpointing to save memory at
                        the expense of slower backward pass. (default: False)
  --gradient_checkpointing_kwargs GRADIENT_CHECKPOINTING_KWARGS, --gradient-checkpointing-kwargs GRADIENT_CHECKPOINTING_KWARGS
                        Gradient checkpointing key word arguments such as
                        `use_reentrant`. Will be passed to
                        `torch.utils.checkpoint.checkpoint` through
                        `model.gradient_checkpointing_enable`. (default: None)
  --include_inputs_for_metrics [INCLUDE_INPUTS_FOR_METRICS], --include-inputs-for-metrics [INCLUDE_INPUTS_FOR_METRICS]
                        This argument is deprecated and will be removed in
                        version 5 of 🤗 Transformers. Use `include_for_metrics`
                        instead. (default: False)
  --include_for_metrics INCLUDE_FOR_METRICS [INCLUDE_FOR_METRICS ...], --include-for-metrics INCLUDE_FOR_METRICS [INCLUDE_FOR_METRICS ...]
                        List of strings to specify additional data to include
                        in the `compute_metrics` function.Options: 'inputs',
                        'loss'. (default: [])
  --eval_do_concat_batches [EVAL_DO_CONCAT_BATCHES], --eval-do-concat-batches [EVAL_DO_CONCAT_BATCHES]
                        Whether to recursively concat
                        inputs/losses/labels/predictions across batches. If
                        `False`, will instead store them as lists, with each
                        batch kept separate. (default: True)
  --no_eval_do_concat_batches, --no-eval-do-concat-batches
                        Whether to recursively concat
                        inputs/losses/labels/predictions across batches. If
                        `False`, will instead store them as lists, with each
                        batch kept separate. (default: False)
  --fp16_backend {auto,apex,cpu_amp}, --fp16-backend {auto,apex,cpu_amp}
                        Deprecated. Use half_precision_backend instead
                        (default: auto)
  --push_to_hub_model_id PUSH_TO_HUB_MODEL_ID, --push-to-hub-model-id PUSH_TO_HUB_MODEL_ID
                        The name of the repository to which push the
                        `Trainer`. (default: None)
  --push_to_hub_organization PUSH_TO_HUB_ORGANIZATION, --push-to-hub-organization PUSH_TO_HUB_ORGANIZATION
                        The name of the organization in with to which push the
                        `Trainer`. (default: None)
  --push_to_hub_token PUSH_TO_HUB_TOKEN, --push-to-hub-token PUSH_TO_HUB_TOKEN
                        The token to use to push to the Model Hub. (default:
                        None)
  --mp_parameters MP_PARAMETERS, --mp-parameters MP_PARAMETERS
                        Used by the SageMaker launcher to send mp-specific
                        args. Ignored in Trainer (default: )
  --auto_find_batch_size [AUTO_FIND_BATCH_SIZE], --auto-find-batch-size [AUTO_FIND_BATCH_SIZE]
                        Whether to automatically decrease the batch size in
                        half and rerun the training loop again each time a
                        CUDA Out-of-Memory was reached (default: False)
  --full_determinism [FULL_DETERMINISM], --full-determinism [FULL_DETERMINISM]
                        Whether to call enable_full_determinism instead of
                        set_seed for reproducibility in distributed training.
                        Important: this will negatively impact the
                        performance, so only use it for debugging. (default:
                        False)
  --torchdynamo TORCHDYNAMO
                        This argument is deprecated, use
                        `--torch_compile_backend` instead. (default: None)
  --ray_scope RAY_SCOPE, --ray-scope RAY_SCOPE
                        The scope to use when doing hyperparameter search with
                        Ray. By default, `"last"` will be used. Ray will then
                        use the last checkpoint of all trials, compare those,
                        and select the best one. However, other options are
                        also available. See the Ray documentation (https://doc
                        s.ray.io/en/latest/tune/api_docs/analysis.html#ray.tun
                        e.ExperimentAnalysis.get_best_trial) for more options.
                        (default: last)
  --ddp_timeout DDP_TIMEOUT, --ddp-timeout DDP_TIMEOUT
                        Overrides the default timeout for distributed training
                        (value should be given in seconds). (default: 1800)
  --torch_compile [TORCH_COMPILE], --torch-compile [TORCH_COMPILE]
                        If set to `True`, the model will be wrapped in
                        `torch.compile`. (default: False)
  --torch_compile_backend TORCH_COMPILE_BACKEND, --torch-compile-backend TORCH_COMPILE_BACKEND
                        Which backend to use with `torch.compile`, passing one
                        will trigger a model compilation. (default: None)
  --torch_compile_mode TORCH_COMPILE_MODE, --torch-compile-mode TORCH_COMPILE_MODE
                        Which mode to use with `torch.compile`, passing one
                        will trigger a model compilation. (default: None)
  --include_tokens_per_second [INCLUDE_TOKENS_PER_SECOND], --include-tokens-per-second [INCLUDE_TOKENS_PER_SECOND]
                        If set to `True`, the speed metrics will include `tgs`
                        (tokens per second per device). (default: False)
  --include_num_input_tokens_seen [INCLUDE_NUM_INPUT_TOKENS_SEEN], --include-num-input-tokens-seen [INCLUDE_NUM_INPUT_TOKENS_SEEN]
                        Whether to track the number of input tokens seen. Can
                        be `'all'` to count all tokens, `'non_padding'` to
                        count only non-padding tokens, or a boolean (`True`
                        maps to `'all'`, `False` to `'no'`). (default: False)
  --neftune_noise_alpha NEFTUNE_NOISE_ALPHA, --neftune-noise-alpha NEFTUNE_NOISE_ALPHA
                        Activates neftune noise embeddings into the model.
                        NEFTune has been proven to drastically improve model
                        performances for instruction fine-tuning. Check out
                        the original paper here:
                        https://huggingface.co/papers/2310.05914 and the
                        original code here:
                        https://github.com/neelsjain/NEFTune. Only supported
                        for `PreTrainedModel` and `PeftModel` classes.
                        (default: None)
  --optim_target_modules OPTIM_TARGET_MODULES, --optim-target-modules OPTIM_TARGET_MODULES
                        Target modules for the optimizer defined in the
                        `optim` argument. Only used for the GaLore optimizer
                        at the moment. (default: None)
  --batch_eval_metrics [BATCH_EVAL_METRICS], --batch-eval-metrics [BATCH_EVAL_METRICS]
                        Break eval metrics calculation into batches to save
                        memory. (default: False)
  --eval_on_start [EVAL_ON_START], --eval-on-start [EVAL_ON_START]
                        Whether to run through the entire `evaluation` step at
                        the very beginning of training as a sanity check.
                        (default: False)
  --use_liger_kernel [USE_LIGER_KERNEL], --use-liger-kernel [USE_LIGER_KERNEL]
                        Whether or not to enable the Liger Kernel for model
                        training. (default: False)
  --liger_kernel_config LIGER_KERNEL_CONFIG, --liger-kernel-config LIGER_KERNEL_CONFIG
                        Configuration to be used for Liger Kernel. When
                        use_liger_kernel=True, this dict is passed as keyword
                        arguments to the `_apply_liger_kernel_to_instance`
                        function, which specifies which kernels to apply.
                        Available options vary by model but typically include:
                        'rope', 'swiglu', 'cross_entropy',
                        'fused_linear_cross_entropy', 'rms_norm', etc. If
                        None, use the default kernel configurations. (default:
                        None)
  --eval_use_gather_object [EVAL_USE_GATHER_OBJECT], --eval-use-gather-object [EVAL_USE_GATHER_OBJECT]
                        Whether to run recursively gather object in a nested
                        list/tuple/dictionary of objects from all devices.
                        (default: False)
  --average_tokens_across_devices [AVERAGE_TOKENS_ACROSS_DEVICES], --average-tokens-across-devices [AVERAGE_TOKENS_ACROSS_DEVICES]
                        Whether or not to average tokens across devices. If
                        enabled, will use all_reduce to synchronize
                        num_tokens_in_batch for precise loss calculation.
                        Reference: https://github.com/huggingface/transformers
                        /issues/34242 (default: True)
  --no_average_tokens_across_devices, --no-average-tokens-across-devices
                        Whether or not to average tokens across devices. If
                        enabled, will use all_reduce to synchronize
                        num_tokens_in_batch for precise loss calculation.
                        Reference: https://github.com/huggingface/transformers
                        /issues/34242 (default: False)
"""

#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import transformers
print(transformers.__version__)
import numpy as np
from datasets import load_dataset, load_metric
import evaluate 

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

os.environ["WANDB_DISABLED"] = "true"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    hidden_dropout_prob: float = field(
        default="0.1",
        metadata={"help": "hidden_dropout_prob in ALBERT."},
    )
    attention_probs_dropout_prob: float = field(
        default="0.0",
        metadata={"help": "attention_probs_dropout_prob in ALBERT."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # parser.add_argument("--local_rank", type=int, default=-1)
    # parser.add_argument("--local-rank", type=int, default=-1)  # Add this line

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print('training_args', training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    os.makedirs(training_args.logging_dir, exist_ok=True)
    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            # logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(training_args.logging_dir, data_args.task_name + '.log'), mode='a', encoding=None, delay=False)
        ],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.INFO)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        transformers.utils.logging.enable_propagation()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    
    # model.args = ModelArguments(model_name_or_path='bert-large-cased', config_name=None, tokenizer_name=None, cache_dir=None, use_fast_tokenizer=True, model_revision='main', use_auth_token=False, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.0)

    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    from transformers.models.bert import BertForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")



    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
        # metric = evaluate.load("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    """
    model 
    BertForSequenceClassification(
    (bert): BertModel(
        (embeddings): BertEmbeddings(
        (word_embeddings): Embedding(28996, 1024, padding_idx=0)
        (position_embeddings): Embedding(512, 1024)
        (token_type_embeddings): Embedding(2, 1024)
        (LayerNorm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder(
        (layer): ModuleList(
            (0-23): 24 x BertLayer(
            (attention): BertAttention(
                (self): BertSdpaSelfAttention(
                (query): Linear(in_features=1024, out_features=1024, bias=True)
                (key): Linear(in_features=1024, out_features=1024, bias=True)
                (value): Linear(in_features=1024, out_features=1024, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
                )
                (output): BertSelfOutput(
                (dense): Linear(in_features=1024, out_features=1024, bias=True)
                (LayerNorm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): BertIntermediate(
                (dense): Linear(in_features=1024, out_features=4096, bias=True)
                (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
                (dense): Linear(in_features=4096, out_features=1024, bias=True)
                (LayerNorm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
        )
        )
        (pooler): BertPooler(
        (dense): Linear(in_features=1024, out_features=1024, bias=True)
        (activation): Tanh()
        )
    )
    (dropout): Dropout(p=0.1, inplace=False)
    (classifier): Linear(in_features=1024, out_features=2, bias=True)
    )
    --------------------------------
    tokenizer 
        BertTokenizerFast(name_or_path='bert-large-cased', vocab_size=28996, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=False, added_tokens_decoder={
        0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    }
    )
    --------------------------------
    config 
    BertConfig {
    "architectures": [
        "BertForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.0,
    "classifier_dropout": null,
    "directionality": "bidi",
    "finetuning_task": "sst2",
    "gradient_checkpointing": false,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "pad_token_id": 0,
    "pooler_fc_size": 768,
    "pooler_num_attention_heads": 12,
    "pooler_num_fc_layers": 3,
    "pooler_size_per_head": 128,
    "pooler_type": "first_token_transform",
    "position_embedding_type": "absolute",
    "transformers_version": "4.57.2",
    "type_vocab_size": 2,
    "use_cache": true,
    "vocab_size": 28996
    }
    --------------------------------
    train_dataset 67349
    eval_dataset 872
    """
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ########################### 
    # import torch
    # # Take some raw samples
    # raw_batch = [train_dataset[i] for i in range(4)]

    # # Pass through datacollator (this mimics what Trainer does)
    # model_batch = data_collator(raw_batch)

    # print(model_batch)

    # print('model_args.model_revision', model_args.model_revision)
    # print(model)

    # vocab_size = tokenizer.vocab_size    
    # batch_size = 4
    # seq_len = 32
    # random_input_ids = torch.randint(
    #     low=0, 
    #     high=vocab_size, 
    #     size=(batch_size, seq_len)
    # )
    # random_attention_mask = torch.ones(
    #     size=(batch_size, seq_len),
    #     dtype=torch.long
    # )
    # outputs = model(
    #     input_ids=random_input_ids.cuda(),
    #     attention_mask=random_attention_mask.cuda()
    # )
    # print(outputs.logits.shape)


    # from torchinfo import summary
    # print(summary(model, (1, 28996)))
    # exit()
    # ########################### 
    
    logger.info(model.config)

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
