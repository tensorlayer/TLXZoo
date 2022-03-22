from ...config.config import BaseModelConfig, BaseTaskConfig
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME
import os
from ...utils.registry import Registers


@Registers.model_configs.register
class T5Config(BaseModelConfig):
    model_type = "t5"

    def __init__(
            self,
            vocab_size=32128,
            d_model=768,
            d_kv=64,
            d_ff=3072,
            num_layers=12,
            num_decoder_layers=None,
            num_heads=12,
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="relu",
            is_encoder_decoder=True,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.decoder_start_token_id = decoder_start_token_id
        super().__init__(
            weights_path=weights_path,
            **kwargs,
        )


@Registers.task_configs.register
class T5ForConditionalGenerationTaskConfig(BaseTaskConfig):
    task_type = "t5_for_conditional_generation"
    model_config_type = T5Config

    def __init__(self,
                 model_config: model_config_type = None,
                 tie_word_embeddings=True,
                 weights_path=TASK_WEIGHT_NAME,
                 **kwargs):
        if model_config is None:
            model_config = self.model_config_type()
        self.tie_word_embeddings = tie_word_embeddings
        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(T5ForConditionalGenerationTaskConfig, self).__init__(model_config, **kwargs)


@Registers.task_configs.register
class T5ForTextClassificationTaskConfig(BaseTaskConfig):
    task_type = "t5_for_text_classification"
    model_config_type = T5Config

    def __init__(self,
                 model_config: model_config_type = None,
                 weights_path=TASK_WEIGHT_NAME,
                 n_class=2,
                 initializer_factor=1.0,
                 **kwargs):

        if model_config is None:
            model_config = self.model_config_type()

        self.n_class = n_class
        self.initializer_factor = initializer_factor
        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(T5ForTextClassificationTaskConfig, self).__init__(model_config, **kwargs)


@Registers.task_configs.register
class T5ForPairTextClassificationTaskConfig(BaseTaskConfig):
    task_type = "t5_for_text_classification"
    model_config_type = T5Config

    def __init__(self,
                 model_config: model_config_type = None,
                 weights_path=TASK_WEIGHT_NAME,
                 n_class=2,
                 initializer_factor=1.0,
                 **kwargs):

        if model_config is None:
            model_config = self.model_config_type()

        self.n_class = n_class
        self.initializer_factor = initializer_factor
        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(T5ForPairTextClassificationTaskConfig, self).__init__(model_config, **kwargs)


@Registers.task_configs.register
class T5ForTokenClassificationTaskConfig(BaseTaskConfig):
    task_type = "t5_for_text_classification"
    model_config_type = T5Config

    def __init__(self,
                 model_config: model_config_type = None,
                 weights_path=TASK_WEIGHT_NAME,
                 initializer_factor=1.0,
                 n_class=15,
                 **kwargs):

        if model_config is None:
            model_config = self.model_config_type()

        self.n_class = n_class
        self.initializer_factor = initializer_factor
        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(T5ForTokenClassificationTaskConfig, self).__init__(model_config, **kwargs)