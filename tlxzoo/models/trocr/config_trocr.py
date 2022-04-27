from ...config.config import BaseModelConfig, BaseTaskConfig
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME
from ...utils.registry import Registers


class VitModelConfig(BaseModelConfig):
    def __init__(
            self,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=384,
            patch_size=16,
            num_channels=3,
            qkv_bias=True,
            **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias

        super().__init__(
            **kwargs,
        )


class TrOCRDecoderModelConfig(BaseModelConfig):
    def __init__(
            self,
            vocab_size=50265,
            d_model=512,
            decoder_layers=6,
            decoder_attention_heads=8,
            decoder_ffn_dim=4096,
            activation_function="gelu",
            max_position_embeddings=128,
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            decoder_start_token_id=2,
            classifier_dropout=0.0,
            init_std=0.02,
            decoder_layerdrop=0.0,
            use_cache=False,
            scale_embedding=False,
            use_learned_position_embeddings=True,
            layernorm_embedding=True,
            cross_attention_hidden_size=768,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = self.hidden_size = d_model
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.activation_function = activation_function
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.decoder_start_token_id = decoder_start_token_id
        self.classifier_dropout = classifier_dropout
        self.init_std = init_std
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding
        self.use_learned_position_embeddings = use_learned_position_embeddings
        self.layernorm_embedding = layernorm_embedding
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.cross_attention_hidden_size = cross_attention_hidden_size

        super().__init__(
            **kwargs,
        )


@Registers.model_configs.register
class TrOCRModelConfig(BaseModelConfig):
    model_type = "trocr"

    def __init__(
            self,
            vit_config=None,
            trocr_decoder_config=None,
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):

        self.vit_config = VitModelConfig() if vit_config is None else vit_config
        self.trocr_decoder_config = TrOCRDecoderModelConfig() if trocr_decoder_config is None else trocr_decoder_config
        super().__init__(
            weights_path=weights_path,
            **kwargs,
        )

    def _post_dict(self, _dict):
        _dict["vit_config"] = _dict["vit_config"].to_dict()
        _dict["trocr_decoder_config"] = _dict["trocr_decoder_config"].to_dict()
        return super(BaseModelConfig, self)._post_dict(_dict)


@Registers.task_configs.register
class TrOCRForOpticalCharacterRecognitionTaskConfig(BaseTaskConfig):
    task_type = "trocr_for_optical_character_recognition"
    model_config_type = TrOCRModelConfig

    def __init__(self,
                 model_config: model_config_type = None,
                 weights_path=TASK_WEIGHT_NAME,
                 **kwargs):
        if model_config is None:
            model_config = self.model_config_type()

        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(TrOCRForOpticalCharacterRecognitionTaskConfig, self).__init__(model_config, **kwargs)