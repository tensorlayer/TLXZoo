from ...config.config import BaseModelConfig, BaseTaskConfig
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME
from ...utils.registry import Registers


@Registers.model_configs.register
class Wav2Vec2ModelConfig(BaseModelConfig):
    model_type = "wav2vec2"

    def __init__(
            self,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout=0.1,
            activation_dropout=0.1,
            attention_dropout=0.1,
            feat_proj_dropout=0.1,
            feat_quantizer_dropout=0.0,
            layerdrop=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            feat_extract_norm="group",
            feat_extract_activation="gelu",
            feat_extract_dropout=0.0,
            hidden_dropout_prob=0.1,
            conv_dim=(512, 512, 512, 512, 512, 512, 512),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_bias=False,
            num_conv_pos_embeddings=128,
            num_conv_pos_embedding_groups=16,
            do_stable_layer_norm=False,
            apply_spec_augment=True,
            mask_time_prob=0.05,
            mask_time_length=10,
            mask_feature_prob=0.0,
            mask_feature_length=10,
            num_codevectors_per_group=320,
            num_codevector_groups=2,
            contrastive_logits_temperature=0.1,
            num_negatives=100,
            codevector_dim=256,
            proj_codevector_dim=256,
            use_weighted_layer_sum=False,
            classifier_proj_size=256,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = list(conv_dim)
        self.conv_stride = list(conv_stride)
        self.conv_kernel = list(conv_kernel)
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_feat_extract_layers = len(self.conv_dim)
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.feat_proj_dropout = feat_proj_dropout
        self.layerdrop = layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.do_stable_layer_norm = do_stable_layer_norm
        self.use_weighted_layer_sum = use_weighted_layer_sum
        self.classifier_proj_size = classifier_proj_size

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length

        self.num_codevectors_per_group = num_codevectors_per_group
        self.num_codevector_groups = num_codevector_groups
        self.contrastive_logits_temperature = contrastive_logits_temperature
        self.feat_quantizer_dropout = feat_quantizer_dropout
        self.num_negatives = num_negatives
        self.codevector_dim = codevector_dim
        self.proj_codevector_dim = proj_codevector_dim
        self.feat_extract_dropout = feat_extract_dropout
        self.hidden_dropout_prob = hidden_dropout_prob

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(
            weights_path=weights_path,
            **kwargs,
        )


@Registers.task_configs.register
class Wav2Vec2ForAutomaticSpeechRecognitionTaskConfig(BaseTaskConfig):
    task_type = "wav2vec_for_automatic_speech_recognition"
    model_config_type = Wav2Vec2ModelConfig

    def __init__(self,
                 model_config: model_config_type = None,
                 vocab_size=32,
                 final_dropout=0.1,
                 diversity_loss_weight=0.1,
                 ctc_loss_reduction="sum",
                 ctc_zero_infinity=False,
                 weights_path=TASK_WEIGHT_NAME,
                 **kwargs):
        if model_config is None:
            model_config = self.model_config_type()
        self.vocab_size = vocab_size
        self.final_dropout = final_dropout
        self.diversity_loss_weight = diversity_loss_weight
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(Wav2Vec2ForAutomaticSpeechRecognitionTaskConfig, self).__init__(model_config, **kwargs)
