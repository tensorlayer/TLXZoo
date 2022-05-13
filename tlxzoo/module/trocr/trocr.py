from .trocr_decoder import *
from .vit import *
import tensorlayerx as tlx


class TrOCR(tlx.nn.Module):
    def __init__(self, hidden_size=768,
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
                 *inputs, **kwargs):
        """
        :param hidden_size: (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder(vit) layers and the pooler layer.
        :param num_hidden_layers: (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the vit encoder.
        :param num_attention_heads: (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the vit encoder.
        :param intermediate_size: (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the vit encoder.
        :param hidden_act: (:obj:`str`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the vit encoder and pooler.
        :param hidden_dropout_prob: (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, vit encoder, and pooler.
        :param attention_probs_dropout_prob: (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the encoder attention probabilities.
        :param initializer_range: (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for encoder initializing all weight matrices.
        :param layer_norm_eps: (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the encoder layer normalization layers.
        :param image_size: (:obj:`int`, `optional`, defaults to :obj:`224`):
            The size (resolution) of each image.
        :param patch_size: (:obj:`int`, `optional`, defaults to :obj:`16`):
            The size (resolution) of each patch.
        :param num_channels: (:obj:`int`, `optional`, defaults to :obj:`3`):
            The number of input channels.
        :param qkv_bias: (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to add a bias to the queries, keys and values.
        :param vocab_size: (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the TrOCR model.
        :param d_model: (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        :param decoder_layers: (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        :param decoder_attention_heads: (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        :param decoder_ffn_dim: (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        :param activation_function: (:obj:`str`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the pooler. If string, :obj:`"gelu"`,
            :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        :param max_position_embeddings: (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        :param dropout: (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, and pooler.
        :param attention_dropout: (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        :param activation_dropout: (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        """
        super().__init__(*inputs, **kwargs)

        self.vit = ViTModel(image_size, patch_size, num_channels, hidden_size, initializer_range, hidden_dropout_prob,
                            num_attention_heads, attention_probs_dropout_prob, intermediate_size, hidden_act,
                            layer_norm_eps, num_hidden_layers, add_pooling_layer=False)

        self.trocr_decoder = TrOCRForCausalLM(dropout, decoder_layerdrop, pad_token_id, d_model, scale_embedding,
                                              vocab_size, use_learned_position_embeddings, max_position_embeddings,
                                              layernorm_embedding, decoder_attention_heads, attention_dropout,
                                              activation_function, activation_dropout, cross_attention_hidden_size,
                                              decoder_ffn_dim, decoder_layers, use_cache)
        self.qkv_bias = qkv_bias
        self.decoder_start_token_id = decoder_start_token_id
        self.classifier_dropout = classifier_dropout
        self.init_std = init_std
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def loss_fn(self, logits, input_ids, attention_mask):
        loss = tlx.losses.cross_entropy_seq_with_mask
        logits = tlx.reshape(logits, shape=(-1, self.vocab_size))

        input_ids_shape = tlx.get_tensor_shape(input_ids)
        labels = tlx.concat([input_ids[:, 1:], tlx.ones([input_ids_shape[0], 1], dtype=input_ids.dtype)], axis=1)
        mask = tlx.concat([attention_mask[:, 1:], tlx.zeros([input_ids_shape[0], 1], dtype=attention_mask.dtype)],
                          axis=1)
        return loss(logits=logits, target_seqs=labels, input_mask=mask)

    def generate_one(self, inputs=None, max_length=64):
        decoder_start_token_id = self.bos_token_id
        start_tokens = decoder_start_token_id * tlx.ones((shape_list(inputs)[0], 1), dtype=tlx.int64)

        outputs = self.vit(inputs)
        encoder_hidden_states = outputs[0]

        while int(start_tokens[0][-1]) != self.eos_token_id and \
                start_tokens.shape[1] < max_length:
            attention_mask = tlx.ones_like(start_tokens)
            logits = self.trocr_decoder(start_tokens, attention_mask, encoder_hidden_states=encoder_hidden_states)
            last_tokens = tlx.argmax(logits, -1)[:, -1:]
            start_tokens = tlx.concat([start_tokens, last_tokens], axis=-1)
        return start_tokens

    def forward(self, inputs, input_ids=None, attention_mask=None):
        outputs = self.vit(inputs)
        encoder_hidden_states = outputs[0]
        logits = self.trocr_decoder(input_ids, attention_mask, encoder_hidden_states=encoder_hidden_states)

        return logits
