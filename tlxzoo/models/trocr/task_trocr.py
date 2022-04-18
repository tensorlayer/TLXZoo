from ...task.task import BaseForOpticalCharacterRecognition
from .vit import *
from .trocr_decoder import *
from ...utils.output import BaseTaskOutput
import tensorlayerx as tlx


class TrOCRForOpticalCharacterRecognition(BaseForOpticalCharacterRecognition):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.vit = ViTModel(self.config.model_config.vit_config, add_pooling_layer=False)
        self.trocr_decoder = TrOCRForCausalLM(self.config.model_config.trocr_decoder_config)

    def _load_huggingface_tf_weight(self, weight_path):
        import h5py
        file = h5py.File(weight_path, "r")

        names = []
        for w in self.all_weights:
            name = w.name
            coder = name.split("/")[0]
            huggingface_weight_name = f"{coder}/tf_vi_t_for_image_classification_4/" + name
            huggingface_weight_name = huggingface_weight_name.replace("conv/filters:0", "conv/kernel:0")
            huggingface_weight_name = huggingface_weight_name.replace("projection/filters:0", "projection/kernel:0")
            huggingface_weight_name = huggingface_weight_name.replace("biases:0", "bias:0")
            huggingface_weight_name = huggingface_weight_name.replace("weights:0", "kernel:0")

            if huggingface_weight_name not in file:
                print(f"{huggingface_weight_name} do not init.")
                continue
            names.append(huggingface_weight_name)
            w.assign(file[huggingface_weight_name])

        for root_name, g in file.items():
            for _, weights_dirs in g.attrs.items():
                for i in weights_dirs:
                    name = root_name + "/" + str(i)
                    if name not in names:
                        print(f"{name} do not use.")

        return self

    def loss_fn(self, logits, input_ids, attention_mask):
        loss = tlx.losses.cross_entropy_seq_with_mask
        # logits = tlx.softmax(logits, axis=-1)
        logits = tlx.reshape(logits, shape=(-1, self.config.model_config.trocr_decoder_config.vocab_size))

        input_ids_shape = tlx.get_tensor_shape(input_ids)
        labels = tlx.concat([input_ids[:, 1:], tlx.ones([input_ids_shape[0], 1], dtype=input_ids.dtype)], axis=1)
        mask = tlx.concat([attention_mask[:, 1:], tlx.zeros([input_ids_shape[0], 1], dtype=attention_mask.dtype)], axis=1)
        return loss(logits=logits, target_seqs=labels, input_mask=mask)

    def generate_one(self, inputs=None, max_length=64):
        decoder_start_token_id = self.config.model_config.trocr_decoder_config.bos_token_id
        start_tokens = decoder_start_token_id * tlx.ones((shape_list(inputs)[0], 1), dtype=tlx.int64)

        outputs = self.vit(inputs)
        encoder_hidden_states = outputs[0]

        while int(start_tokens[0][-1]) != self.config.model_config.trocr_decoder_config.eos_token_id and \
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

        return BaseTaskOutput(logits=logits)

