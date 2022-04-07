from ...feature.feature import BaseTextFeature, Trie
from ...config.config import BaseFeatureConfig
from ...utils.registry import Registers
import json
import numpy as np
import tensorlayerx as tlx
from itertools import groupby


@Registers.feature_configs.register
class Wav2Vec2FeatureConfig(BaseFeatureConfig):
    def __init__(self,
                 vocab_file,
                 feature_size=1,
                 sampling_rate=16000,
                 padding_value=0.0,
                 return_attention_mask=False,
                 do_normalize=True,
                 bos_token="<s>",
                 eos_token="</s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 word_delimiter_token="|",
                 do_lower_case=False,
                 speech_max_length=None,
                 text_max_length=None,
                 **kwargs):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        self.vocab_file = vocab_file
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word_delimiter_token = word_delimiter_token
        self.do_lower_case = do_lower_case
        self.speech_max_length = speech_max_length
        self.text_max_length = text_max_length

        super(Wav2Vec2FeatureConfig, self).__init__(**kwargs)


@Registers.features.register
class Wav2Vec2Feature(BaseTextFeature):
    config_class = Wav2Vec2FeatureConfig

    def __init__(
            self,
            config,
            **kwargs
    ):
        self.config = config
        super(Wav2Vec2Feature, self).__init__(config, **kwargs)

        with open(self.config.vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.unique_no_split_tokens = []
        for token in self.encoder.keys():
            if len(token) > 1:
                self.unique_no_split_tokens.append(token)

        self._create_trie(self.unique_no_split_tokens)

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            if hasattr(self, "do_lower_case") and self.do_lower_case and token not in self.all_special_tokens:
                trie.add(token.lower())
            else:
                trie.add(token)
        self.tokens_trie = trie

    def tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer.
        """
        if self.config.do_lower_case:
            text = text.upper()

        return list(text.replace(" ", self.config.word_delimiter_token))

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.config.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        result = self.decoder.get(index, self.config.unk_token)
        return result

    def process_speech(self, speech):
        if self.config.do_normalize:
            input_values = (speech - speech.mean()) / np.sqrt(speech.var() + 1e-7)
        else:
            input_values = speech
        return input_values

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        return self._convert_token_to_id(token)

    def string_to_ids(self, text):
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        return ids

    def __call__(self, speech, text, *args, **kwargs):
        speech = speech.astype(np.float32)
        input_values = self.process_speech(speech)
        # input_values = input_values.astype(np.float32)
        input_ids = self.string_to_ids(text)
        input_ids = np.array(input_ids, dtype=np.int32)
        # return {"input_values": input_values}, {"input_ids": input_ids}
        return input_values, input_ids

    def pad_and_create_pixel_mask(
        self, pixel_values_list
    ):

        max_size = max([speech.shape[0] for speech in pixel_values_list])
        if self.config.speech_max_length:
            if max_size > self.config.speech_max_length:
                max_size = self.config.speech_max_length
        padded_pixel_values = []
        pixel_mask = []
        for pixel_value in pixel_values_list:

            padded_pixel_value = np.zeros(max_size, dtype=np.float32)
            mask = np.zeros(max_size, dtype=bool)
            if pixel_value.shape[0] > max_size:
                padded_pixel_value = pixel_value[:max_size]
                mask = np.ones(max_size, dtype=bool)
            else:
                padded_pixel_value[: pixel_value.shape[0]] = np.copy(pixel_value)
                mask[: pixel_value.shape[0]] = True

            padded_pixel_values.append(padded_pixel_value)
            mask = mask.astype(np.int32)
            pixel_mask.append(mask)

        return padded_pixel_values, pixel_mask

    def pad_ids(self, ids):
        max_size = max([label.shape[0] for label in ids])
        if self.config.text_max_length:
            if max_size > self.config.text_max_length:
                max_size = self.config.text_max_length

        padded_ids = []
        for label in ids:
            padded_id = np.zeros(max_size, dtype=np.int32)
            if label.shape[0] > max_size:
                padded_id = label[:max_size]
            else:
                padded_id[: label.shape[0]] = np.copy(label)
            padded_ids.append(padded_id)

        return padded_ids

    def collate_fn(self, data):
        input_values = [i[0] for i in data]
        padded_pixel_values, pixel_mask = self.pad_and_create_pixel_mask(input_values)

        # inputs = {"inputs": padded_pixel_values, "pixel_mask": pixel_mask}

        labels = [i[1][0] for i in data]
        texts = [i[1][1] for i in data]
        padded_ids = self.pad_ids(labels)
        # labels = {"labels": padded_ids}

        if len(data) >= 2:
            new_data = []
            for i, j, k, l in zip(padded_pixel_values, pixel_mask, padded_ids, texts):
                new_data.append(({"inputs": i, "pixel_mask": j}, {"labels": k, "texts": l}))

            return tlx.dataflow.dataloader.utils.default_collate(new_data)
        else:
            return tlx.dataflow.dataloader.utils.default_convert(({"inputs": np.array(padded_pixel_values),
                                                                   "pixel_mask": np.array(pixel_mask)},
                                                                  {"labels": np.array(padded_ids), "texts": texts}))

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens

    def convert_tokens_to_string(
            self, tokens, group_tokens=True, spaces_between_special_tokens= False):
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        # group same tokens into non-repeating tokens in CTC style decoding
        if group_tokens:
            tokens = [token_group[0] for token_group in groupby(tokens)]

        # filter self.pad_token which is used as CTC-blank token
        filtered_tokens = list(filter(lambda token: token != self.config.pad_token, tokens))

        if spaces_between_special_tokens:
            join_token = " "
        else:
            join_token = ""

        # replace delimiter token
        string = join_token.join(
            [" " if token == self.config.word_delimiter_token else token for token in filtered_tokens]
        ).strip()

        if self.config.do_lower_case:
            string = string.lower()
        return string

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (:obj:`str`): The text to clean up.

        Returns:
            :obj:`str`: The cleaned-up string.
        """
        out_string = (
            out_string.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
        )
        return out_string

    def ids_to_string(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True,
                      group_tokens=True, spaces_between_special_tokens=False):
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        text = self.convert_tokens_to_string(
            result, group_tokens=group_tokens, spaces_between_special_tokens=spaces_between_special_tokens
        )

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text






